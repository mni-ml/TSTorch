//! Gradient utility ops: in-place clip, global L2 norm-squared.
//!
//! `grad_norm_sq` runs the CUB-style two-pass reduction on top of
//! `grad_norm_sq_partial` (pass 1) and `sum_block` (pass 2) — same pattern
//! as `ops::reduce::sum_all`, but with `gradᵢ²` as the per-element value.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

fn pick_block(n: usize) -> usize {
    for &b in &CANDIDATE_BLOCKS {
        if n % b == 0 {
            return b;
        }
    }
    1
}

const PASS1_BLOCK: usize = 2048;
const FINAL_BLOCKS: [usize; 12] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

fn pick_final_block(n: usize) -> usize {
    for &b in &FINAL_BLOCKS {
        if b >= n {
            return b;
        }
    }
    panic!(
        "grad_norm_sq size {n} exceeds max pass-2 tile {}",
        FINAL_BLOCKS[FINAL_BLOCKS.len() - 1]
    )
}

/// In-place `grad *= scale`.  Returns the same `grad` id for convenience.
pub fn grad_clip(store: &mut TensorStore, grad: TensorId, scale: f32) -> TensorId {
    let shape = store.shape(grad).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    {
        let gt = store.tensor_mut(grad);
        let _ = kernels::grad_clip(gt.partition([block]), scale)
            .sync_on(&rt.stream)
            .expect("grad_clip kernel");
    }
    grad
}

/// Returns a freshly-allocated `[1]` scalar tensor with `Σ gradᵢ²`.
pub fn grad_norm_sq(store: &mut TensorStore, grad: TensorId) -> TensorId {
    let n = shape_size(&store.shape(grad).to_vec());
    let rt = runtime();

    if n == 0 {
        return store.from_slice(&[0.0f32], &[1]);
    }

    // Pass 1: per-block partial sum of squares.
    let nblocks1 = n.div_ceil(PASS1_BLOCK);
    let mut partials = api::zeros::<f32>(&[nblocks1])
        .sync_on(&rt.stream)
        .expect("alloc partials");
    {
        let gt = store.tensor(grad);
        let gv = gt.view(&[n]).expect("view grad");
        let _ = kernels::grad_norm_sq_partial((&mut partials).partition([1]), &gv)
            .generics(vec![PASS1_BLOCK.to_string()])
            .sync_on(&rt.stream)
            .expect("grad_norm_sq_partial");
    }

    // Pass 2: single-block sum over partials (zero-padded slack).
    launch_final_reduce(store, &partials, nblocks1)
}

fn launch_final_reduce(store: &mut TensorStore, src: &Tensor<f32>, len: usize) -> TensorId {
    let rt = runtime();
    let block = pick_final_block(len);
    let mut result = api::zeros::<f32>(&[1])
        .sync_on(&rt.stream)
        .expect("alloc result");
    {
        let sv = src.view(&[len]).expect("view src");
        let _ = kernels::sum_block((&mut result).partition([1]), &sv)
            .generics(vec![block.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_block final");
    }
    store.insert_tensor(result, vec![1])
}
