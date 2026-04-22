//! Reduction ops on top of the cuTile reduce kernels.
//!
//! * **Global sum / mean** — `sum_all`, `mean_all` drive the two-pass
//!   `sum_block` reduction described below.
//! * **Along-dim sum / mean / max** — `sum_along_dim` & friends collapse a
//!   single axis.  The kernels (`*_along_last`) only reduce the last axis;
//!   the ops layer permutes arbitrary axes to the last position via the
//!   `permute` kernel.
//! * **Broadcast** — `sum_broadcast` expands a `[*, 1]`-shaped reduction
//!   back across a new last dim, matching `sum_broadcast_f32` in the CUDA
//!   backend.
//!
//! Global reductions use a strict two-pass pattern on top of cuTile.
//! Pass 1 (multi-block): launches `sum_block::<PASS1_BLOCK>` with a grid of
//! `ceil(n / PASS1_BLOCK)` blocks.  Each block does an in-tile `reduce_sum`
//! and writes one scalar into a partials buffer — O(n) → O(n / PASS1_BLOCK).
//! Pass 2 (single block): launches the same `sum_block` kernel with grid = 1
//! and a BLOCK const large enough to cover all of pass 1's partials.
//!
//! The last block in pass 1 and the sole block in pass 2 are the usual
//! "partial tile" case: when `n` (resp. `nblocks1`) is not a multiple of
//! the tile size, `Tensor::partition()` returns a zero-padded view so
//! out-of-range tile lanes load 0.0f32 — the identity for sum.  No host
//! tail, no divisibility constraints on n.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::TileKernel;

/// Elements-per-block in pass 1 of the global reduction.
const PASS1_BLOCK: usize = 2048;

/// Candidate pass-2 tile sizes (powers of 2).
const FINAL_BLOCKS: [usize; 12] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

fn pick_final_block(n: usize) -> usize {
    for &b in &FINAL_BLOCKS {
        if b >= n {
            return b;
        }
    }
    panic!(
        "reduce size {n} exceeds max pass-2 tile {}",
        FINAL_BLOCKS[FINAL_BLOCKS.len() - 1]
    )
}

/// Sum over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn sum_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec());
    let rt = runtime();

    if n == 0 {
        return store.from_slice(&[0.0f32], &[1]);
    }
    if n == 1 {
        let host = store.to_host(a);
        return store.from_slice(&host, &[1]);
    }

    let input_1d: Tensor<f32> = store
        .tensor(a)
        .dup()
        .sync_on(&rt.stream)
        .expect("dup input")
        .reshape(&[n])
        .expect("reshape input to 1D");

    let max_final = *FINAL_BLOCKS.last().unwrap();

    if n <= max_final {
        return launch_final_reduce(store, &input_1d, n);
    }

    let nblocks1 = n.div_ceil(PASS1_BLOCK);
    let mut partials = api::zeros::<f32>(&[nblocks1])
        .sync_on(&rt.stream)
        .expect("alloc partials");
    {
        let xv = input_1d.view(&[n]).expect("view input");
        let _ = kernels::sum_block((&mut partials).partition([1]), &xv)
            .generics(vec![PASS1_BLOCK.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_block pass 1");
    }

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

/// Mean over all elements.
pub fn mean_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec()).max(1);
    let sum_id = sum_all(store, a);
    let scaled = crate::ops::elementwise::mul_scalar(store, sum_id, 1.0 / n as f32);
    store.free(sum_id);
    scaled
}

// ---------------------------------------------------------------------------
// Along-dim reductions.
//
// Input shape is viewed as `[outer, dim, inner]`.  For `dim == last` (inner
// = 1) the kernels reduce directly.  For an arbitrary axis, the ops layer
// permutes the target axis to the last before launching, mirroring the CUDA
// backend's fast path.
// ---------------------------------------------------------------------------

const ROW_CANDIDATES: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn pick_row_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

/// Axis normalization: `dim < 0` counts from the end.
fn normalize_axis(dim: i32, rank: usize) -> usize {
    if dim < 0 {
        (rank as i32 + dim) as usize
    } else {
        dim as usize
    }
}

/// Builds a permutation that moves `axis` to the end.
fn perm_axis_to_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).filter(|&i| i != axis).collect();
    perm.push(axis);
    perm
}

/// Builds the inverse of `perm_axis_to_last` — moves the last back to `axis`.
fn perm_last_to_axis(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank - 1).collect();
    perm.insert(axis, rank - 1);
    perm
}

enum ReduceKind {
    Sum,
    Mean,
    Max,
}

fn along_last(
    store: &mut TensorStore,
    a: TensorId,
    kind: ReduceKind,
    keepdim: bool,
) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    assert!(rank >= 1, "reduce: rank must be >= 1");
    let dim = *shape.last().unwrap();
    let outer: usize = shape[..rank - 1].iter().product();
    let bm = pick_row_block(outer.max(1));
    let rt = runtime();

    let mut out_flat = api::zeros::<f32>(&[outer])
        .sync_on(&rt.stream)
        .expect("alloc");
    {
        let xt = store.tensor(a);
        let xv = xt.view(&[outer, dim]).expect("view x");
        let gen = vec![bm.to_string(), dim.to_string()];
        match kind {
            ReduceKind::Sum => {
                let _ = kernels::sum_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("sum_along_last");
            }
            ReduceKind::Mean => {
                let _ = kernels::mean_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("mean_along_last");
            }
            ReduceKind::Max => {
                let _ = kernels::max_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("max_along_last");
            }
        }
    }

    // Final shape: outer shape with an optional trailing 1 for keepdim.
    let mut out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    if keepdim {
        out_shape.push(1);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let logical = out_flat.reshape(&out_shape).expect("reshape");
    store.insert_tensor(logical, out_shape)
}

fn along_dim_impl(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
    kind: ReduceKind,
) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank);
    assert!(axis < rank, "reduce: dim {dim} out of range for rank {rank}");

    if axis == rank - 1 {
        return along_last(store, a, kind, keepdim);
    }

    let perm = perm_axis_to_last(rank, axis);
    let permuted = crate::ops::elementwise::permute(store, a, &perm);
    let reduced = along_last(store, permuted, kind, keepdim);
    store.free(permuted);

    if keepdim {
        // permuted shape: shape[..axis] ++ shape[axis+1..] ++ [shape[axis]]
        // after keepdim reduce: shape[..axis] ++ shape[axis+1..] ++ [1]
        // we want: shape[..axis] ++ [1] ++ shape[axis+1..]
        let inv = perm_last_to_axis(rank, axis);
        let unpermuted = crate::ops::elementwise::permute(store, reduced, &inv);
        store.free(reduced);
        unpermuted
    } else {
        reduced
    }
}

pub fn sum_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Sum)
}

pub fn mean_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Mean)
}

pub fn max_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Max)
}

/// `sum_broadcast(x, dim, size)` — broadcast `x` across a new axis of length
/// `size` inserted at position `dim`.  Used as the forward half of
/// sum-backward (dy → dx = broadcast(dy) to x's original shape).
pub fn sum_broadcast(
    store: &mut TensorStore,
    x: TensorId,
    dim: i32,
    size: usize,
) -> TensorId {
    let shape = store.shape(x).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank + 1);
    assert!(axis <= rank, "sum_broadcast: dim out of range");

    if axis == rank {
        // Expanding the last axis directly — invoke broadcast_last.
        let outer: usize = shape.iter().product::<usize>().max(1);
        let bm = pick_row_block(outer);
        let rt = runtime();
        let mut out = api::zeros::<f32>(&[outer, size])
            .sync_on(&rt.stream)
            .expect("alloc");
        {
            let xt = store.tensor(x);
            let xv = xt.view(&[outer]).expect("view x");
            let _ = kernels::broadcast_last((&mut out).partition([bm, size]), &xv)
                .generics(vec![bm.to_string(), size.to_string()])
                .sync_on(&rt.stream)
                .expect("broadcast_last kernel");
        }
        let mut final_shape: Vec<usize> = shape;
        final_shape.push(size);
        let logical = out.reshape(&final_shape).expect("reshape");
        return store.insert_tensor(logical, final_shape);
    }

    // General case: broadcast to end, then permute the new axis into place.
    let intermediate = sum_broadcast(store, x, rank as i32, size);
    let new_rank = rank + 1;
    let inv = perm_last_to_axis(new_rank, axis);
    let result = crate::ops::elementwise::permute(store, intermediate, &inv);
    store.free(intermediate);
    result
}
