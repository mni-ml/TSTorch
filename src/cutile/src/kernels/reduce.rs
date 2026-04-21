//! Reduction cuTile kernels.
//!
//! `sum_block` drives both passes of the CUB-style strict two-pass global
//! reduction (see `ops/reduce.rs`).  The last block in pass 1 and the sole
//! block in pass 2 handle the "partial tile" case automatically: cuTile's
//! `Tensor::partition()` returns a zero-padded view, so out-of-range tile
//! lanes load the additive identity 0.0f32 and the reduction stays correct
//! for arbitrary `n`.

#[cutile::module]
pub mod reduce_kernels {
    use cutile::core::*;

    /// Reduce a `BLOCK`-sized tile of `x` to a single scalar via
    /// `reduce_sum`, writing it at `pid.0` in `z`.
    #[cutile::entry()]
    pub fn sum_block<const BLOCK: i32>(
        z: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x: Partition<f32, { [BLOCK] }> = x.partition(const_shape![BLOCK]);
        let tile_x: Tile<f32, { [BLOCK] }> = part_x.load([pid.0]);
        let s_scalar: Tile<f32, { [] }> = reduce_sum(tile_x, 0i32);
        let s_one: Tile<f32, { [1] }> = s_scalar.reshape(const_shape![1]);
        z.store(s_one);
    }
}

pub use reduce_kernels::sum_block;
