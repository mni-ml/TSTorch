//! Elementwise cuTile kernels.
//!
//! All shapes are flattened to 1D at launch time — the kernels are compiled
//! for a fixed rank of 1.

#[cutile::module]
pub mod elementwise_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx + ty);
    }

    #[cutile::entry()]
    pub fn sub<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx - ty);
    }

    #[cutile::entry()]
    pub fn mul<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx * ty);
    }

    #[cutile::entry()]
    pub fn div<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx / ty);
    }

    #[cutile::entry()]
    pub fn neg<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(zero - tx);
    }

    #[cutile::entry()]
    pub fn mul_scalar<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        s: f32,
    ) {
        let tx = load_tile_like_1d(x, z);
        let s_tile = s.broadcast(z.shape());
        z.store(s_tile * tx);
    }

    /// Fused saxpy: `z = a·x + y`.  Two GMEM reads, one FMA, one GMEM write —
    /// contrast with the two-kernel path (mul_scalar + add) which round-trips
    /// a temporary through HBM.
    #[cutile::entry()]
    pub fn saxpy<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        a: f32,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let a_tile = a.broadcast(z.shape());
        z.store(a_tile * tx + ty);
    }
}

pub use elementwise_kernels::{add, div, mul, mul_scalar, neg, saxpy, sub};
