//! Activation cuTile kernels.

#[cutile::module]
pub mod activation_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn relu<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(zero, tx));
    }

    /// dx = grad where x > 0, else 0.
    #[cutile::entry()]
    pub fn relu_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        grad: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, dx);
        let tg = load_tile_like_1d(grad, dx);
        let zero: Tile<f32, S> = constant(0.0f32, dx.shape());
        let pos = gt_tile(tx, zero);
        dx.store(select(pos, tg, zero));
    }
}

pub use activation_kernels::{relu, relu_backward};
