pub mod model;
pub mod ppo;

pub use model::ActorCritic;
pub use ppo::{new_ppo, Ppo, PpoConfig, Rollout, UpdateStats};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

/// Training backend (CPU + autodiff), inference backend (CPU only), and device.
pub type B = Autodiff<NdArray<f32>>;
pub type IB = NdArray<f32>;
pub type Dev = NdArrayDevice;

pub fn burn_smoke_test() {
    let device = Dev::default();
    let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let y = (x.clone() * x.clone()).sum();
    let grads = y.backward();
    let gx = x.grad(&grads).expect("no gradient");
    println!("d/dx sum(x^2) = {:?}  (expect [2.0, 4.0, 6.0])", gx.into_data().to_vec::<f32>().unwrap());
}
