use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

pub type B = Autodiff<NdArray<f32>>;

pub fn burn_smoke_test() {
    let device = Default::default();
    let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let y = (x.clone() * x.clone()).sum();
    let grads = y.backward();
    let gx = x.grad(&grads).expect("no gradient");
    println!("d/dx sum(x^2) = {:?}  (expect [2, 4, 6])", gx.into_data());
}
