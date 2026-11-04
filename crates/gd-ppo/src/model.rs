use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Separate actor and critic towers (2 hidden tanh layers each).
/// Actor output: one logit, P(click) = sigmoid(logit). Critic output: state value.
#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    p1: Linear<B>,
    p2: Linear<B>,
    pi: Linear<B>,
    v1: Linear<B>,
    v2: Linear<B>,
    v: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn new(obs_dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            p1: LinearConfig::new(obs_dim, hidden).init(device),
            p2: LinearConfig::new(hidden, hidden).init(device),
            pi: LinearConfig::new(hidden, 1).init(device),
            v1: LinearConfig::new(obs_dim, hidden).init(device),
            v2: LinearConfig::new(hidden, hidden).init(device),
            v: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// x: [batch, obs_dim] -> (logits [batch,1], values [batch,1])
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let a = self.p1.forward(x.clone()).tanh();
        let a = self.p2.forward(a).tanh();
        let logit = self.pi.forward(a);

        let c = self.v1.forward(x).tanh();
        let c = self.v2.forward(c).tanh();
        let value = self.v.forward(c);
        (logit, value)
    }
}
