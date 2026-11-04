use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub success: bool,
}

pub trait Env {
    fn obs_dim(&self) -> usize;
    fn reset(&mut self, start_x: f32, obs: &mut [f32]) -> Result<()>;
    fn step(&mut self, action: u8, obs: &mut [f32]) -> Result<StepResult>;
    fn close(&mut self) {}
}
