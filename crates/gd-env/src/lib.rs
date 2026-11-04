pub mod conn;
pub mod game;
pub mod mock;
pub mod vec_env;

pub use conn::Connection;
pub use game::{process_obs, GdEnv, GdEnvConfig, PROC_OBS_DIM};
pub use mock::MockEnv;
pub use vec_env::VecEnv;

use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub success: bool,
}

/// Observations are written into a caller-owned buffer, so the rollout loop never allocates.
pub trait Env {
    fn obs_dim(&self) -> usize;
    fn reset(&mut self, start_x: f32, obs: &mut [f32]) -> Result<()>;
    fn step(&mut self, action: u8, obs: &mut [f32]) -> Result<StepResult>;
    fn close(&mut self) {}
}
