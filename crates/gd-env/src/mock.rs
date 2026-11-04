//! Toy environment for tests and `--mock` runs. At counter c, the correct action is
//! click iff c % 4 == 3; a wrong action kills you. Reaching `len` is success.
//! start_x = starting counter, so backplay works exactly like on the real level.
use anyhow::Result;

use crate::{Env, StepResult};

pub struct MockEnv {
    c: usize,
    len: usize,
    prev: f32,
}

impl MockEnv {
    pub const OBS_DIM: usize = 8;

    pub fn new(len: usize) -> Self {
        Self { c: 0, len, prev: 0.0 }
    }

    /// Same shape as the real INFO result: ascending snapshot x positions.
    pub fn snapshot_xs(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32).collect()
    }

    fn write_obs(&self, obs: &mut [f32]) {
        obs.fill(0.0);
        obs[self.c % 4] = 1.0;
        obs[4] = self.c as f32 / self.len as f32;
        obs[5] = self.prev;
    }
}

impl Env for MockEnv {
    fn obs_dim(&self) -> usize {
        Self::OBS_DIM
    }

    fn reset(&mut self, start_x: f32, obs: &mut [f32]) -> Result<()> {
        self.c = (start_x.max(0.0) as usize).min(self.len - 1);
        self.prev = 0.0;
        self.write_obs(obs);
        Ok(())
    }

    fn step(&mut self, action: u8, obs: &mut [f32]) -> Result<StepResult> {
        let need = self.c % 4 == 3;
        self.prev = action as f32;
        let r = if (action != 0) != need {
            StepResult { reward: -1.0, done: true, success: false }
        } else {
            self.c += 1;
            if self.c >= self.len {
                StepResult { reward: 1.0, done: true, success: true }
            } else {
                StepResult { reward: 0.1, done: false, success: false }
            }
        };
        self.write_obs(obs);
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_play_succeeds() {
        let mut e = MockEnv::new(8);
        let mut obs = [0f32; MockEnv::OBS_DIM];
        e.reset(0.0, &mut obs).unwrap();
        let mut last = StepResult::default();
        for c in 0..8 {
            last = e.step((c % 4 == 3) as u8, &mut obs).unwrap();
        }
        assert!(last.done && last.success);
    }

    #[test]
    fn wrong_action_dies_and_start_x_works() {
        let mut e = MockEnv::new(8);
        let mut obs = [0f32; MockEnv::OBS_DIM];
        e.reset(3.0, &mut obs).unwrap();
        assert_eq!(obs[3], 1.0); // c % 4 == 3
        let r = e.step(0, &mut obs).unwrap();
        assert!(r.done && !r.success);
    }
}
