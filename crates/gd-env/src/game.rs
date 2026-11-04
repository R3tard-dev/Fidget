//! The real environment: reward shaping and observation normalisation on top of the socket.
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use gd_protocol::{err_name, Header, Request, NUM_OBJ, OBJ_FEATS, OBS_DIM, PLAYER_FEATS};

use crate::{Connection, Env, StepResult};

/// 7 player features + previous action + objects.
pub const PROC_OBS_DIM: usize = 8 + NUM_OBJ * OBJ_FEATS;

// Normalisation constants. They are guesses at GD's unit scales; tune them if features look off.
const Y_SCALE: f32 = 300.0;
const VY_SCALE: f32 = 16.0;
const MODE_SCALE: f32 = 7.0;
const TYPE_SCALE: f32 = 32.0;
const REL_SCALE: f32 = 300.0;
const SIZE_SCALE: f32 = 30.0;

#[derive(Debug, Clone)]
pub struct GdEnvConfig {
    pub skip: u8,
    pub max_steps: u32,
    pub progress_scale: f32,
    pub death_penalty: f32, // negative number
    pub success_bonus: f32,
    pub step_penalty: f32,
    pub io_timeout: Duration,
}

impl Default for GdEnvConfig {
    fn default() -> Self {
        Self {
            skip: 4,
            max_steps: 4000,
            progress_scale: 0.01,
            death_penalty: -5.0,
            success_bonus: 10.0,
            step_penalty: 0.0,
            io_timeout: Duration::from_secs(30),
        }
    }
}

fn check(h: &Header) -> Result<()> {
    if h.err != 0 {
        bail!("mod reported error {} ({})", h.err, err_name(h.err));
    }
    Ok(())
}

/// raw layout: [x, y, vy, mode, speed, grav, ground, rsv, 16 x (type, rel_x, rel_y, w, h)]
pub fn process_obs(raw: &[f32; OBS_DIM], prev_action: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PROC_OBS_DIM);
    out[0] = raw[1] / Y_SCALE;
    out[1] = raw[2] / VY_SCALE;
    out[2] = raw[3] / MODE_SCALE;
    out[3] = raw[4];
    out[4] = raw[5];
    out[5] = raw[6];
    out[6] = raw[7];
    out[7] = prev_action;
    for i in 0..NUM_OBJ {
        let src = &raw[PLAYER_FEATS + i * OBJ_FEATS..PLAYER_FEATS + (i + 1) * OBJ_FEATS];
        let dst = &mut out[8 + i * OBJ_FEATS..8 + (i + 1) * OBJ_FEATS];
        if src[0] < 0.0 {
            dst.fill(0.0); // empty slot
        } else {
            dst[0] = src[0] / TYPE_SCALE;
            dst[1] = src[1] / REL_SCALE;
            dst[2] = src[2] / REL_SCALE;
            dst[3] = src[3] / SIZE_SCALE;
            dst[4] = src[4] / SIZE_SCALE;
        }
    }
}

pub struct GdEnv {
    conn: Connection,
    cfg: GdEnvConfig,
    raw: [f32; OBS_DIM],
    prev_action: f32,
    steps: u32,
}

impl GdEnv {
    pub fn connect(port: u16, cfg: GdEnvConfig) -> Result<Self> {
        let mut conn = Connection::connect(port, cfg.io_timeout)?;
        let mut raw = [0f32; OBS_DIM];
        let h = conn.call(&Request::hello(), &mut raw)?;
        check(&h)?;
        Ok(Self { conn, cfg, raw, prev_action: 0.0, steps: 0 })
    }

    /// x positions of the recorded snapshots, ascending (empty if no reference run was recorded).
    pub fn snapshot_xs(&mut self) -> Result<Vec<f32>> {
        let h = self.conn.call(&Request::info(0), &mut self.raw)?;
        check(&h)?;
        let count = h.tick as usize;
        let mut xs = Vec::with_capacity(count);
        for page in 0..count.div_ceil(OBS_DIM) {
            if page > 0 {
                let h = self.conn.call(&Request::info(page as u32), &mut self.raw)?;
                check(&h)?;
            }
            let n = (count - page * OBS_DIM).min(OBS_DIM);
            xs.extend_from_slice(&self.raw[..n]);
        }
        Ok(xs)
    }
}

impl Env for GdEnv {
    fn obs_dim(&self) -> usize {
        PROC_OBS_DIM
    }

    fn reset(&mut self, start_x: f32, obs: &mut [f32]) -> Result<()> {
        let h = self.conn.call(&Request::reset(start_x), &mut self.raw)?;
        check(&h)?;
        self.prev_action = 0.0; // the mod releases the button on reset
        self.steps = 0;
        process_obs(&self.raw, 0.0, obs);
        Ok(())
    }

    fn step(&mut self, action: u8, obs: &mut [f32]) -> Result<StepResult> {
        let h = self.conn.call(&Request::step(action != 0, self.cfg.skip), &mut self.raw)?;
        check(&h)?;
        self.steps += 1;

        let mut reward = h.dx.max(0.0) * self.cfg.progress_scale - self.cfg.step_penalty;
        if h.dead() {
            reward += self.cfg.death_penalty;
        }
        if h.success() {
            reward += self.cfg.success_bonus;
        }
        let timeout = self.steps >= self.cfg.max_steps;

        self.prev_action = if action != 0 { 1.0 } else { 0.0 };
        process_obs(&self.raw, self.prev_action, obs);
        Ok(StepResult { reward, done: h.done() || timeout, success: h.success() })
    }

    fn close(&mut self) {
        let _ = self.conn.call(&Request::close(), &mut self.raw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slots_are_zero_and_player_scaled() {
        let mut raw = [0f32; OBS_DIM];
        for i in 0..NUM_OBJ {
            raw[PLAYER_FEATS + i * OBJ_FEATS] = -1.0;
        }
        raw[1] = 150.0;
        let mut out = [9f32; PROC_OBS_DIM];
        process_obs(&raw, 1.0, &mut out);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[7], 1.0);
        assert!(out[8..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn real_object_is_scaled_and_nonzero_size() {
        let mut raw = [0f32; OBS_DIM];
        for i in 0..NUM_OBJ {
            raw[PLAYER_FEATS + i * OBJ_FEATS] = -1.0;
        }
        let o = PLAYER_FEATS;
        raw[o..o + 5].copy_from_slice(&[0.0, 150.0, -60.0, 30.0, 30.0]);
        let mut out = [0f32; PROC_OBS_DIM];
        process_obs(&raw, 0.0, &mut out);
        assert_eq!(&out[8..13], &[0.0, 0.5, -0.2, 1.0, 1.0]);
    }
}

impl GdEnv {
    /// Keeps retrying until the game instance is listening (it only starts the server after
    /// the level has loaded and the reference replay has finished).
    pub fn connect_retry(port: u16, cfg: GdEnvConfig, wait: Duration) -> Result<Self> {
        let start = Instant::now();
        let mut last_print: Option<Instant> = None;
        loop {
            match Self::connect(port, cfg.clone()) {
                Ok(e) => return Ok(e),
                Err(err) => {
                    if start.elapsed() >= wait {
                        return Err(err.context(format!("gave up on port {port} after {}s", wait.as_secs())));
                    }
                    if last_print.map_or(true, |t| t.elapsed() >= Duration::from_secs(5)) {
                        eprintln!("waiting for the game on port {port}: {err:#}");
                        last_print = Some(Instant::now());
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }
    }
}
