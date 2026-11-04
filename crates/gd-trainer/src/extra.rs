//! Step 6 helpers: cross-instance snapshot validation and greedy evaluation.
use anyhow::{bail, Result};
use gd_env::{StepResult, VecEnv};
use gd_ppo::{new_ppo, Dev, PpoConfig};

use crate::train::{build_envs, Args};

/// All game instances must have rebuilt the same snapshots from the reference replay.
pub fn check_snapshots(i: usize, base: &[f32], other: &[f32]) -> Result<()> {
    if base.len() != other.len() {
        bail!(
            "instance {i} has {} snapshots but instance 0 has {}; check each instance's replay popup (was the replay OK and the same level loaded?)",
            other.len(),
            base.len()
        );
    }
    let max = base.iter().zip(other).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max > 1.0 {
        bail!("instance {i} snapshots differ from instance 0 by up to {max:.2} units; the replays diverged");
    }
    if max > 0.01 {
        eprintln!("warning: instance {i} snapshot positions differ by up to {max:.4} units (replay not bit-exact)");
    }
    Ok(())
}

/// Plays `args.eval` episodes from the real level start with the greedy policy (click iff logit > 0).
pub fn eval(args: &Args) -> Result<()> {
    std::env::set_var("RAYON_NUM_THREADS", args.threads.max(1).to_string());
    let (envs, _snapshots, dim) = build_envs(args)?;
    let n = envs.len();
    let mut venv = VecEnv::new(envs)?;

    let device = Dev::default();
    let mut ppo = new_ppo(PpoConfig { hidden: args.hidden, ..Default::default() }, dim, &device);
    let model_path = format!("{}/model_latest", args.ckpt_dir);
    ppo.load(&model_path)?;
    let infer = ppo.infer_model();

    let mut obs = vec![0f32; n * dim];
    let mut logits: Vec<f32> = Vec::with_capacity(n);
    let mut values: Vec<f32> = Vec::with_capacity(n);
    let mut acts = vec![0u8; n];
    let mut res = vec![StepResult::default(); n];
    let mut len = vec![0u32; n];
    let mut done_idx: Vec<usize> = Vec::with_capacity(n);
    venv.reset_all(&vec![0f32; n], &mut obs)?;

    let (mut eps, mut succ, mut sum_len) = (0u64, 0u64, 0f64);
    while eps < args.eval {
        ppo.act_value(&infer, &obs, n, &mut logits, &mut values);
        for e in 0..n {
            acts[e] = (logits[e] > 0.0) as u8;
        }
        venv.step_all(&acts, &mut obs, &mut res)?;
        done_idx.clear();
        for e in 0..n {
            len[e] += 1;
            if res[e].done {
                eps += 1;
                succ += res[e].success as u64;
                sum_len += len[e] as f64;
                len[e] = 0;
                done_idx.push(e);
            }
        }
        if !done_idx.is_empty() {
            let starts = vec![0f32; done_idx.len()];
            venv.reset_subset(&done_idx, &starts, &mut obs)?;
        }
    }
    println!(
        "eval from level start: {succ}/{eps} episodes completed the level ({:.1}%), mean length {:.0} steps",
        100.0 * succ as f64 / eps.max(1) as f64,
        sum_len / eps.max(1) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_snapshots_pass() {
        assert!(check_snapshots(1, &[100.0, 200.0], &[100.0, 200.0]).is_ok());
    }

    #[test]
    fn length_mismatch_fails() {
        assert!(check_snapshots(1, &[100.0, 200.0], &[100.0]).is_err());
    }

    #[test]
    fn large_drift_fails_small_drift_warns() {
        assert!(check_snapshots(1, &[100.0], &[103.0]).is_err());
        assert!(check_snapshots(1, &[100.0], &[100.05]).is_ok());
    }
}
