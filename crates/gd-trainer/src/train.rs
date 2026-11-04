use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use gd_env::{Env, GdEnv, GdEnvConfig, MockEnv, StepResult, VecEnv};
use gd_ppo::{new_ppo, Dev, PpoConfig, Rollout};

use crate::curriculum::{Curriculum, CurriculumConfig};
use crate::logger::{CsvLogger, Row};

pub const USAGE: &str = "\
gd-trainer [options]
  --mock                 train on a toy env (no game needed)
  --envs N               parallel game instances (ports base..base+N-1)   [1]
  --base-port P          first port                                       [47000]
  --skip K               physics ticks per agent step                     [4]
  --horizon T            steps per env per PPO iteration                  [128]
  --iters N              max PPO iterations                               [100000]
  --hidden H             MLP width                                        [128]
  --lr X --gamma X --lambda X --clip X --epochs N --minibatches N --ent X
  --stride K             use every K-th snapshot as a curriculum stage    [1]
  --window N             episodes in the success window                   [200]
  --threshold X          success rate needed to advance (strict >)        [0.95]
  --frontier-frac X      share of episodes from the frontier stage        [0.7]
  --max-steps N          episode step cap                                 [4000]
  --progress-scale X --death-penalty X --success-bonus X
  --threads N            CPU threads for burn (leave cores for the games) [2]
  --seed N  --save-every N  --ckpt-dir DIR  --csv FILE  --resume";

#[derive(Debug, Clone)]
pub struct Args {
    pub mock: bool,
    pub envs: usize,
    pub base_port: u16,
    pub skip: u8,
    pub horizon: usize,
    pub iters: u64,
    pub hidden: usize,
    pub lr: f64,
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f32,
    pub epochs: usize,
    pub minibatches: usize,
    pub ent: f32,
    pub stride: usize,
    pub window: usize,
    pub threshold: f32,
    pub frontier_frac: f32,
    pub max_steps: u32,
    pub progress_scale: f32,
    pub death_penalty: f32,
    pub success_bonus: f32,
    pub threads: usize,
    pub seed: u64,
    pub save_every: u64,
    pub ckpt_dir: String,
    pub csv: String,
    pub resume: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mock: false,
            envs: 1,
            base_port: gd_protocol::BASE_PORT,
            skip: 4,
            horizon: 128,
            iters: 100_000,
            hidden: 128,
            lr: 3e-4,
            gamma: 0.995,
            lambda: 0.95,
            clip: 0.2,
            epochs: 4,
            minibatches: 4,
            ent: 0.01,
            stride: 1,
            window: 200,
            threshold: 0.95,
            frontier_frac: 0.7,
            max_steps: 4000,
            progress_scale: 0.01,
            death_penalty: -5.0,
            success_bonus: 10.0,
            threads: 2,
            seed: 1,
            save_every: 10,
            ckpt_dir: "checkpoints".into(),
            csv: "train_log.csv".into(),
            resume: false,
        }
    }
}

fn get<T: std::str::FromStr>(argv: &[String], name: &str, default: T) -> T {
    argv.windows(2).find(|w| w[0] == name).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}

fn has(argv: &[String], name: &str) -> bool {
    argv.iter().any(|a| a == name)
}

impl Args {
    pub fn from_env() -> Self {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let d = Args::default();
        Args {
            mock: has(&argv, "--mock"),
            envs: get(&argv, "--envs", d.envs),
            base_port: get(&argv, "--base-port", d.base_port),
            skip: get(&argv, "--skip", d.skip),
            horizon: get(&argv, "--horizon", d.horizon),
            iters: get(&argv, "--iters", d.iters),
            hidden: get(&argv, "--hidden", d.hidden),
            lr: get(&argv, "--lr", d.lr),
            gamma: get(&argv, "--gamma", d.gamma),
            lambda: get(&argv, "--lambda", d.lambda),
            clip: get(&argv, "--clip", d.clip),
            epochs: get(&argv, "--epochs", d.epochs),
            minibatches: get(&argv, "--minibatches", d.minibatches),
            ent: get(&argv, "--ent", d.ent),
            stride: get(&argv, "--stride", d.stride),
            window: get(&argv, "--window", d.window),
            threshold: get(&argv, "--threshold", d.threshold),
            frontier_frac: get(&argv, "--frontier-frac", d.frontier_frac),
            max_steps: get(&argv, "--max-steps", d.max_steps),
            progress_scale: get(&argv, "--progress-scale", d.progress_scale),
            death_penalty: get(&argv, "--death-penalty", d.death_penalty),
            success_bonus: get(&argv, "--success-bonus", d.success_bonus),
            threads: get(&argv, "--threads", d.threads),
            seed: get(&argv, "--seed", d.seed),
            save_every: get(&argv, "--save-every", d.save_every),
            ckpt_dir: get(&argv, "--ckpt-dir", d.ckpt_dir.clone()),
            csv: get(&argv, "--csv", d.csv.clone()),
            resume: has(&argv, "--resume"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainSummary {
    pub iterations: u64,
    pub frontier: usize,
    pub stages: usize,
    pub finished: bool,
}

const MOCK_LEN: usize = 24;

fn build_envs(args: &Args) -> Result<(Vec<Box<dyn Env + Send>>, Vec<f32>, usize)> {
    let mut envs: Vec<Box<dyn Env + Send>> = Vec::new();
    if args.mock {
        for _ in 0..args.envs {
            envs.push(Box::new(MockEnv::new(MOCK_LEN)));
        }
        return Ok((envs, MockEnv::snapshot_xs(MOCK_LEN), MockEnv::OBS_DIM));
    }

    let cfg = GdEnvConfig {
        skip: args.skip,
        max_steps: args.max_steps,
        progress_scale: args.progress_scale,
        death_penalty: args.death_penalty,
        success_bonus: args.success_bonus,
        ..Default::default()
    };
    let mut xs: Vec<f32> = Vec::new();
    for i in 0..args.envs {
        let port = args.base_port + i as u16;
        let mut e = GdEnv::connect(port, cfg.clone()).with_context(|| format!("env {i} on port {port}"))?;
        let xi = e.snapshot_xs()?;
        if i == 0 {
            xs = xi;
        } else if xi.len() != xs.len() {
            bail!("env {i} has {} snapshots but env 0 has {}; all instances need the same reference run", xi.len(), xs.len());
        }
        envs.push(Box::new(e));
    }
    Ok((envs, xs, gd_env::PROC_OBS_DIM))
}

fn read_meta(path: &str) -> Option<(u64, u64, usize)> {
    let s = std::fs::read_to_string(path).ok()?;
    let (mut it, mut st, mut fr) = (None, None, None);
    for line in s.lines() {
        let (k, v) = line.split_once('=')?;
        match k {
            "iter" => it = v.parse().ok(),
            "steps" => st = v.parse().ok(),
            "frontier" => fr = v.parse().ok(),
            _ => {}
        }
    }
    Some((it?, st?, fr?))
}

pub fn run(args: &Args) -> Result<TrainSummary> {
    // Must happen before burn/rayon first runs: caps the threads used by ndarray matmuls.
    std::env::set_var("RAYON_NUM_THREADS", args.threads.max(1).to_string());

    let (envs, snapshot_xs, dim) = build_envs(args)?;
    let n = envs.len();
    let mut venv = VecEnv::new(envs)?;

    let mut cur = Curriculum::new(
        &snapshot_xs,
        CurriculumConfig { window: args.window, threshold: args.threshold, frontier_frac: args.frontier_frac, stride: args.stride },
        args.seed.wrapping_add(17),
    )?;
    println!(
        "{} envs, obs_dim {dim}, {} curriculum stages (x from {:.0} down to {:.0}), skip {}, horizon {}",
        n, cur.len(), cur.starts()[0], cur.starts()[cur.len() - 1], args.skip, args.horizon
    );

    let device = Dev::default();
    let mut ppo = new_ppo(
        PpoConfig {
            hidden: args.hidden,
            lr: args.lr,
            gamma: args.gamma,
            lambda: args.lambda,
            clip: args.clip,
            epochs: args.epochs,
            minibatches: args.minibatches,
            ent_coef: args.ent,
            seed: args.seed,
            ..Default::default()
        },
        dim,
        &device,
    );

    std::fs::create_dir_all(&args.ckpt_dir)?;
    let model_path = format!("{}/model_latest", args.ckpt_dir);
    let meta_path = format!("{}/meta.txt", args.ckpt_dir);
    let (mut iter, mut total_steps) = (0u64, 0u64);
    if args.resume {
        if !Path::new(&format!("{model_path}.mpk")).exists() {
            bail!("--resume given but {model_path}.mpk does not exist");
        }
        ppo.load(&model_path)?;
        if let Some((i, s, f)) = read_meta(&meta_path) {
            iter = i;
            total_steps = s;
            cur.set_frontier(f);
        }
        println!("resumed: iteration {iter}, {total_steps} steps, frontier stage {}", cur.frontier());
    }
    let mut csv = CsvLogger::open(&args.csv)?;

    // ---- reusable buffers ----
    let mut ro = Rollout::new(args.horizon, n, dim);
    let mut obs = vec![0f32; n * dim];
    let mut logits: Vec<f32> = Vec::with_capacity(n);
    let mut values: Vec<f32> = Vec::with_capacity(n);
    let mut act_f = vec![0f32; n];
    let mut logp_buf = vec![0f32; n];
    let mut act_u8 = vec![0u8; n];
    let mut results = vec![StepResult::default(); n];
    let mut env_stage = vec![0usize; n];
    let mut ep_ret = vec![0f32; n];
    let mut ep_len = vec![0u32; n];
    let mut done_idx: Vec<usize> = Vec::with_capacity(n);
    let mut starts: Vec<f32> = Vec::with_capacity(n);

    for e in 0..n {
        let (s, x) = cur.sample();
        env_stage[e] = s;
        starts.push(x);
    }
    venv.reset_all(&starts, &mut obs)?;

    let mut iters_done = 0u64;
    while iters_done < args.iters && !cur.finished() {
        let t0 = Instant::now();
        let infer = ppo.infer_model();
        let (mut ep_count, mut ep_succ, mut sum_ret, mut sum_len) = (0u32, 0u32, 0f64, 0f64);

        for t in 0..args.horizon {
            let base = t * n * dim;
            ro.obs[base..base + n * dim].copy_from_slice(&obs);

            ppo.act_value(&infer, &obs, n, &mut logits, &mut values);
            ppo.sample(&logits, &mut act_f, &mut logp_buf);
            for e in 0..n {
                act_u8[e] = act_f[e] as u8;
            }
            ro.set_step(t, &act_f, &logp_buf, &values);

            venv.step_all(&act_u8, &mut obs, &mut results)?;

            done_idx.clear();
            starts.clear();
            for e in 0..n {
                let r = results[e];
                ro.rew[t * n + e] = r.reward;
                ro.done[t * n + e] = if r.done { 1.0 } else { 0.0 };
                ep_ret[e] += r.reward;
                ep_len[e] += 1;
                if r.done {
                    ep_count += 1;
                    ep_succ += r.success as u32;
                    sum_ret += ep_ret[e] as f64;
                    sum_len += ep_len[e] as f64;
                    ep_ret[e] = 0.0;
                    ep_len[e] = 0;

                    if let Some(adv) = cur.record(env_stage[e], r.success) {
                        if adv.finished {
                            println!(">>> curriculum complete: final stage solved ({:.1}% success)", adv.rate * 100.0);
                        } else {
                            println!(
                                ">>> stage {} -> {} (x = {:.0}, window success {:.1}%)",
                                adv.from, adv.to, cur.starts()[adv.to], adv.rate * 100.0
                            );
                        }
                        ppo.save(&format!("{}/stage_{}", args.ckpt_dir, adv.from))?;
                    }
                    let (s, x) = cur.sample();
                    env_stage[e] = s;
                    done_idx.push(e);
                    starts.push(x);
                }
            }
            if !done_idx.is_empty() {
                venv.reset_subset(&done_idx, &starts, &mut obs)?;
            }
        }

        // bootstrap value of the state after the last step
        ppo.act_value(&infer, &obs, n, &mut logits, &mut values);
        ro.compute_gae(&values, args.gamma, args.lambda);
        let st = ppo.update(&ro);

        iter += 1;
        iters_done += 1;
        let steps = (args.horizon * n) as u64;
        total_steps += steps;
        let secs = t0.elapsed().as_secs_f64().max(1e-9);
        let (wr, wn) = cur.window_rate();
        let ec = ep_count.max(1) as f64;
        let row = Row {
            iter,
            total_steps,
            sps: steps as f64 / secs,
            ticks_per_s: steps as f64 * args.skip as f64 / secs,
            frontier: cur.frontier(),
            stages: cur.len(),
            start_x: cur.frontier_x(),
            window_rate: wr,
            window_n: wn,
            iter_success: if ep_count > 0 { ep_succ as f32 / ep_count as f32 } else { 0.0 },
            iter_episodes: ep_count,
            mean_return: (sum_ret / ec) as f32,
            mean_len: (sum_len / ec) as f32,
            policy_loss: st.policy_loss,
            value_loss: st.value_loss,
            entropy: st.entropy,
            kl: st.approx_kl,
            clip_frac: st.clip_frac,
        };
        println!(
            "it {:>5} | steps {:>9} | {:>6.0} sps ({:>6.0} ticks/s) | stage {:>3}/{} x={:>7.0} | win {:>5.1}% (n={:>3}) | succ {:>5.1}% ({:>3} eps) | ret {:>7.2} len {:>6.1} | ent {:.3} kl {:.4}",
            row.iter, row.total_steps, row.sps, row.ticks_per_s, row.frontier + 1, row.stages, row.start_x,
            row.window_rate * 100.0, row.window_n, row.iter_success * 100.0, row.iter_episodes,
            row.mean_return, row.mean_len, row.entropy, row.kl
        );
        csv.row(&row)?;

        if iter % args.save_every.max(1) == 0 || cur.finished() {
            ppo.save(&model_path)?;
            std::fs::write(&meta_path, format!("iter={iter}\nsteps={total_steps}\nfrontier={}\n", cur.frontier()))?;
        }
    }

    ppo.save(&model_path)?;
    std::fs::write(&meta_path, format!("iter={iter}\nsteps={total_steps}\nfrontier={}\n", cur.frontier()))?;
    Ok(TrainSummary { iterations: iters_done, frontier: cur.frontier(), stages: cur.len(), finished: cur.finished() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(tag: &str) -> Args {
        let dir = std::env::temp_dir().join(format!("gdrl_test_{tag}_{}", std::process::id()));
        Args {
            mock: true,
            envs: 2,
            horizon: 32,
            hidden: 32,
            window: 50,
            threads: 1,
            ckpt_dir: dir.join("ckpt").to_string_lossy().into_owned(),
            csv: dir.join("log.csv").to_string_lossy().into_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn mock_training_smoke() {
        let mut a = quick("smoke");
        std::fs::create_dir_all(Path::new(&a.csv).parent().unwrap()).unwrap();
        a.iters = 3;
        let s = run(&a).unwrap();
        assert_eq!(s.iterations, 3);
        assert!(std::fs::read_to_string(&a.csv).unwrap().lines().count() >= 4);
    }

    /// Slow: `cargo test --release -p gd-trainer -- --ignored`
    #[test]
    #[ignore]
    fn ppo_solves_mock_curriculum() {
        let mut a = quick("learn");
        std::fs::create_dir_all(Path::new(&a.csv).parent().unwrap()).unwrap();
        a.envs = 4;
        a.horizon = 128;
        a.iters = 1500;
        let s = run(&a).unwrap();
        assert!(s.finished, "curriculum stuck at stage {}/{}", s.frontier + 1, s.stages);
    }
}
