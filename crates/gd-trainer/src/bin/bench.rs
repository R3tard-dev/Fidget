//! Throughput benchmark.
//!   bench --mock [--max-envs 8] [--secs 5] [--policy]       trainer-side ceiling, no game needed
//!   bench [--max-envs N] [--base-port 47000] [--skip 4] [--secs 10] [--policy] [--random-start]
//! Runs with 1..N game instances (the first k ports) and reports scaling.
use std::time::{Duration, Instant};

use anyhow::Result;
use gd_env::{Env, GdEnv, GdEnvConfig, MockEnv, StepResult, VecEnv};
use gd_ppo::{new_ppo, Dev, PpoConfig};
use rand::Rng;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.windows(2).find(|w| w[0] == name).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn sizes(max: usize) -> Vec<usize> {
    if max <= 8 {
        return (1..=max).collect();
    }
    let mut v = vec![1];
    let mut k = 2;
    while k < max {
        v.push(k);
        k *= 2;
    }
    v.push(max);
    v
}

fn build(n: usize, mock: bool, base: u16, skip: u8, wait: u64, need_xs: bool) -> Result<(Vec<Box<dyn Env + Send>>, Vec<f32>)> {
    let mut envs: Vec<Box<dyn Env + Send>> = Vec::new();
    if mock {
        for _ in 0..n {
            envs.push(Box::new(MockEnv::new(24)));
        }
        return Ok((envs, MockEnv::snapshot_xs(24)));
    }
    let cfg = GdEnvConfig { skip, ..Default::default() };
    let mut xs: Vec<f32> = Vec::new();
    for i in 0..n {
        let mut e = GdEnv::connect_retry(base + i as u16, cfg.clone(), Duration::from_secs(wait))?;
        if i == 0 && need_xs {
            xs = e.snapshot_xs()?;
        }
        envs.push(Box::new(e));
    }
    Ok((envs, xs))
}

fn run_size(n: usize, mock: bool, base: u16, skip: u8, secs: f64, policy: bool, random_start: bool, threads_wait: u64) -> Result<f64> {
    let (envs, xs) = build(n, mock, base, skip, threads_wait, random_start)?;
    let mut venv = VecEnv::new(envs)?;
    let d = venv.obs_dim();
    let mut rng = rand::rng();
    let pick = |rng: &mut rand::rngs::ThreadRng| -> f32 {
        if random_start && !xs.is_empty() {
            xs[rng.random_range(0..xs.len())]
        } else {
            0.0
        }
    };

    let device = Dev::default();
    let mut ppo = new_ppo(PpoConfig { hidden: 128, ..Default::default() }, d, &device);
    let infer = ppo.infer_model();

    let mut obs = vec![0f32; n * d];
    let mut res = vec![StepResult::default(); n];
    let mut acts = vec![0u8; n];
    let mut act_f = vec![0f32; n];
    let mut logp = vec![0f32; n];
    let mut logits: Vec<f32> = Vec::with_capacity(n);
    let mut values: Vec<f32> = Vec::with_capacity(n);
    let mut click = vec![false; n];
    let mut done_idx: Vec<usize> = Vec::with_capacity(n);
    let mut starts: Vec<f32> = Vec::with_capacity(n);

    for _ in 0..n {
        starts.push(pick(&mut rng));
    }
    venv.reset_all(&starts, &mut obs)?;

    let (mut t_infer, mut t_env, mut t_reset) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut iters, mut episodes) = (0u64, 0u64);
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        let ti = Instant::now();
        if policy {
            ppo.act_value(&infer, &obs, n, &mut logits, &mut values);
            ppo.sample(&logits, &mut act_f, &mut logp);
            for e in 0..n {
                acts[e] = act_f[e] as u8;
            }
        } else {
            for e in 0..n {
                if rng.random_bool(0.05) {
                    click[e] = !click[e];
                }
                acts[e] = click[e] as u8;
            }
        }
        t_infer += ti.elapsed();

        let te = Instant::now();
        venv.step_all(&acts, &mut obs, &mut res)?;
        t_env += te.elapsed();

        done_idx.clear();
        starts.clear();
        for e in 0..n {
            if res[e].done {
                done_idx.push(e);
                starts.push(pick(&mut rng));
            }
        }
        if !done_idx.is_empty() {
            episodes += done_idx.len() as u64;
            let tr = Instant::now();
            venv.reset_subset(&done_idx, &starts, &mut obs)?;
            t_reset += tr.elapsed();
        }
        iters += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let steps = iters * n as u64;
    let sps = steps as f64 / elapsed;
    let per_iter = |t: Duration| t.as_secs_f64() * 1000.0 / iters.max(1) as f64;
    println!(
        "{n:>3} envs | {sps:>9.0} steps/s | {:>10.0} ticks/s | {:>8.0} steps/s/env | per iter ms: policy {:.3}  env {:.3}  reset {:.3} | {episodes} episodes",
        sps * skip as f64,
        sps / n as f64,
        per_iter(t_infer),
        per_iter(t_env),
        per_iter(t_reset),
    );
    Ok(sps)
}

fn main() -> Result<()> {
    let mock = flag("--mock");
    let max_envs: usize = arg("--max-envs", 4);
    let secs: f64 = arg("--secs", 8.0);
    let skip: u8 = arg("--skip", 4);
    let base: u16 = arg("--base-port", gd_protocol::BASE_PORT);
    let policy = flag("--policy");
    let random_start = flag("--random-start");
    let threads: usize = arg("--threads", 2);
    let wait: u64 = arg("--wait", 120);
    std::env::set_var("RAYON_NUM_THREADS", threads.max(1).to_string());

    println!(
        "benchmark: {} | {} | skip {skip} | {secs}s per size | burn threads {threads}",
        if mock { "MOCK env (trainer-side ceiling)" } else { "real game instances" },
        if policy { "policy inference ON" } else { "random actions" },
    );
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for n in sizes(max_envs) {
        let sps = run_size(n, mock, base, skip, secs, policy, random_start, wait)?;
        rows.push((n, sps));
    }
    let single = rows[0].1;
    println!("\nscaling efficiency vs 1 env:");
    for (n, sps) in &rows {
        println!("  {n:>3} envs: {:>5.1}%", 100.0 * sps / (single * *n as f64));
    }
    let best = rows.iter().map(|r| r.1).fold(0.0f64, f64::max);
    let knee = rows.iter().find(|r| r.1 >= 0.9 * best).map(|r| r.0).unwrap_or(1);
    println!("peak {best:.0} steps/s; smallest env count within 90% of peak: {knee}");
    Ok(())
}
