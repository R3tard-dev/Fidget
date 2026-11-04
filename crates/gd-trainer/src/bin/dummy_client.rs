//! Random-click client: checks protocol, tick consistency and raw step throughput.
//! Usage: dummy_client [--port 47000] [--steps 2000] [--skip 1]
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use gd_env::Connection;
use gd_protocol::{err_name, Header, Request, BASE_PORT, NUM_OBJ, OBJ_FEATS, OBS_DIM, PLAYER_FEATS, PROTOCOL_VERSION};
use rand::Rng;

const PLAYER_NAMES: [&str; PLAYER_FEATS] = ["x", "y", "vy", "mode", "speed", "grav", "ground", "rsv"];

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn check(h: &Header) -> Result<()> {
    if h.err != 0 {
        bail!("mod reported error {} ({})", h.err, err_name(h.err));
    }
    Ok(())
}

fn print_obs(obs: &[f32; OBS_DIM]) {
    let mut s = String::new();
    for (n, v) in PLAYER_NAMES.iter().zip(obs.iter()) {
        s += &format!("{n}={v:.3} ");
    }
    println!("  player: {s}");
    for i in 0..NUM_OBJ.min(3) {
        let o = &obs[PLAYER_FEATS + i * OBJ_FEATS..PLAYER_FEATS + (i + 1) * OBJ_FEATS];
        println!("  obj[{i}]: type={} rel=({:.1},{:.1}) size=({:.1},{:.1})", o[0], o[1], o[2], o[3], o[4]);
    }
}

fn main() -> Result<()> {
    let port: u16 = arg("--port", BASE_PORT);
    let steps: u64 = arg("--steps", 2000);
    let skip: u8 = arg("--skip", 1);

    let mut conn = Connection::connect(port, Duration::from_secs(15))?;
    let mut obs = [0f32; OBS_DIM];

    let h = conn.call(&Request::hello(), &mut obs)?;
    check(&h)?;
    println!("connected: mod protocol v{} (client v{PROTOCOL_VERSION}), obs_dim {}", h.version, h.obs_dim);

    let h = conn.call(&Request::reset(0.0), &mut obs)?;
    check(&h)?;
    println!("after reset (tick {}):", h.tick);
    print_obs(&obs);

    let mut rng = rand::rng();
    let mut click = false;
    let mut hist: BTreeMap<i32, u32> = BTreeMap::new();
    let (mut episodes, mut ep_steps, mut best_x) = (0u32, 0u32, 0f32);
    let t0 = Instant::now();

    for step in 0..steps {
        if rng.random_bool(0.05) {
            click = !click;
        }
        let h = conn.call(&Request::step(click, skip), &mut obs)?;
        check(&h)?;
        ep_steps += 1;
        best_x = best_x.max(obs[0]);

        if skip == 1 && !h.done() {
            *hist.entry((h.dx * 1000.0).round() as i32).or_insert(0) += 1;
        }
        if step < 3 {
            println!("step {step}: click={click} dx={:.4} tick={} flags={}", h.dx, h.tick, h.flags);
            print_obs(&obs);
        } else if step % 200 == 0 {
            println!("step {step}: x={:.1} y={:.1} vy={:.2} dx={:.3}", obs[0], obs[1], obs[2], h.dx);
        }

        if h.done() {
            episodes += 1;
            println!("episode {episodes} over: {ep_steps} steps, x={:.1}, success={}", obs[0], h.success());
            ep_steps = 0;
            let h = conn.call(&Request::reset(0.0), &mut obs)?;
            check(&h)?;
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    println!("\n{steps} steps (skip {skip}) in {secs:.2}s = {:.0} steps/s, {:.0} ticks/s", steps as f64 / secs, steps as f64 * skip as f64 / secs);
    println!("episodes: {episodes}, best x: {best_x:.1}");

    if !hist.is_empty() {
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        println!("dx histogram (dx*1000 -> count), top 6. Healthy = one dominant bucket:");
        for (k, n) in v.iter().take(6) {
            println!("  {k:>6} -> {n}");
        }
    }

    conn.call(&Request::close(), &mut obs)?;
    Ok(())
}
