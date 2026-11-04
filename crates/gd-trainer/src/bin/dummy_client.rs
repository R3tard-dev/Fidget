//! Usage:
//!   dummy_client [--port 47000] [--steps 2000] [--skip 1] [--random-start]   random clicks + throughput
//!   dummy_client --restore-test                                               reset to several start_x
//!   dummy_client --determinism [--ticks 120]                                  restore fidelity: run twice, compare
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use gd_env::Connection;
use gd_protocol::{err_name, Header, Request, BASE_PORT, NUM_OBJ, OBJ_FEATS, OBS_DIM, PLAYER_FEATS, PROTOCOL_VERSION};
use rand::Rng;

const PLAYER_NAMES: [&str; PLAYER_FEATS] = ["x", "y", "vy", "mode", "speed", "grav", "ground", "rsv"];

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2).find(|w| w[0] == name).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
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

struct Info {
    spacing: f32,
    xs: Vec<f32>,
}

fn fetch_info(conn: &mut Connection, obs: &mut [f32; OBS_DIM]) -> Result<Info> {
    let h = conn.call(&Request::info(0), obs)?;
    check(&h)?;
    let count = h.tick as usize;
    let spacing = h.dx;
    let mut xs = Vec::with_capacity(count);
    for page in 0..count.div_ceil(OBS_DIM) {
        if page > 0 {
            let h = conn.call(&Request::info(page as u32), obs)?;
            check(&h)?;
        }
        let n = (count - page * OBS_DIM).min(OBS_DIM);
        xs.extend_from_slice(&obs[..n]);
    }
    Ok(Info { spacing, xs })
}

fn sample_points(info: &Info, n: usize) -> Vec<f32> {
    if info.xs.is_empty() {
        return vec![0.0];
    }
    (0..n).map(|i| info.xs[(i * (info.xs.len() - 1)) / (n - 1).max(1)]).collect()
}

fn restore_test(conn: &mut Connection, obs: &mut [f32; OBS_DIM], info: &Info) -> Result<()> {
    for sx in sample_points(info, 8) {
        let h = conn.call(&Request::reset(sx), obs)?;
        check(&h)?;
        let off = (obs[0] - sx).abs();
        println!(
            "requested x={sx:.1} -> restored x={:.3} (off by {off:.3}{}) y={:.3} vy={:.3} mode={} speed={} grav={} ground={}",
            obs[0], if off > 1.0 { " <-- MISMATCH" } else { "" }, obs[1], obs[2], obs[3], obs[4], obs[5], obs[6]
        );
        let mut dxs = [0f32; 3];
        for d in dxs.iter_mut() {
            let h = conn.call(&Request::step(false, 1), obs)?;
            check(&h)?;
            *d = h.dx;
            if h.done() {
                break;
            }
        }
        println!("    first 3 step dx: {dxs:?}");
    }
    Ok(())
}

fn run_script(
    conn: &mut Connection,
    obs: &mut [f32; OBS_DIM],
    start_x: f32,
    ticks: u32,
    seed: u32,
) -> Result<([f32; OBS_DIM], u32, bool)> {
    let h = conn.call(&Request::reset(start_x), obs)?;
    check(&h)?;
    let mut rng = seed;
    let mut click = false;
    let mut ended_at = ticks;
    let mut dead = false;
    for t in 0..ticks {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        if (rng >> 24) < 8 {
            click = !click; // ~3% toggle chance per tick
        }
        let h = conn.call(&Request::step(click, 1), obs)?;
        check(&h)?;
        if h.done() {
            ended_at = t + 1;
            dead = h.dead();
            break;
        }
    }
    Ok((*obs, ended_at, dead))
}

fn determinism(conn: &mut Connection, obs: &mut [f32; OBS_DIM], info: &Info, ticks: u32) -> Result<()> {
    if info.xs.is_empty() {
        bail!("no snapshots; record a reference run first (record-first setting)");
    }
    let mut fails = 0;
    let pts = sample_points(info, 6);
    for (i, &sx) in pts.iter().enumerate() {
        let seed = 12345 + i as u32;
        let (a, na, da) = run_script(conn, obs, sx, ticks, seed)?;
        let (b, nb, db) = run_script(conn, obs, sx, ticks, seed)?;
        if a == b && na == nb && da == db {
            println!("PASS  x={sx:.1}: {na} ticks, end x={:.3} y={:.3} dead={da}", a[0], a[1]);
        } else {
            fails += 1;
            let idx = a.iter().zip(b.iter()).position(|(p, q)| p != q);
            println!(
                "FAIL  x={sx:.1}: ticks {na}/{nb} dead {da}/{db}, first differing obs idx {idx:?} ({:?} vs {:?})",
                idx.map(|k| a[k]),
                idx.map(|k| b[k])
            );
        }
    }
    println!("determinism: {} / {} passed", pts.len() - fails, pts.len());
    Ok(())
}

fn main() -> Result<()> {
    let port: u16 = arg("--port", BASE_PORT);
    let steps: u64 = arg("--steps", 2000);
    let skip: u8 = arg("--skip", 1);
    let ticks: u32 = arg("--ticks", 120);
    let random_start = flag("--random-start");

    let mut conn = Connection::connect(port, Duration::from_secs(15))?;
    let mut obs = [0f32; OBS_DIM];

    let h = conn.call(&Request::hello(), &mut obs)?;
    check(&h)?;
    println!("connected: mod protocol v{} (client v{PROTOCOL_VERSION}), obs_dim {}", h.version, h.obs_dim);

    let info = fetch_info(&mut conn, &mut obs)?;
    if info.xs.is_empty() {
        println!("snapshots: none (reset(start_x > 0) will report an error)");
    } else {
        println!(
            "snapshots: {} (spacing {:.0}), x from {:.1} to {:.1}",
            info.xs.len(), info.spacing, info.xs[0], info.xs[info.xs.len() - 1]
        );
    }

    if flag("--restore-test") {
        restore_test(&mut conn, &mut obs, &info)?;
        conn.call(&Request::close(), &mut obs)?;
        return Ok(());
    }
    if flag("--determinism") {
        determinism(&mut conn, &mut obs, &info, ticks)?;
        conn.call(&Request::close(), &mut obs)?;
        return Ok(());
    }

    let mut rng = rand::rng();
    let start = |rng: &mut rand::rngs::ThreadRng| -> f32 {
        if random_start && !info.xs.is_empty() {
            info.xs[rng.random_range(0..info.xs.len())]
        } else {
            0.0
        }
    };

    let h = conn.call(&Request::reset(start(&mut rng)), &mut obs)?;
    check(&h)?;
    println!("after reset (tick {}):", h.tick);
    print_obs(&obs);

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
            let h = conn.call(&Request::reset(start(&mut rng)), &mut obs)?;
            check(&h)?;
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    println!(
        "\n{steps} steps (skip {skip}) in {secs:.2}s = {:.0} steps/s, {:.0} ticks/s",
        steps as f64 / secs, steps as f64 * skip as f64 / secs
    );
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
