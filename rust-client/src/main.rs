use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use tokio::time::{sleep, Duration};

#[derive(Deserialize, Debug, Clone, Default)]
struct GameState {
    #[serde(default)]
    x: f32,
    #[serde(default)]
    y: f32,
    #[serde(default)]
    vy: f32,
    #[serde(default)]
    dist: f32,
    #[serde(default)]
    dead: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // --- ModuRL DQN agent setup (mirrors the official DQN example) ---
    let obs_size = 4usize; // x, y, vy, dist
    let action_count = 2usize; // 0 = release, 1 = jump

    let online_var_map = VarMap::new();
    let online_q_network = candle_nn::MLP::builder()
        .input_size(obs_size)
        .output_size(action_count)
        .vb(VarBuilder::from_varmap(&online_var_map, DType::F32, &device))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build online Q-network");

    let mut target_var_map = VarMap::new();
    let target_q_network = candle_nn::MLP::builder()
        .input_size(obs_size)
        .output_size(action_count)
        .vb(VarBuilder::from_varmap(&target_var_map, DType::F32, &device))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build target Q-network");

    let optimizer = AdamW::new(
        online_var_map.all_vars(),
        ParamsAdamW {
            lr: 2.5e-4,
            ..Default::default()
        },
    )
    .expect("failed to build optimizer");

    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(action_count as u32))
        .observation_space(Box::new(BoxSpace::new_with_universal_bounds(
            vec![obs_size],
            -1e9,
            1e9,
            &device,
        )))
        .online_q_network(online_q_network)
        .target_q_network(target_q_network)
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(1_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(100_000)
        .epsilon_schedule(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        })
        .replay_storage_config(ReplayStorageConfig::new(
            ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .build()
        .expect("DQN configuration should be valid");

    // --- Connect to the Geode mod socket ---
    println!("Connecting to Geometry Dash bridge on 127.0.0.1:3000 ...");
    let stream = loop {
        match TcpStream::connect("127.0.0.1:3000") {
            Ok(s) => break s,
            Err(_) => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    };
    stream.set_nonblocking(false).ok();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = stream;

    println!("Connected. Starting training loop.");

    let mut prev_x: f32 = 0.0;
    let mut episode_reward: f32 = 0.0;
    let mut steps: u64 = 0;

    loop {
        // Read one JSON line from the game
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.is_empty() {
            sleep(Duration::from_millis(1)).await;
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let state: GameState = match serde_json::from_str(trimmed) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Compute reward: distance traveled since last frame, minus death penalty
        let dx = state.x - prev_x;
        let reward = if state.dead {
            -10.0f
        } else if dx > 0.0 {
            dx * 0.01
        } else {
            0.0
        };
        prev_x = if state.dead { 0.0 } else { state.x };
        episode_reward += reward;

        // Build observation tensor [1, 4]
        let obs = Tensor::from_vec(
            vec![state.x, state.y, state.vy, state.dist],
            (1, obs_size),
            &device,
        )
        .unwrap();

        // Agent selects action
        let action_tensor = agent.act(&obs).unwrap();
        let action_val: u32 = action_tensor.to_vec2::<u32>().unwrap()[0][0];
        let cmd = if action_val == 1 { "JUMP" } else { "RELEASE" };

        // Send command to game
        let _ = writeln!(writer, "{}", cmd);
        let _ = writer.flush();

        if state.dead {
            println!(
                "Episode ended. reward={:.3} steps={}",
                episode_reward, steps
            );
            episode_reward = 0.0;
            steps = 0;
            prev_x = 0.0;
        } else {
            steps += 1;
        }

        // Periodic training tick (ModuRL learns over MultiGym; here we just
        // keep the agent alive. For full training, feed transitions into a
        // custom Gym and call agent.learn().)
        if steps % 1000 == 0 {
            println!("steps={} eps~", steps);
        }
    }
}