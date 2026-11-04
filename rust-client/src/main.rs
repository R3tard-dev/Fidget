use burn::optim::AdamWConfig;
use burn::tensor::{DType, Device, Int, Tensor};
use modurl::prelude::*;
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

const DTYPE: DType = DType::F32;

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

// ---------------------------------------------------------------------------
// Custom Gym environment that talks to the Geometry Dash bridge over TCP
// ---------------------------------------------------------------------------
struct GDGameEnv {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    prev_x: f32,
    device: Device,
}

impl GDGameEnv {
    fn connect(device: Device) -> Self {
        println!("Connecting to Geometry Dash bridge on 127.0.0.1:3000 ...");
        let stream = loop {
            match TcpStream::connect("127.0.0.1:3000") {
                Ok(s) => break s,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        };
        stream.set_nonblocking(false).ok();
        let reader = BufReader::new(stream.try_clone().unwrap());
        let writer = stream;
        println!("Connected.");
        Self {
            reader,
            writer,
            prev_x: 0.0,
            device,
        }
    }

    fn read_state(&mut self) -> GameState {
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line).is_err() || line.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(s) = serde_json::from_str::<GameState>(trimmed) {
                return s;
            }
        }
    }
}

impl Gym for GDGameEnv {
    type Error = std::io::Error;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        let action_val: i64 = action.into_scalar::<i64>().unwrap_or(0);
        let cmd = if action_val == 1 { "JUMP" } else { "RELEASE" };
        let _ = writeln!(self.writer, "{}", cmd);
        let _ = self.writer.flush();

        let state = self.read_state();

        let dx = state.x - self.prev_x;
        let reward = if state.dead {
            -10.0f32
        } else if dx > 0.0 {
            (dx * 0.01) as f32
        } else {
            0.0f32
        };
        self.prev_x = if state.dead { 0.0 } else { state.x };
        let done = state.dead;

        let obs = Tensor::from_floats(
            [state.x, state.y, state.vy, state.dist],
            &self.device,
        );

        Ok(StepInfo {
            observation: obs,
            reward,
            done,
            truncated: false,
            info: (),
        })
    }

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.prev_x = 0.0;
        let state = self.read_state();
        let obs = Tensor::from_floats(
            [state.x, state.y, state.vy, state.dist],
            &self.device,
        );
        Ok(ResetInfo {
            observation: obs,
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::<2>::new_with_universal_bounds([1, 4], -1e9, 1e9, &self.device)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_device = Device::Cpu;
    let device = env_device.clone().autodiff();

    let env = GDGameEnv::connect(env_device.clone());
    let envs = vec![env];
    let env = VectorizedGymWrapper::new(envs).unwrap();
    let mut env = DeviceMultiGymWrapper::new(env, env_device, device.clone());

    let observation_space = env.observation_space();
    let observation_size = observation_space.shape()[0];

    let online_q_network = MLP::builder()
        .input_size(observation_size)
        .output_size(2)
        .options((&device, DTYPE))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build online Q-network");

    let optimizer = AdamWConfig::new().init();

    let mut agent = DQNAgent::builder()
        .dtype(DTYPE)
        .action_space(env.action_space())
        .observation_space(observation_space)
        .online_q_network(online_q_network)
        .optimizer(optimizer)
        .learning_rate(2.5e-4)
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
        .build()
        .expect("DQN configuration should be valid");

    println!("Training for 100000 steps...");
    agent
        .learn(&mut env, 100_000)
        .expect("DQN learning failed");
    println!("Training complete.");
    Ok(())
}