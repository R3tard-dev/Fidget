use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;

pub struct Row {
    pub iter: u64,
    pub total_steps: u64,
    pub sps: f64,
    pub ticks_per_s: f64,
    pub frontier: usize,
    pub stages: usize,
    pub start_x: f32,
    pub window_rate: f32,
    pub window_n: usize,
    pub iter_success: f32,
    pub iter_episodes: u32,
    pub mean_return: f32,
    pub mean_len: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub kl: f32,
    pub clip_frac: f32,
}

pub struct CsvLogger {
    w: BufWriter<std::fs::File>,
}

const HEADER: &str = "iter,total_steps,sps,ticks_per_s,frontier,stages,start_x,window_rate,window_n,iter_success,iter_episodes,mean_return,mean_len,policy_loss,value_loss,entropy,kl,clip_frac";

impl CsvLogger {
    pub fn open(path: &str) -> Result<Self> {
        let fresh = !Path::new(path).exists() || std::fs::metadata(path)?.len() == 0;
        let f = OpenOptions::new().create(true).append(true).open(path)?;
        let mut w = BufWriter::new(f);
        if fresh {
            writeln!(w, "{HEADER}")?;
            w.flush()?;
        }
        Ok(Self { w })
    }

    pub fn row(&mut self, r: &Row) -> Result<()> {
        writeln!(
            self.w,
            "{},{},{:.1},{:.1},{},{},{:.1},{:.4},{},{:.4},{},{:.4},{:.1},{:.5},{:.5},{:.5},{:.5},{:.4}",
            r.iter, r.total_steps, r.sps, r.ticks_per_s, r.frontier, r.stages, r.start_x, r.window_rate,
            r.window_n, r.iter_success, r.iter_episodes, r.mean_return, r.mean_len, r.policy_loss,
            r.value_loss, r.entropy, r.kl, r.clip_frac
        )?;
        self.w.flush()?;
        Ok(())
    }
}
