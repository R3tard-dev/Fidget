//! N environments, one persistent worker thread each. Observation buffers circulate between
//! this struct and the workers (moved through the channels), so stepping does not allocate.
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, bail, Result};

use crate::{Env, StepResult};

enum Kind {
    Reset,
    Step,
    Close,
}

struct Job {
    kind: Kind,
    action: u8,
    start_x: f32,
    buf: Vec<f32>,
}

struct Reply {
    buf: Vec<f32>,
    res: StepResult,
    err: Option<String>,
}

struct Worker {
    tx: Sender<Job>,
    rx: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
    spare: Vec<f32>,
}

fn worker_loop(mut env: Box<dyn Env + Send>, rx: Receiver<Job>, tx: Sender<Reply>) {
    while let Ok(mut job) = rx.recv() {
        let mut res = StepResult::default();
        let mut err = None;
        match job.kind {
            Kind::Reset => {
                if let Err(e) = env.reset(job.start_x, &mut job.buf) {
                    err = Some(format!("{e:#}"));
                }
            }
            Kind::Step => match env.step(job.action, &mut job.buf) {
                Ok(r) => res = r,
                Err(e) => err = Some(format!("{e:#}")),
            },
            Kind::Close => {
                env.close();
                break;
            }
        }
        if tx.send(Reply { buf: job.buf, res, err }).is_err() {
            break;
        }
    }
}

pub struct VecEnv {
    workers: Vec<Worker>,
    obs_dim: usize,
}

impl VecEnv {
    pub fn new(envs: Vec<Box<dyn Env + Send>>) -> Result<Self> {
        if envs.is_empty() {
            bail!("VecEnv needs at least one env");
        }
        let obs_dim = envs[0].obs_dim();
        if envs.iter().any(|e| e.obs_dim() != obs_dim) {
            bail!("all envs must have the same obs_dim");
        }
        let mut workers = Vec::with_capacity(envs.len());
        for (i, env) in envs.into_iter().enumerate() {
            let (job_tx, job_rx) = channel::<Job>();
            let (rep_tx, rep_rx) = channel::<Reply>();
            let handle = thread::Builder::new()
                .name(format!("env-{i}"))
                .spawn(move || worker_loop(env, job_rx, rep_tx))?;
            workers.push(Worker { tx: job_tx, rx: rep_rx, handle: Some(handle), spare: vec![0.0; obs_dim] });
        }
        Ok(Self { workers, obs_dim })
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn send(&mut self, i: usize, kind: Kind, action: u8, start_x: f32) -> Result<()> {
        let w = &mut self.workers[i];
        let buf = std::mem::take(&mut w.spare);
        w.tx.send(Job { kind, action, start_x, buf }).map_err(|_| anyhow!("env worker {i} is gone"))
    }

    fn recv(&mut self, i: usize, obs: &mut [f32]) -> Result<StepResult> {
        let reply = self.workers[i].rx.recv().map_err(|_| anyhow!("env worker {i} died"))?;
        self.workers[i].spare = reply.buf;
        if let Some(e) = reply.err {
            bail!("env {i}: {e}");
        }
        let d = self.obs_dim;
        obs[i * d..(i + 1) * d].copy_from_slice(&self.workers[i].spare);
        Ok(reply.res)
    }

    /// Reset the listed envs concurrently; their rows of `obs` are overwritten.
    pub fn reset_subset(&mut self, idx: &[usize], starts: &[f32], obs: &mut [f32]) -> Result<()> {
        for (k, &i) in idx.iter().enumerate() {
            self.send(i, Kind::Reset, 0, starts[k])?;
        }
        for &i in idx {
            self.recv(i, obs)?;
        }
        Ok(())
    }

    pub fn reset_all(&mut self, starts: &[f32], obs: &mut [f32]) -> Result<()> {
        let idx: Vec<usize> = (0..self.workers.len()).collect();
        self.reset_subset(&idx, starts, obs)
    }

    pub fn step_all(&mut self, actions: &[u8], obs: &mut [f32], res: &mut [StepResult]) -> Result<()> {
        for i in 0..self.workers.len() {
            self.send(i, Kind::Step, actions[i], 0.0)?;
        }
        for i in 0..self.workers.len() {
            res[i] = self.recv(i, obs)?;
        }
        Ok(())
    }
}

impl Drop for VecEnv {
    fn drop(&mut self) {
        for w in &mut self.workers {
            let _ = w.tx.send(Job { kind: Kind::Close, action: 0, start_x: 0.0, buf: Vec::new() });
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockEnv;

    #[test]
    fn steps_and_resets_in_parallel() {
        let envs: Vec<Box<dyn Env + Send>> = (0..3).map(|_| Box::new(MockEnv::new(8)) as Box<dyn Env + Send>).collect();
        let mut v = VecEnv::new(envs).unwrap();
        let d = v.obs_dim();
        let mut obs = vec![0f32; 3 * d];
        let mut res = vec![StepResult::default(); 3];

        v.reset_all(&[0.0, 1.0, 3.0], &mut obs).unwrap();
        assert_eq!(obs[0], 1.0); // env0: c=0 -> one-hot slot 0
        assert_eq!(obs[d + 1], 1.0); // env1: c=1
        assert_eq!(obs[2 * d + 3], 1.0); // env2: c=3

        v.step_all(&[0, 0, 0], &mut obs, &mut res).unwrap();
        assert!(!res[0].done && !res[1].done && res[2].done); // env2 needed a click

        v.reset_subset(&[2], &[0.0], &mut obs).unwrap();
        assert_eq!(obs[2 * d], 1.0);
    }
}
