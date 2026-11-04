use anyhow::{anyhow, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{activation, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::model::ActorCritic;
use crate::{Dev, B, IB};

#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub hidden: usize,
    pub lr: f64,
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f32,
    pub epochs: usize,
    pub minibatches: usize,
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub target_kl: f32,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            lr: 3e-4,
            gamma: 0.995,
            lambda: 0.95,
            clip: 0.2,
            epochs: 4,
            minibatches: 4,
            ent_coef: 0.01,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            target_kl: 0.03,
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_frac: f32,
}

/// Flat rollout storage, indexed [t * n + env]; allocated once and reused every iteration.
pub struct Rollout {
    pub t: usize,
    pub n: usize,
    pub dim: usize,
    pub obs: Vec<f32>,
    pub act: Vec<f32>,
    pub logp: Vec<f32>,
    pub val: Vec<f32>,
    pub rew: Vec<f32>,
    pub done: Vec<f32>, // 1.0 if the episode ended after this step
    pub adv: Vec<f32>,
    pub ret: Vec<f32>,
}

impl Rollout {
    pub fn new(t: usize, n: usize, dim: usize) -> Self {
        let m = t * n;
        Self {
            t,
            n,
            dim,
            obs: vec![0.0; m * dim],
            act: vec![0.0; m],
            logp: vec![0.0; m],
            val: vec![0.0; m],
            rew: vec![0.0; m],
            done: vec![0.0; m],
            adv: vec![0.0; m],
            ret: vec![0.0; m],
        }
    }

    pub fn set_step(&mut self, step: usize, act: &[f32], logp: &[f32], val: &[f32]) {
        let b = step * self.n;
        self.act[b..b + self.n].copy_from_slice(act);
        self.logp[b..b + self.n].copy_from_slice(logp);
        self.val[b..b + self.n].copy_from_slice(val);
    }

    /// Generalised advantage estimation. `last_val[e]` bootstraps the value after the final step.
    pub fn compute_gae(&mut self, last_val: &[f32], gamma: f32, lambda: f32) {
        let n = self.n;
        for e in 0..n {
            let mut gae = 0.0f32;
            for t in (0..self.t).rev() {
                let i = t * n + e;
                let next_val = if t + 1 == self.t { last_val[e] } else { self.val[(t + 1) * n + e] };
                let nonterm = 1.0 - self.done[i];
                let delta = self.rew[i] + gamma * next_val * nonterm - self.val[i];
                gae = delta + gamma * lambda * nonterm * gae;
                self.adv[i] = gae;
                self.ret[i] = gae + self.val[i];
            }
        }
    }
}

pub fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

pub struct Ppo<O> {
    pub cfg: PpoConfig,
    model: ActorCritic<B>,
    optim: O,
    device: Dev,
    rng: StdRng,
    obs_dim: usize,
}

pub fn new_ppo(cfg: PpoConfig, obs_dim: usize, device: &Dev) -> Ppo<impl Optimizer<ActorCritic<B>, B>> {
    let model = ActorCritic::<B>::new(obs_dim, cfg.hidden, device);
    let optim = AdamConfig::new()
        .with_epsilon(1e-5)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
        .init::<B, ActorCritic<B>>();
    let rng = StdRng::seed_from_u64(cfg.seed);
    Ppo { cfg, model, optim, device: device.clone(), rng, obs_dim }
}

fn scalar(t: Tensor<B, 1>) -> f32 {
    t.into_data().to_vec::<f32>().expect("f32 scalar")[0]
}

impl<O: Optimizer<ActorCritic<B>, B>> Ppo<O> {
    /// Autodiff-free copy of the current weights; build once per rollout.
    pub fn infer_model(&self) -> ActorCritic<IB> {
        self.model.valid()
    }

    /// One batched forward pass for all envs: obs is [n * obs_dim].
    pub fn act_value(&self, m: &ActorCritic<IB>, obs: &[f32], n: usize, logits: &mut Vec<f32>, values: &mut Vec<f32>) {
        let x = Tensor::<IB, 2>::from_data(TensorData::new(obs.to_vec(), [n, self.obs_dim]), &self.device);
        let (l, v) = m.forward(x);
        logits.clear();
        logits.extend(l.into_data().to_vec::<f32>().expect("logits"));
        values.clear();
        values.extend(v.into_data().to_vec::<f32>().expect("values"));
    }

    /// Bernoulli sampling from logits; writes actions (0/1) and their log-probabilities.
    pub fn sample(&mut self, logits: &[f32], act: &mut [f32], logp: &mut [f32]) {
        for (i, &l) in logits.iter().enumerate() {
            let p = 1.0 / (1.0 + (-l).exp());
            let a = self.rng.random::<f32>() < p;
            act[i] = if a { 1.0 } else { 0.0 };
            logp[i] = if a { log_sigmoid(l) } else { log_sigmoid(-l) };
        }
    }

    pub fn update(&mut self, ro: &Rollout) -> UpdateStats {
        let total = ro.t * ro.n;
        let dim = ro.dim;
        let device = self.device.clone();
        let eps = self.cfg.clip;

        // advantage normalisation over the whole batch
        let mean = ro.adv.iter().sum::<f32>() / total as f32;
        let var = ro.adv.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / total as f32;
        let inv_std = 1.0 / (var.sqrt() + 1e-8);
        let adv_n: Vec<f32> = ro.adv.iter().map(|a| (a - mean) * inv_std).collect();

        let mb = (total / self.cfg.minibatches.max(1)).max(1);
        let mut idx: Vec<usize> = (0..total).collect();
        let mut model = self.model.clone();

        let mut st = UpdateStats::default();
        let mut count = 0.0f32;

        'epochs: for _ in 0..self.cfg.epochs {
            idx.shuffle(&mut self.rng);
            let (mut ep_kl, mut ep_n) = (0.0f32, 0.0f32);

            for chunk in idx.chunks(mb) {
                let m = chunk.len();
                let mut o = Vec::with_capacity(m * dim);
                let mut a = Vec::with_capacity(m);
                let mut lp = Vec::with_capacity(m);
                let mut ad = Vec::with_capacity(m);
                let mut rt = Vec::with_capacity(m);
                for &i in chunk {
                    o.extend_from_slice(&ro.obs[i * dim..(i + 1) * dim]);
                    a.push(ro.act[i]);
                    lp.push(ro.logp[i]);
                    ad.push(adv_n[i]);
                    rt.push(ro.ret[i]);
                }
                let col = |v: Vec<f32>| Tensor::<B, 2>::from_data(TensorData::new(v, [m, 1]), &device);
                let obs_t = Tensor::<B, 2>::from_data(TensorData::new(o, [m, dim]), &device);
                let act_t = col(a);
                let old_lp = col(lp);
                let adv_t = col(ad);
                let ret_t = col(rt);

                let (logit, value) = model.forward(obs_t);
                let p = activation::sigmoid(logit.clone());
                let lp1 = activation::log_sigmoid(logit.clone());
                let lp0 = activation::log_sigmoid(logit.neg());

                let new_lp = act_t.clone() * lp1.clone() + act_t.neg().add_scalar(1.0) * lp0.clone();
                let entropy = (p.clone() * lp1 + p.neg().add_scalar(1.0) * lp0).neg().mean();

                let ratio = (new_lp - old_lp).exp();
                let surr1 = ratio.clone() * adv_t.clone();
                let surr2 = ratio.clone().clamp(1.0 - eps, 1.0 + eps) * adv_t;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                let diff = value - ret_t;
                let value_loss = (diff.clone() * diff).mean().mul_scalar(0.5);

                let loss = policy_loss.clone()
                    + value_loss.clone().mul_scalar(self.cfg.vf_coef)
                    - entropy.clone().mul_scalar(self.cfg.ent_coef);

                // diagnostics (CPU side)
                let ratio_v = ratio.into_data().to_vec::<f32>().expect("ratio");
                let (mut kl, mut clipped) = (0.0f32, 0.0f32);
                for &r in &ratio_v {
                    kl += (r - 1.0) - r.ln();
                    if (r - 1.0).abs() > eps {
                        clipped += 1.0;
                    }
                }
                let nr = ratio_v.len() as f32;
                st.approx_kl += kl / nr;
                st.clip_frac += clipped / nr;
                st.policy_loss += scalar(policy_loss);
                st.value_loss += scalar(value_loss);
                st.entropy += scalar(entropy);
                count += 1.0;
                ep_kl += kl / nr;
                ep_n += 1.0;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = self.optim.step(self.cfg.lr, model, grads);
            }

            if ep_kl / ep_n.max(1.0) > self.cfg.target_kl * 1.5 {
                break 'epochs;
            }
        }

        self.model = model;
        if count > 0.0 {
            st.policy_loss /= count;
            st.value_loss /= count;
            st.entropy /= count;
            st.approx_kl /= count;
            st.clip_frac /= count;
        }
        st
    }

    /// Writes `<path>.mpk` (full precision).
    pub fn save(&self, path: &str) -> Result<()> {
        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model.clone().save_file(path.to_string(), &rec).map_err(|e| anyhow!("saving {path}: {e:?}"))
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let device = self.device.clone();
        self.model = self
            .model
            .clone()
            .load_file(path.to_string(), &rec, &device)
            .map_err(|e| anyhow!("loading {path}: {e:?}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_matches_naive_reference() {
        let (t, gamma, lambda) = (5usize, 0.9f32, 0.8f32);
        let mut ro = Rollout::new(t, 1, 1);
        ro.rew = vec![1.0, 0.5, -1.0, 2.0, 1.0];
        ro.val = vec![0.5, 0.4, 0.3, 0.9, 0.2];
        ro.done = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let last = [0.7f32];
        ro.compute_gae(&last, gamma, lambda);

        for s in 0..t {
            let (mut adv, mut coef) = (0.0f32, 1.0f32);
            for k in s..t {
                let nonterm = 1.0 - ro.done[k];
                let next_v = if k + 1 == t { last[0] } else { ro.val[k + 1] };
                let delta = ro.rew[k] + gamma * next_v * nonterm - ro.val[k];
                adv += coef * delta;
                if ro.done[k] == 1.0 {
                    break;
                }
                coef *= gamma * lambda;
            }
            assert!((ro.adv[s] - adv).abs() < 1e-5, "t={s}: {} vs {adv}", ro.adv[s]);
            assert!((ro.ret[s] - (adv + ro.val[s])).abs() < 1e-5);
        }
    }

    #[test]
    fn log_sigmoid_is_stable() {
        assert!((log_sigmoid(0.0) + std::f32::consts::LN_2).abs() < 1e-6);
        assert!(log_sigmoid(-100.0).is_finite() && log_sigmoid(100.0).is_finite());
        assert!((log_sigmoid(100.0)).abs() < 1e-6);
    }

    #[test]
    fn update_runs_and_is_finite() {
        let cfg = PpoConfig { hidden: 16, epochs: 2, minibatches: 2, ..Default::default() };
        let dev = Dev::default();
        let (t, n, dim) = (8usize, 2usize, 4usize);
        let mut ppo = new_ppo(cfg, dim, &dev);
        let mut ro = Rollout::new(t, n, dim);
        for (i, x) in ro.obs.iter_mut().enumerate() {
            *x = (i as f32 * 0.37).sin();
        }
        for i in 0..t * n {
            ro.act[i] = (i % 2) as f32;
            ro.logp[i] = -std::f32::consts::LN_2;
            ro.rew[i] = (i as f32).sin();
        }
        ro.compute_gae(&[0.0, 0.0], 0.99, 0.95);
        let st = ppo.update(&ro);
        assert!(st.policy_loss.is_finite() && st.value_loss.is_finite());
        assert!(st.entropy > 0.0 && st.entropy <= std::f32::consts::LN_2 + 1e-4);
    }
}
