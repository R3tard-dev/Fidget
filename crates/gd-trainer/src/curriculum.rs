//! Backplay curriculum: stages are start positions ordered from the end of the level to the
//! beginning. The frontier advances when the success rate over the last `window` frontier-stage
//! episodes exceeds `threshold`.
use std::collections::VecDeque;

use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    pub window: usize,
    pub threshold: f32,
    pub frontier_frac: f32,
    pub stride: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self { window: 200, threshold: 0.95, frontier_frac: 0.7, stride: 1 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Advance {
    pub from: usize,
    pub to: usize,
    pub rate: f32,
    pub finished: bool, // true when the last stage (the real level start) was just solved
}

pub struct Curriculum {
    starts: Vec<f32>,
    frontier: usize,
    window: VecDeque<bool>,
    cfg: CurriculumConfig,
    finished: bool,
    rng: StdRng,
}

impl Curriculum {
    pub fn new(snapshot_xs: &[f32], cfg: CurriculumConfig, seed: u64) -> Result<Self> {
        if snapshot_xs.is_empty() {
            bail!("no snapshots: record a reference run in the mod first (record-first setting)");
        }
        let mut xs = snapshot_xs.to_vec();
        xs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // end -> beginning
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        if *xs.last().unwrap() > 1.0 {
            xs.push(0.0); // the real level start is the final stage
        }
        let stride = cfg.stride.max(1);
        let last = xs.len() - 1;
        let starts: Vec<f32> = xs
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0 || *i == last)
            .map(|(_, &x)| x)
            .collect();
        Ok(Self {
            starts,
            frontier: 0,
            window: VecDeque::with_capacity(cfg.window + 1),
            cfg,
            finished: false,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn starts(&self) -> &[f32] {
        &self.starts
    }
    pub fn len(&self) -> usize {
        self.starts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
    pub fn frontier(&self) -> usize {
        self.frontier
    }
    pub fn set_frontier(&mut self, f: usize) {
        self.frontier = f.min(self.starts.len() - 1);
        self.window.clear();
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn frontier_x(&self) -> f32 {
        self.starts[self.frontier]
    }

    /// (success rate, number of episodes) in the current window
    pub fn window_rate(&self) -> (f32, usize) {
        let n = self.window.len();
        if n == 0 {
            return (0.0, 0);
        }
        (self.window.iter().filter(|&&s| s).count() as f32 / n as f32, n)
    }

    /// 70% frontier / 30% uniformly from already-solved stages.
    pub fn sample(&mut self) -> (usize, f32) {
        let stage = if self.frontier == 0 || self.rng.random::<f32>() < self.cfg.frontier_frac {
            self.frontier
        } else {
            self.rng.random_range(0..self.frontier)
        };
        (stage, self.starts[stage])
    }

    pub fn record(&mut self, stage: usize, success: bool) -> Option<Advance> {
        if self.finished || stage != self.frontier {
            return None;
        }
        self.window.push_back(success);
        while self.window.len() > self.cfg.window {
            self.window.pop_front();
        }
        if self.window.len() < self.cfg.window {
            return None;
        }
        let (rate, _) = self.window_rate();
        if rate <= self.cfg.threshold {
            return None;
        }
        let from = self.frontier;
        self.window.clear();
        if self.frontier + 1 < self.starts.len() {
            self.frontier += 1;
            Some(Advance { from, to: self.frontier, rate, finished: false })
        } else {
            self.finished = true;
            Some(Advance { from, to: from, rate, finished: true })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cur(window: usize) -> Curriculum {
        let xs: Vec<f32> = (0..10).map(|i| 100.0 + i as f32 * 100.0).collect();
        Curriculum::new(&xs, CurriculumConfig { window, ..Default::default() }, 7).unwrap()
    }

    #[test]
    fn stages_are_end_to_start_with_zero_appended() {
        let c = cur(10);
        assert_eq!(c.len(), 11);
        assert_eq!(c.starts()[0], 1000.0);
        assert_eq!(*c.starts().last().unwrap(), 0.0);
        assert!(c.starts().windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn stride_keeps_last_stage() {
        let xs: Vec<f32> = (0..10).map(|i| 100.0 + i as f32 * 100.0).collect();
        let c = Curriculum::new(&xs, CurriculumConfig { stride: 4, ..Default::default() }, 1).unwrap();
        assert_eq!(c.starts(), &[1000.0, 600.0, 200.0, 0.0]);
    }

    #[test]
    fn advances_only_strictly_above_threshold() {
        let mut c = cur(200);
        for i in 0..200 {
            assert!(c.record(0, i % 20 != 0).is_none() || i == 199); // exactly 95% -> no advance
        }
        assert_eq!(c.frontier(), 0);

        let mut c = cur(200);
        let mut adv = None;
        for i in 0..200 {
            adv = c.record(0, i >= 9).or(adv); // 191/200 = 95.5%
        }
        assert_eq!(c.frontier(), 1);
        assert!(adv.is_some());
        assert_eq!(c.window_rate().1, 0); // window cleared on advance
    }

    #[test]
    fn ignores_non_frontier_episodes() {
        let mut c = cur(5);
        c.set_frontier(3);
        for _ in 0..50 {
            assert!(c.record(1, true).is_none());
        }
        assert_eq!(c.window_rate().1, 0);
    }

    #[test]
    fn sampling_is_70_30_over_solved_stages() {
        let mut c = cur(10);
        c.set_frontier(3);
        let n = 20000;
        let mut at_frontier = 0;
        for _ in 0..n {
            let (s, x) = c.sample();
            assert!(s <= 3);
            assert_eq!(x, c.starts()[s]);
            if s == 3 {
                at_frontier += 1;
            }
        }
        let f = at_frontier as f32 / n as f32;
        assert!((0.67..0.73).contains(&f), "frontier fraction {f}");
    }

    #[test]
    fn finishes_after_last_stage() {
        let mut c = cur(2);
        let last = c.len() - 1;
        c.set_frontier(last);
        assert!(c.record(last, true).is_none());
        let a = c.record(last, true).unwrap();
        assert!(a.finished && c.finished());
    }
}
