use serde_json::{json, Value};

use crate::Rubric;

/// Trajectory rubric with exponential discounting, mirroring Python's
/// `ExponentialDiscountingTrajectoryRubric`: accumulates (action, observation)
/// pairs, returns `intermediate_reward` until `done`, then the terminal score
/// from `score_fn`. `compute_step_rewards` assigns gamma^(T-1-t) * final.
pub struct ExponentialDiscounting<F>
where
    F: Fn(&[(Value, Value)]) -> f64 + Send,
{
    pub gamma: f64,
    pub intermediate_reward: f64,
    score_fn: F,
    trajectory: Vec<(Value, Value)>,
    final_score: Option<f64>,
}

impl<F> ExponentialDiscounting<F>
where
    F: Fn(&[(Value, Value)]) -> f64 + Send,
{
    pub fn new(gamma: f64, score_fn: F) -> Self {
        Self {
            gamma,
            intermediate_reward: 0.0,
            score_fn,
            trajectory: vec![],
            final_score: None,
        }
    }

    pub fn trajectory(&self) -> &[(Value, Value)] {
        &self.trajectory
    }

    pub fn compute_step_rewards(&self) -> Vec<f64> {
        let total = self.trajectory.len();
        let final_score = self.final_score.unwrap_or(0.0);
        (0..total)
            .map(|t| self.gamma.powi((total - 1 - t) as i32) * final_score)
            .collect()
    }
}

impl<F> Rubric for ExponentialDiscounting<F>
where
    F: Fn(&[(Value, Value)]) -> f64 + Send,
{
    fn forward(&mut self, action: &Value, observation: &Value) -> f64 {
        self.trajectory.push((action.clone(), observation.clone()));
        if observation["done"].as_bool().unwrap_or(false) {
            let score = (self.score_fn)(&self.trajectory);
            self.final_score = Some(score);
            score
        } else {
            self.intermediate_reward
        }
    }

    fn reset(&mut self) {
        self.trajectory.clear();
        self.final_score = None;
    }

    fn state_dict(&self) -> Value {
        json!({
            "gamma": self.gamma,
            "intermediate_reward": self.intermediate_reward,
            "trajectory_len": self.trajectory.len(),
        })
    }
}

type ScoreFn = fn(&[(Value, Value)]) -> f64;

/// Chess-style win/loss rubric: terminal score is the final observation's
/// reward (+1 win, -1 loss, 0 draw), discounted per step.
pub struct WinLossRubric {
    inner: ExponentialDiscounting<ScoreFn>,
}

fn last_reward(trajectory: &[(Value, Value)]) -> f64 {
    trajectory
        .last()
        .and_then(|(_, obs)| obs["reward"].as_f64())
        .unwrap_or(0.0)
}

impl WinLossRubric {
    pub fn new(gamma: f64) -> Self {
        Self {
            inner: ExponentialDiscounting::new(gamma, last_reward),
        }
    }

    pub fn compute_step_rewards(&self) -> Vec<f64> {
        self.inner.compute_step_rewards()
    }
}

impl Rubric for WinLossRubric {
    fn forward(&mut self, action: &Value, observation: &Value) -> f64 {
        self.inner.forward(action, observation)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }

    fn state_dict(&self) -> Value {
        self.inner.state_dict()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_intermediate_until_done() {
        let mut rubric = WinLossRubric::new(0.5);
        let a = json!({"move": "e2e4"});
        assert_eq!(
            rubric.forward(&a, &json!({"done": false, "reward": 0.0})),
            0.0
        );
        assert_eq!(
            rubric.forward(&a, &json!({"done": false, "reward": 0.0})),
            0.0
        );
        let final_score = rubric.forward(&a, &json!({"done": true, "reward": 1.0}));
        assert_eq!(final_score, 1.0);
    }

    #[test]
    fn discounted_step_rewards() {
        let mut rubric = WinLossRubric::new(0.5);
        let a = json!({});
        rubric.forward(&a, &json!({"done": false}));
        rubric.forward(&a, &json!({"done": false}));
        rubric.forward(&a, &json!({"done": true, "reward": -1.0}));
        assert_eq!(rubric.compute_step_rewards(), vec![-0.25, -0.5, -1.0]);
    }

    #[test]
    fn reset_clears_trajectory() {
        let mut rubric = WinLossRubric::new(0.9);
        rubric.forward(&json!({}), &json!({"done": true, "reward": 1.0}));
        rubric.reset();
        assert!(rubric.compute_step_rewards().is_empty());
    }
}
