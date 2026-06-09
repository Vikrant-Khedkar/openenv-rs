mod judge;
mod trajectory;

pub use judge::{LlmClient, LlmJudge, OpenAiCompatClient};
pub use trajectory::{ExponentialDiscounting, WinLossRubric};

use serde_json::{json, Value};

/// Reward computation over (action, observation) pairs, mirroring Python
/// OpenEnv's `Rubric`. Operates on JSON values since rubrics sit at the
/// protocol layer where payloads are dynamic.
pub trait Rubric: Send {
    fn forward(&mut self, action: &Value, observation: &Value) -> f64;

    /// Clear per-episode state. Called by envs on reset.
    fn reset(&mut self) {}

    fn state_dict(&self) -> Value {
        json!({})
    }
}

type Hook = Box<dyn Fn(&Value, &Value, f64) + Send>;

/// Wraps a rubric with pre/post forward hooks and `last_score` caching,
/// the counterpart of Python `Rubric.__call__`.
pub struct RubricRunner {
    rubric: Box<dyn Rubric>,
    pre_hooks: Vec<Box<dyn Fn(&Value, &Value) + Send>>,
    post_hooks: Vec<Hook>,
    last_score: Option<f64>,
}

impl RubricRunner {
    pub fn new(rubric: impl Rubric + 'static) -> Self {
        Self {
            rubric: Box::new(rubric),
            pre_hooks: vec![],
            post_hooks: vec![],
            last_score: None,
        }
    }

    pub fn register_pre_hook(&mut self, hook: impl Fn(&Value, &Value) + Send + 'static) {
        self.pre_hooks.push(Box::new(hook));
    }

    pub fn register_post_hook(&mut self, hook: impl Fn(&Value, &Value, f64) + Send + 'static) {
        self.post_hooks.push(Box::new(hook));
    }

    pub fn call(&mut self, action: &Value, observation: &Value) -> f64 {
        for hook in &self.pre_hooks {
            hook(action, observation);
        }
        let score = self.rubric.forward(action, observation);
        for hook in &self.post_hooks {
            hook(action, observation, score);
        }
        self.last_score = Some(score);
        score
    }

    pub fn last_score(&self) -> Option<f64> {
        self.last_score
    }

    pub fn reset(&mut self) {
        self.rubric.reset();
        self.last_score = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Constant(f64);

    impl Rubric for Constant {
        fn forward(&mut self, _a: &Value, _o: &Value) -> f64 {
            self.0
        }
    }

    #[test]
    fn runner_hooks_and_last_score() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runner = RubricRunner::new(Constant(0.7));
        let c = calls.clone();
        runner.register_pre_hook(move |_, _| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c = calls.clone();
        runner.register_post_hook(move |_, _, score| {
            assert_eq!(score, 0.7);
            c.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(runner.last_score(), None);
        let s = runner.call(&json!({}), &json!({}));
        assert_eq!(s, 0.7);
        assert_eq!(runner.last_score(), Some(0.7));
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        runner.reset();
        assert_eq!(runner.last_score(), None);
    }
}
