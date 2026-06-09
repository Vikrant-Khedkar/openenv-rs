use regex::Regex;
use serde_json::{json, Value};

use crate::Rubric;

/// Minimal completion client used by [`LlmJudge`].
pub trait LlmClient: Send {
    fn complete(&self, prompt: &str) -> Result<String, String>;
}

/// Chat-completions client for any OpenAI-compatible API.
pub struct OpenAiCompatClient {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl LlmClient for OpenAiCompatClient {
    fn complete(&self, prompt: &str) -> Result<String, String> {
        let resp: Value = ureq::post(&format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        ))
        .set("Authorization", &format!("Bearer {}", self.api_key))
        .send_json(json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }))
        .map_err(|e| e.to_string())?
        .into_json()
        .map_err(|e| e.to_string())?;

        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| format!("unexpected completion response: {resp}"))
    }
}

/// LLM-as-judge rubric: renders `{action}` / `{observation}` into a prompt
/// template, extracts the first numeric score from the response, and clamps
/// to [0, 1] when `normalize` is set. Mirrors Python's `LLMJudge`.
pub struct LlmJudge {
    pub prompt_template: String,
    pub default_score: f64,
    pub normalize: bool,
    client: Box<dyn LlmClient>,
    score_pattern: Regex,
}

impl LlmJudge {
    pub fn new(prompt_template: impl Into<String>, client: impl LlmClient + 'static) -> Self {
        Self {
            prompt_template: prompt_template.into(),
            default_score: 0.0,
            normalize: true,
            client: Box::new(client),
            score_pattern: Regex::new(r"(\d+\.?\d*)").expect("valid default pattern"),
        }
    }

    pub fn with_score_pattern(mut self, pattern: &str) -> Result<Self, regex::Error> {
        self.score_pattern = Regex::new(pattern)?;
        Ok(self)
    }

    fn render_prompt(&self, action: &Value, observation: &Value) -> String {
        self.prompt_template
            .replace("{action}", &action.to_string())
            .replace("{observation}", &observation.to_string())
    }

    fn parse_score(&self, response: &str) -> f64 {
        let Some(caps) = self.score_pattern.captures(response) else {
            return self.default_score;
        };
        let text = caps
            .get(1)
            .or_else(|| caps.get(0))
            .map(|m| m.as_str())
            .unwrap_or("");
        let Ok(mut score) = text.parse::<f64>() else {
            return self.default_score;
        };
        if self.normalize {
            score = score.clamp(0.0, 1.0);
        }
        score
    }
}

impl Rubric for LlmJudge {
    fn forward(&mut self, action: &Value, observation: &Value) -> f64 {
        let prompt = self.render_prompt(action, observation);
        match self.client.complete(&prompt) {
            Ok(response) => self.parse_score(&response),
            Err(_) => self.default_score,
        }
    }

    fn state_dict(&self) -> Value {
        json!({
            "prompt_template": self.prompt_template,
            "score_pattern": self.score_pattern.as_str(),
            "default_score": self.default_score,
            "normalize": self.normalize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Canned(&'static str);

    impl LlmClient for Canned {
        fn complete(&self, _prompt: &str) -> Result<String, String> {
            Ok(self.0.to_string())
        }
    }

    struct Failing;

    impl LlmClient for Failing {
        fn complete(&self, _prompt: &str) -> Result<String, String> {
            Err("down".into())
        }
    }

    #[test]
    fn parses_and_normalizes_scores() {
        let mut judge = LlmJudge::new("Rate {action} given {observation}", Canned("Score: 0.8"));
        assert_eq!(judge.forward(&json!({"a": 1}), &json!({"o": 2})), 0.8);

        let mut judge = LlmJudge::new("x", Canned("I give it 7 out of 10"));
        assert_eq!(judge.forward(&json!({}), &json!({})), 1.0);

        let mut judge = LlmJudge::new("x", Canned("no numbers here"));
        assert_eq!(judge.forward(&json!({}), &json!({})), 0.0);
    }

    #[test]
    fn client_failure_returns_default() {
        let mut judge = LlmJudge::new("x", Failing);
        judge.default_score = 0.5;
        assert_eq!(judge.forward(&json!({}), &json!({})), 0.5);
    }

    #[test]
    fn prompt_rendering_substitutes_placeholders() {
        let judge = LlmJudge::new("A={action} O={observation}", Canned("1"));
        let prompt = judge.render_prompt(&json!({"x": 1}), &json!("obs"));
        assert_eq!(prompt, r#"A={"x":1} O="obs""#);
    }
}
