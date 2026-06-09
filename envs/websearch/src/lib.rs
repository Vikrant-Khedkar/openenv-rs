use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WebSearchAction {
    pub query: String,
    #[serde(default)]
    pub temp_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebContent {
    pub title: String,
    pub content: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WebSearchObservation {
    pub content: String,
    pub web_contents: Vec<WebContent>,
    pub done: bool,
    pub reward: f64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WebSearchState {
    pub episode_id: Option<String>,
    pub step_count: u64,
}

/// Web search via the Serper.dev Google Search API (snippet results).
/// Requires SERPER_API_KEY unless the action carries `temp_api_key`.
pub struct WebSearchEnvironment {
    api_key: Option<String>,
    endpoint: String,
    top_k: usize,
    episode_id: String,
    step_count: u64,
}

impl Default for WebSearchEnvironment {
    fn default() -> Self {
        Self {
            api_key: std::env::var("SERPER_API_KEY").ok(),
            endpoint: "https://google.serper.dev/search".into(),
            top_k: 5,
            episode_id: Uuid::new_v4().to_string(),
            step_count: 0,
        }
    }
}

pub fn format_web_contents(web_contents: &[WebContent], query: &str) -> String {
    let mut lines = vec![format!("Search results for: {query}\n")];
    for (i, result) in web_contents.iter().enumerate() {
        lines.push(format!("[{}] {}", i + 1, result.title));
        let url = if result.url.is_empty() {
            "N/A"
        } else {
            &result.url
        };
        lines.push(format!("    URL: {url}"));
        let truncated: String = result.content.chars().take(500).collect();
        let ellipsis = if result.content.chars().count() > 500 {
            "..."
        } else {
            ""
        };
        lines.push(format!("    {truncated}{ellipsis}"));
        lines.push(String::new());
    }
    lines.join("\n")
}

impl WebSearchEnvironment {
    /// Override the search endpoint (used by tests to point at a mock server).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    fn search(&self, query: &str, api_key: &str) -> Result<Vec<WebContent>, String> {
        let resp: Value = ureq::post(&self.endpoint)
            .set("X-API-KEY", api_key)
            .set("Content-Type", "application/json")
            .send_json(json!({"q": query, "num": self.top_k, "gl": "us", "hl": "en"}))
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;

        let organic = resp["organic"].as_array().cloned().unwrap_or_default();
        Ok(organic
            .iter()
            .take(self.top_k)
            .map(|r| WebContent {
                title: r["title"].as_str().unwrap_or("").into(),
                content: r["snippet"].as_str().unwrap_or("").into(),
                url: r["link"].as_str().unwrap_or("").into(),
            })
            .collect())
    }

    fn error_observation(&self, query: &str, error: String) -> WebSearchObservation {
        let mut metadata = Map::new();
        metadata.insert("query".into(), json!(query));
        metadata.insert("error".into(), json!(error.clone()));
        WebSearchObservation {
            content: format!("[ERROR] Search failed due to: {error}"),
            web_contents: vec![],
            done: false,
            reward: 0.0,
            metadata,
        }
    }
}

impl Environment for WebSearchEnvironment {
    type Action = WebSearchAction;
    type Observation = WebSearchObservation;
    type State = WebSearchState;

    fn reset(&mut self, _req: ResetRequest) -> Result<WebSearchObservation, EnvError> {
        self.episode_id = Uuid::new_v4().to_string();
        self.step_count = 0;
        Ok(WebSearchObservation {
            content: String::new(),
            web_contents: vec![],
            done: false,
            reward: 0.0,
            metadata: Map::new(),
        })
    }

    fn step(&mut self, action: WebSearchAction) -> Result<WebSearchObservation, EnvError> {
        self.step_count += 1;
        let query = action.query.trim().to_string();

        let Some(api_key) = action.temp_api_key.clone().or_else(|| self.api_key.clone()) else {
            return Ok(self.error_observation(&query, "SERPER_API_KEY is not set".into()));
        };

        match self.search(&query, &api_key) {
            Ok(contents) if !contents.is_empty() => {
                let mut metadata = Map::new();
                metadata.insert("query".into(), json!(query));
                Ok(WebSearchObservation {
                    content: format_web_contents(&contents, &query),
                    web_contents: contents,
                    done: false,
                    reward: 0.0,
                    metadata,
                })
            }
            Ok(_) => {
                let mut metadata = Map::new();
                metadata.insert("query".into(), json!(query));
                metadata.insert("error".into(), json!("No search results found"));
                Ok(WebSearchObservation {
                    content: format!("[ERROR] No search results found for query: {query}"),
                    web_contents: vec![],
                    done: false,
                    reward: 0.0,
                    metadata,
                })
            }
            Err(e) => Ok(self.error_observation(&query, e)),
        }
    }

    fn state(&self) -> WebSearchState {
        WebSearchState {
            episode_id: Some(self.episode_id.clone()),
            step_count: self.step_count,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            "websearch_env",
            "Google search via the Serper.dev API, returning snippet results",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_results_like_python() {
        let contents = vec![WebContent {
            title: "Paris".into(),
            content: "Capital of France".into(),
            url: "https://example.com".into(),
        }];
        let s = format_web_contents(&contents, "capital of france");
        assert!(s.starts_with("Search results for: capital of france\n"));
        assert!(s.contains("[1] Paris"));
        assert!(s.contains("URL: https://example.com"));
    }

    #[test]
    fn missing_api_key_yields_error_observation() {
        let mut env = WebSearchEnvironment {
            api_key: None,
            ..Default::default()
        };
        env.reset(ResetRequest::default()).unwrap();
        let obs = env
            .step(WebSearchAction {
                query: "anything".into(),
                temp_api_key: None,
            })
            .unwrap();
        assert!(obs.content.starts_with("[ERROR]"));
        assert!(obs.web_contents.is_empty());
        assert!(!obs.done);
    }

    #[test]
    fn unreachable_endpoint_is_caught() {
        let mut env = WebSearchEnvironment {
            api_key: Some("k".into()),
            ..Default::default()
        }
        .with_endpoint("http://127.0.0.1:1/search");
        env.reset(ResetRequest::default()).unwrap();
        let obs = env
            .step(WebSearchAction {
                query: "x".into(),
                temp_api_key: None,
            })
            .unwrap();
        assert!(obs.content.starts_with("[ERROR] Search failed"));
    }
}
