use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: HealthStatus::Healthy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WsErrorCode {
    InvalidJson,
    UnknownType,
    ValidationError,
    ExecutionError,
    CapacityReached,
    FactoryError,
    SessionError,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResetRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRequest {
    pub action: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Envelope for both `/reset` and `/step` responses, and the `data` field of
/// WS observation messages: `{"observation": {...}, "reward": ..., "done": ...}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResponse {
    pub observation: Value,
    pub reward: Option<f64>,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentMetadata {
    pub name: String,
    pub description: String,
    pub readme_content: Option<String>,
    pub version: Option<String>,
    pub author: Option<String>,
    pub documentation_url: Option<String>,
}

impl EnvironmentMetadata {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            readme_content: None,
            version: Some("1.0.0".into()),
            author: None,
            documentation_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaResponse {
    pub action: Value,
    pub observation: Value,
    pub state: Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    pub max_concurrent_envs: usize,
    pub session_timeout: Option<f64>,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_envs: 1,
            session_timeout: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsIncoming {
    Reset {
        #[serde(default)]
        data: Map<String, Value>,
    },
    Step {
        data: Value,
    },
    State {
        #[serde(default)]
        data: Map<String, Value>,
    },
    Mcp {
        data: Value,
    },
    Close {
        #[serde(default)]
        data: Map<String, Value>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct WsErrorData {
    pub message: String,
    pub code: WsErrorCode,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl WsErrorData {
    pub fn new(code: WsErrorCode, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code,
            extra: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsOutgoing {
    Observation { data: StepResponse },
    State { data: Value },
    Mcp { data: Value },
    Error { data: WsErrorData },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ws_incoming_parses_python_shapes() {
        let m: WsIncoming =
            serde_json::from_value(json!({"type": "reset", "data": {"seed": 42}})).unwrap();
        match m {
            WsIncoming::Reset { data } => assert_eq!(data["seed"], json!(42)),
            _ => panic!("wrong variant"),
        }

        let m: WsIncoming =
            serde_json::from_value(json!({"type": "step", "data": {"message": "hi"}})).unwrap();
        match m {
            WsIncoming::Step { data } => assert_eq!(data["message"], "hi"),
            _ => panic!("wrong variant"),
        }

        assert!(serde_json::from_value::<WsIncoming>(json!({"type": "state"})).is_ok());
        assert!(serde_json::from_value::<WsIncoming>(json!({"type": "close", "data": {}})).is_ok());
    }

    #[test]
    fn ws_outgoing_matches_python_shapes() {
        let out = WsOutgoing::Observation {
            data: StepResponse {
                observation: json!({"response": "hi"}),
                reward: None,
                done: false,
            },
        };
        assert_eq!(
            serde_json::to_value(&out).unwrap(),
            json!({"type": "observation", "data": {"observation": {"response": "hi"}, "reward": null, "done": false}})
        );

        let err = WsOutgoing::Error {
            data: WsErrorData::new(WsErrorCode::ValidationError, "bad"),
        };
        assert_eq!(
            serde_json::to_value(&err).unwrap(),
            json!({"type": "error", "data": {"message": "bad", "code": "VALIDATION_ERROR"}})
        );
    }

    #[test]
    fn health_serializes_lowercase() {
        assert_eq!(
            serde_json::to_value(HealthResponse::default()).unwrap(),
            json!({"status": "healthy"})
        );
    }

    #[test]
    fn step_request_keeps_extra_fields() {
        let req: StepRequest =
            serde_json::from_value(json!({"action": {"v": 1}, "timeout_s": 3.0, "render": true}))
                .unwrap();
        assert_eq!(req.extra["render"], json!(true));
    }
}
