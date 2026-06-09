use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct EchoAction {
    pub message: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EchoObservation {
    pub response: String,
    pub done: bool,
    pub reward: Option<f64>,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EchoState {
    pub episode_id: Option<String>,
    pub step_count: u64,
    pub last_message: String,
}

#[derive(Debug, Default)]
pub struct EchoEnvironment {
    episode_id: Option<String>,
    step_count: u64,
    last_message: String,
}

impl Environment for EchoEnvironment {
    type Action = EchoAction;
    type Observation = EchoObservation;
    type State = EchoState;

    fn reset(&mut self, req: ResetRequest) -> Result<EchoObservation, EnvError> {
        self.episode_id = Some(req.episode_id.unwrap_or_else(|| Uuid::new_v4().to_string()));
        self.step_count = 0;
        self.last_message.clear();
        Ok(EchoObservation {
            response: "Echo environment ready".into(),
            done: false,
            reward: None,
            metadata: Map::new(),
        })
    }

    fn step(&mut self, action: EchoAction) -> Result<EchoObservation, EnvError> {
        self.step_count += 1;
        self.last_message = action.message.clone();
        Ok(EchoObservation {
            response: action.message,
            done: false,
            reward: Some(1.0),
            metadata: Map::new(),
        })
    }

    fn state(&self) -> EchoState {
        EchoState {
            episode_id: self.episode_id.clone(),
            step_count: self.step_count,
            last_message: self.last_message.clone(),
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("echo_env", "Echoes back messages sent to it")
    }
}

/// MCP tools matching Python echo_env: `echo_message` and `echo_with_length`.
pub fn mcp_tools() -> openenv_mcp::ToolRegistry {
    use serde_json::json;

    let message_schema = json!({
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    });

    let mut reg = openenv_mcp::ToolRegistry::new();
    reg.register(
        "echo_message",
        "Echo back the provided message",
        message_schema.clone(),
        |args| {
            let msg = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or("missing 'message' argument")?;
            Ok(json!(msg))
        },
    )
    .expect("valid tool name");
    reg.register(
        "echo_with_length",
        "Echo back the message with its length",
        message_schema,
        |args| {
            let msg = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or("missing 'message' argument")?;
            Ok(json!(format!("{msg} (length: {})", msg.chars().count())))
        },
    )
    .expect("valid tool name");
    reg
}
