use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::error::EnvError;
use crate::types::{EnvironmentMetadata, ResetRequest, SchemaResponse, StepResponse};

/// Gym-style environment, mirroring Python OpenEnv's `Environment` base class.
///
/// Observation structs should include `done: bool` and `reward: Option<f64>`
/// fields; the server lifts them into the response envelope the same way
/// Python's `serialize_observation` does.
pub trait Environment: Send + 'static {
    type Action: DeserializeOwned + JsonSchema + Send;
    type Observation: Serialize + JsonSchema;
    type State: Serialize + JsonSchema;

    fn reset(&mut self, req: ResetRequest) -> Result<Self::Observation, EnvError>;
    fn step(&mut self, action: Self::Action) -> Result<Self::Observation, EnvError>;
    fn state(&self) -> Self::State;

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("Environment"),
            "OpenEnv environment",
        )
    }

    fn close(&mut self) {}
}

/// Object-safe wrapper over [`Environment`] operating on raw JSON, used by the
/// server so it can hold heterogeneous env instances behind one type.
pub trait DynEnvironment: Send {
    fn reset(&mut self, req: ResetRequest) -> Result<StepResponse, EnvError>;
    fn step(&mut self, action: Value) -> Result<StepResponse, EnvError>;
    fn state(&self) -> Result<Value, EnvError>;
    fn schemas(&self) -> SchemaResponse;
    fn metadata(&self) -> EnvironmentMetadata;
    fn close(&mut self);
}

/// Adapter that exposes a typed [`Environment`] as a [`DynEnvironment`].
/// A wrapper (rather than a blanket impl) so typed env methods stay
/// unambiguous at call sites.
pub struct DynEnv<E>(pub E);

impl<E: Environment> DynEnv<E> {
    pub fn boxed(env: E) -> Box<dyn DynEnvironment> {
        Box::new(Self(env))
    }
}

impl<E: Environment> DynEnvironment for DynEnv<E> {
    fn reset(&mut self, req: ResetRequest) -> Result<StepResponse, EnvError> {
        let obs = self.0.reset(req)?;
        split_observation(serialize(&obs)?)
    }

    fn step(&mut self, action: Value) -> Result<StepResponse, EnvError> {
        let action: E::Action =
            serde_json::from_value(action).map_err(|e| EnvError::Validation(e.to_string()))?;
        let obs = self.0.step(action)?;
        split_observation(serialize(&obs)?)
    }

    fn state(&self) -> Result<Value, EnvError> {
        serialize(&self.0.state())
    }

    fn schemas(&self) -> SchemaResponse {
        SchemaResponse {
            action: schema_value::<E::Action>(),
            observation: schema_value::<E::Observation>(),
            state: schema_value::<E::State>(),
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        self.0.metadata()
    }

    fn close(&mut self) {
        self.0.close()
    }
}

fn serialize<T: Serialize>(value: &T) -> Result<Value, EnvError> {
    serde_json::to_value(value).map_err(|e| EnvError::Execution(e.to_string()))
}

fn schema_value<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
}

/// Mirror of Python's `serialize_observation`: pull `done`/`reward` out of the
/// observation dict into the envelope and drop `metadata` from the payload.
pub fn split_observation(mut obs: Value) -> Result<StepResponse, EnvError> {
    let map = obs
        .as_object_mut()
        .ok_or_else(|| EnvError::Execution("observation must serialize to an object".into()))?;

    let done = map
        .remove("done")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let reward = map.remove("reward").and_then(|v| match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
        _ => None,
    });
    map.remove("metadata");

    Ok(StepResponse {
        observation: obs,
        reward,
        done,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize, JsonSchema)]
    struct EchoAction {
        message: String,
    }

    #[derive(Serialize, JsonSchema)]
    struct EchoObservation {
        response: String,
        done: bool,
        reward: Option<f64>,
    }

    #[derive(Serialize, JsonSchema)]
    struct EchoState {
        episode_id: Option<String>,
        step_count: u64,
    }

    #[derive(Default)]
    struct Echo {
        episode_id: Option<String>,
        step_count: u64,
    }

    impl Environment for Echo {
        type Action = EchoAction;
        type Observation = EchoObservation;
        type State = EchoState;

        fn reset(&mut self, req: ResetRequest) -> Result<EchoObservation, EnvError> {
            self.episode_id = req.episode_id;
            self.step_count = 0;
            Ok(EchoObservation {
                response: String::new(),
                done: false,
                reward: None,
            })
        }

        fn step(&mut self, action: EchoAction) -> Result<EchoObservation, EnvError> {
            self.step_count += 1;
            Ok(EchoObservation {
                response: action.message,
                done: false,
                reward: Some(1.0),
            })
        }

        fn state(&self) -> EchoState {
            EchoState {
                episode_id: self.episode_id.clone(),
                step_count: self.step_count,
            }
        }
    }

    #[test]
    fn dyn_env_round_trip() {
        let mut env: Box<dyn DynEnvironment> = DynEnv::boxed(Echo::default());

        let r = env
            .reset(ResetRequest {
                episode_id: Some("ep1".into()),
                ..Default::default()
            })
            .unwrap();
        assert!(!r.done);
        assert_eq!(r.observation, json!({"response": ""}));

        let s = env.step(json!({"message": "hello"})).unwrap();
        assert_eq!(s.observation, json!({"response": "hello"}));
        assert_eq!(s.reward, Some(1.0));

        let state = env.state().unwrap();
        assert_eq!(state, json!({"episode_id": "ep1", "step_count": 1}));
    }

    #[test]
    fn dyn_env_invalid_action_is_validation_error() {
        let mut env: Box<dyn DynEnvironment> = DynEnv::boxed(Echo::default());
        let err = env.step(json!({"wrong": 1})).unwrap_err();
        assert!(matches!(err, EnvError::Validation(_)));
    }

    #[test]
    fn split_observation_lifts_envelope_fields() {
        let r = split_observation(
            json!({"x": 1, "done": true, "reward": true, "metadata": {"k": "v"}}),
        )
        .unwrap();
        assert_eq!(r.observation, json!({"x": 1}));
        assert!(r.done);
        assert_eq!(r.reward, Some(1.0));
    }

    #[test]
    fn schemas_are_objects() {
        let env: Box<dyn DynEnvironment> = DynEnv::boxed(Echo::default());
        let s = env.schemas();
        assert!(s.action.is_object());
        assert!(s.observation.is_object());
        assert!(s.state.is_object());
    }
}
