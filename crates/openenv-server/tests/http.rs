use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use openenv_server::EnvServer;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct CounterAction {
    amount: i64,
}

#[derive(Serialize, JsonSchema)]
struct CounterObservation {
    total: i64,
    done: bool,
    reward: Option<f64>,
}

#[derive(Serialize, JsonSchema)]
struct CounterState {
    episode_id: Option<String>,
    step_count: u64,
}

#[derive(Default)]
struct CounterEnv {
    total: i64,
    episode_id: Option<String>,
    step_count: u64,
}

impl Environment for CounterEnv {
    type Action = CounterAction;
    type Observation = CounterObservation;
    type State = CounterState;

    fn reset(&mut self, req: ResetRequest) -> Result<CounterObservation, EnvError> {
        self.total = req.seed.unwrap_or(0) as i64;
        self.episode_id = req.episode_id;
        self.step_count = 0;
        Ok(CounterObservation {
            total: self.total,
            done: false,
            reward: None,
        })
    }

    fn step(&mut self, action: CounterAction) -> Result<CounterObservation, EnvError> {
        self.total += action.amount;
        self.step_count += 1;
        Ok(CounterObservation {
            total: self.total,
            done: self.total >= 10,
            reward: Some(action.amount as f64),
        })
    }

    fn state(&self) -> CounterState {
        CounterState {
            episode_id: self.episode_id.clone(),
            step_count: self.step_count,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("counter_env", "Test counter")
    }
}

async fn spawn_server() -> String {
    let server = EnvServer::new(CounterEnv::default);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = server.router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn http_endpoints_match_python_wire_format() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health, json!({"status": "healthy"}));

    let reset: serde_json::Value = client
        .post(format!("{base}/reset"))
        .json(&json!({"seed": 3, "episode_id": "ep-1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        reset,
        json!({"observation": {"total": 3}, "reward": null, "done": false})
    );

    let step: serde_json::Value = client
        .post(format!("{base}/step"))
        .json(&json!({"action": {"amount": 7}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        step,
        json!({"observation": {"total": 10}, "reward": 7.0, "done": true})
    );

    let state: serde_json::Value = client
        .get(format!("{base}/state"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(state, json!({"episode_id": "ep-1", "step_count": 1}));

    let metadata: serde_json::Value = client
        .get(format!("{base}/metadata"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(metadata["name"], "counter_env");

    let schema: serde_json::Value = client
        .get(format!("{base}/schema"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(schema["action"]["properties"]["amount"].is_object());
    assert!(schema["observation"]["properties"]["total"].is_object());
    assert!(schema["state"]["properties"]["step_count"].is_object());
}

#[tokio::test]
async fn invalid_action_returns_422() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/reset"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    let resp = client
        .post(format!("{base}/step"))
        .json(&json!({"action": {"bogus": true}}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 422);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());
}
