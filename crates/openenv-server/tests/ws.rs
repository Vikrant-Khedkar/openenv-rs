use futures_util::{SinkExt, StreamExt};
use openenv_core::{ConcurrencyConfig, EnvError, Environment, ResetRequest};
use openenv_server::EnvServer;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

#[derive(Deserialize, JsonSchema)]
struct TestAction {
    value: i64,
}

#[derive(Serialize, JsonSchema)]
struct TestObservation {
    total: i64,
    done: bool,
    reward: Option<f64>,
}

#[derive(Serialize, JsonSchema)]
struct TestState {
    episode_id: Option<String>,
    step_count: u64,
}

#[derive(Default)]
struct TestEnv {
    total: i64,
    episode_id: Option<String>,
    step_count: u64,
}

impl Environment for TestEnv {
    type Action = TestAction;
    type Observation = TestObservation;
    type State = TestState;

    fn reset(&mut self, req: ResetRequest) -> Result<TestObservation, EnvError> {
        self.total = 0;
        self.episode_id = req.episode_id;
        self.step_count = 0;
        Ok(TestObservation {
            total: 0,
            done: false,
            reward: None,
        })
    }

    fn step(&mut self, action: TestAction) -> Result<TestObservation, EnvError> {
        self.total += action.value;
        self.step_count += 1;
        Ok(TestObservation {
            total: self.total,
            done: false,
            reward: Some(action.value as f64),
        })
    }

    fn state(&self) -> TestState {
        TestState {
            episode_id: self.episode_id.clone(),
            step_count: self.step_count,
        }
    }
}

async fn spawn_server(max_sessions: usize) -> String {
    let server = EnvServer::with_config(
        TestEnv::default,
        ConcurrencyConfig {
            max_concurrent_envs: max_sessions,
            session_timeout: None,
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = server.router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("ws://{addr}/ws")
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn round_trip(ws: &mut WsStream, msg: Value) -> Value {
    ws.send(Message::Text(msg.to_string().into()))
        .await
        .unwrap();
    loop {
        match ws.next().await.unwrap().unwrap() {
            Message::Text(t) => return serde_json::from_str(&t).unwrap(),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected frame: {other:?}"),
        }
    }
}

#[tokio::test]
async fn ws_session_full_episode() {
    let url = spawn_server(4).await;
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    let resp = round_trip(
        &mut ws,
        json!({"type": "reset", "data": {"episode_id": "ep-ws"}}),
    )
    .await;
    assert_eq!(
        resp,
        json!({"type": "observation", "data": {"observation": {"total": 0}, "reward": null, "done": false}})
    );

    let resp = round_trip(&mut ws, json!({"type": "step", "data": {"value": 5}})).await;
    assert_eq!(resp["data"]["observation"]["total"], 5);
    assert_eq!(resp["data"]["reward"], 5.0);

    let resp = round_trip(&mut ws, json!({"type": "state", "data": {}})).await;
    assert_eq!(
        resp,
        json!({"type": "state", "data": {"episode_id": "ep-ws", "step_count": 1}})
    );

    ws.send(Message::Text(
        json!({"type": "close", "data": {}}).to_string().into(),
    ))
    .await
    .unwrap();
}

#[tokio::test]
async fn ws_sessions_are_isolated() {
    let url = spawn_server(4).await;
    let (mut a, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut b, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    round_trip(&mut a, json!({"type": "reset", "data": {}})).await;
    round_trip(&mut b, json!({"type": "reset", "data": {}})).await;
    round_trip(&mut a, json!({"type": "step", "data": {"value": 100}})).await;

    let resp = round_trip(&mut b, json!({"type": "step", "data": {"value": 1}})).await;
    assert_eq!(resp["data"]["observation"]["total"], 1);
}

#[tokio::test]
async fn ws_error_codes() {
    let url = spawn_server(4).await;
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    ws.send(Message::Text("not json{{".into())).await.unwrap();
    let resp: Value = match ws.next().await.unwrap().unwrap() {
        Message::Text(t) => serde_json::from_str(&t).unwrap(),
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["data"]["code"], "INVALID_JSON");

    let resp = round_trip(&mut ws, json!({"type": "bogus"})).await;
    assert_eq!(resp["data"]["code"], "UNKNOWN_TYPE");

    let resp = round_trip(&mut ws, json!({"type": "step", "data": {"wrong": 1}})).await;
    assert_eq!(resp["data"]["code"], "VALIDATION_ERROR");

    // No MCP registry attached: JSON-RPC error inside an mcp response,
    // matching Python's "Environment does not support MCP".
    let resp = round_trip(
        &mut ws,
        json!({"type": "mcp", "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1}}),
    )
    .await;
    assert_eq!(resp["type"], "mcp");
    assert_eq!(resp["data"]["error"]["code"], -32603);
}

#[tokio::test]
async fn ws_capacity_reached() {
    let url = spawn_server(1).await;
    let (mut a, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    round_trip(&mut a, json!({"type": "reset", "data": {}})).await;
    let (mut b, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    let resp: Value = match b.next().await.unwrap().unwrap() {
        Message::Text(t) => serde_json::from_str(&t).unwrap(),
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["data"]["code"], "CAPACITY_REACHED");
    assert_eq!(resp["data"]["max_sessions"], 1);
}
