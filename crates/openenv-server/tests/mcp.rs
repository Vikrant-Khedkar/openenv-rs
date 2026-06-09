use futures_util::{SinkExt, StreamExt};
use openenv_core::{EnvError, Environment, ResetRequest};
use openenv_mcp::ToolRegistry;
use openenv_server::EnvServer;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

#[derive(Deserialize, JsonSchema)]
struct A {
    #[allow(dead_code)]
    message: String,
}

#[derive(Serialize, JsonSchema)]
struct O {
    done: bool,
    reward: Option<f64>,
}

#[derive(Serialize, JsonSchema)]
struct S {
    step_count: u64,
}

#[derive(Default)]
struct Env;

impl Environment for Env {
    type Action = A;
    type Observation = O;
    type State = S;

    fn reset(&mut self, _req: ResetRequest) -> Result<O, EnvError> {
        Ok(O {
            done: false,
            reward: None,
        })
    }

    fn step(&mut self, _action: A) -> Result<O, EnvError> {
        Ok(O {
            done: false,
            reward: None,
        })
    }

    fn state(&self) -> S {
        S { step_count: 0 }
    }
}

fn tools() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(
        "shout",
        "Uppercase the input",
        json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        |args| {
            let text = args.get("text").and_then(|v| v.as_str()).ok_or("no text")?;
            Ok(json!(text.to_uppercase()))
        },
    )
    .unwrap();
    reg
}

async fn spawn() -> String {
    let server = EnvServer::new(Env::default).with_mcp(tools());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = server.router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn http_mcp_endpoint() {
    let base = spawn().await;
    let client = reqwest::Client::new();

    let resp: Value = client
        .post(format!("{base}/mcp"))
        .json(&json!({"jsonrpc": "2.0", "method": "tools/list", "id": 1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["result"]["tools"][0]["name"], "shout");

    let resp: Value = client
        .post(format!("{base}/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "shout", "arguments": {"text": "hi"}},
            "id": 2
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp, json!({"jsonrpc": "2.0", "result": "HI", "id": 2}));
}

#[tokio::test]
async fn ws_mcp_messages() {
    let base = spawn().await;
    let url = format!("ws://{}/ws", base.strip_prefix("http://").unwrap());
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    ws.send(Message::Text(
        json!({"type": "mcp", "data": {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "shout", "arguments": {"text": "abc"}}, "id": 7}})
            .to_string()
            .into(),
    ))
    .await
    .unwrap();

    let resp: Value = match ws.next().await.unwrap().unwrap() {
        Message::Text(t) => serde_json::from_str(&t).unwrap(),
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(resp["type"], "mcp");
    assert_eq!(resp["data"]["result"], "ABC");
    assert_eq!(resp["data"]["id"], 7);
}
