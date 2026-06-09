use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use openenv_core::{ResetRequest, WsErrorCode, WsErrorData, WsIncoming, WsOutgoing};
use serde_json::json;
use std::time::Duration;

use crate::ServerState;

pub async fn ws_handler(State(state): State<ServerState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_session(state, socket))
}

async fn send(socket: &mut WebSocket, msg: &WsOutgoing) -> Result<(), axum::Error> {
    let text = serde_json::to_string(msg).expect("WsOutgoing serializes");
    socket.send(Message::Text(text.into())).await
}

async fn send_error(socket: &mut WebSocket, data: WsErrorData) {
    let _ = send(socket, &WsOutgoing::Error { data }).await;
}

async fn handle_session(state: ServerState, mut socket: WebSocket) {
    let Ok(_permit) = state.sessions.clone().try_acquire_owned() else {
        let max = state.config.max_concurrent_envs;
        let mut data = WsErrorData::new(
            WsErrorCode::CapacityReached,
            format!("Server at capacity: {max} active sessions (max {max})"),
        );
        data.extra.insert("active_sessions".into(), json!(max));
        data.extra.insert("max_sessions".into(), json!(max));
        send_error(&mut socket, data).await;
        return;
    };

    let mut env = (state.factory)();
    let session_id = uuid::Uuid::new_v4();
    tracing::debug!("ws session {session_id} opened");

    loop {
        let received = match state.config.session_timeout {
            Some(secs) => {
                match tokio::time::timeout(Duration::from_secs_f64(secs), socket.recv()).await {
                    Ok(r) => r,
                    Err(_) => {
                        tracing::debug!("ws session {session_id} timed out");
                        break;
                    }
                }
            }
            None => socket.recv().await,
        };

        let Some(Ok(message)) = received else { break };
        let text = match message {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let parsed: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                send_error(
                    &mut socket,
                    WsErrorData::new(WsErrorCode::InvalidJson, format!("Invalid JSON: {e}")),
                )
                .await;
                continue;
            }
        };

        let msg_type = parsed
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let incoming: WsIncoming = match serde_json::from_value(parsed) {
            Ok(m) => m,
            Err(e) => {
                let (code, message) = match msg_type.as_str() {
                    "reset" | "step" | "state" | "mcp" | "close" => (
                        WsErrorCode::ValidationError,
                        format!("Invalid message: {e}"),
                    ),
                    other => (
                        WsErrorCode::UnknownType,
                        format!("Unknown message type: {other}"),
                    ),
                };
                send_error(&mut socket, WsErrorData::new(code, message)).await;
                continue;
            }
        };

        let response = match incoming {
            WsIncoming::Reset { data } => {
                match serde_json::from_value::<ResetRequest>(serde_json::Value::Object(data)) {
                    Ok(req) => env.reset(req).map(|data| WsOutgoing::Observation { data }),
                    Err(e) => Err(openenv_core::EnvError::Validation(e.to_string())),
                }
            }
            WsIncoming::Step { data } => {
                env.step(data).map(|data| WsOutgoing::Observation { data })
            }
            WsIncoming::State { .. } => env.state().map(|data| WsOutgoing::State { data }),
            WsIncoming::Mcp { data } => {
                let resp = crate::handle_mcp(&state, data);
                Ok(WsOutgoing::Mcp {
                    data: serde_json::to_value(resp).expect("JsonRpcResponse serializes"),
                })
            }
            WsIncoming::Close { .. } => break,
        };

        match response {
            Ok(out) => {
                if send(&mut socket, &out).await.is_err() {
                    break;
                }
            }
            Err(openenv_core::EnvError::Validation(msg)) => {
                let mut data = WsErrorData::new(WsErrorCode::ValidationError, "Invalid message");
                data.extra.insert("errors".into(), json!([msg]));
                send_error(&mut socket, data).await;
            }
            Err(e) => {
                send_error(
                    &mut socket,
                    WsErrorData::new(WsErrorCode::ExecutionError, e.to_string()),
                )
                .await;
            }
        }
    }

    env.close();
    tracing::debug!("ws session {session_id} closed");
}
