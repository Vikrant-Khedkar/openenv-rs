use futures_util::{SinkExt, StreamExt};
use openenv_core::{ResetRequest, StepResponse};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::ClientError;
use crate::ws_url;

/// Async WebSocket client for OpenEnv servers, wire-compatible with the
/// Python `EnvClient`. One client = one server-side session.
pub struct EnvClient {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl EnvClient {
    /// Connect to a server by base URL (`http://host:port` or `ws://host:port/ws`).
    pub async fn connect(base_url: &str) -> Result<Self, ClientError> {
        let url = ws_url(base_url);
        let (ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .map_err(|e| ClientError::Connection(e.to_string()))?;
        Ok(Self { ws })
    }

    pub async fn reset(&mut self, req: ResetRequest) -> Result<StepResponse, ClientError> {
        let data = serde_json::to_value(&req).map_err(|e| ClientError::Protocol(e.to_string()))?;
        let resp = self.request(json!({"type": "reset", "data": data})).await?;
        parse_observation(resp)
    }

    /// Step with a raw action payload (the env-specific action object).
    pub async fn step(&mut self, action: Value) -> Result<StepResponse, ClientError> {
        let resp = self
            .request(json!({"type": "step", "data": action}))
            .await?;
        parse_observation(resp)
    }

    pub async fn state(&mut self) -> Result<Value, ClientError> {
        let resp = self.request(json!({"type": "state", "data": {}})).await?;
        expect_type(resp, "state")
    }

    /// Send a raw MCP JSON-RPC request over the session.
    pub async fn mcp(&mut self, rpc: Value) -> Result<Value, ClientError> {
        let resp = self.request(json!({"type": "mcp", "data": rpc})).await?;
        expect_type(resp, "mcp")
    }

    pub async fn close(mut self) -> Result<(), ClientError> {
        let msg = json!({"type": "close", "data": {}});
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .await
            .map_err(|e| ClientError::Connection(e.to_string()))?;
        let _ = self.ws.close(None).await;
        Ok(())
    }

    async fn request(&mut self, msg: Value) -> Result<Value, ClientError> {
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .await
            .map_err(|e| ClientError::Connection(e.to_string()))?;
        loop {
            match self.ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    return serde_json::from_str(&text)
                        .map_err(|e| ClientError::Protocol(e.to_string()));
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(Message::Close(_))) | None => return Err(ClientError::Closed),
                Some(Ok(_)) => continue,
                Some(Err(e)) => return Err(ClientError::Connection(e.to_string())),
            }
        }
    }
}

fn check_error(resp: &Value) -> Result<(), ClientError> {
    if resp["type"] == "error" {
        return Err(ClientError::Server {
            code: resp["data"]["code"].as_str().unwrap_or("UNKNOWN").into(),
            message: resp["data"]["message"].as_str().unwrap_or("").into(),
        });
    }
    Ok(())
}

fn expect_type(resp: Value, expected: &str) -> Result<Value, ClientError> {
    check_error(&resp)?;
    if resp["type"] != expected {
        return Err(ClientError::Protocol(format!(
            "expected '{expected}' response, got: {resp}"
        )));
    }
    Ok(resp["data"].clone())
}

fn parse_observation(resp: Value) -> Result<StepResponse, ClientError> {
    let data = expect_type(resp, "observation")?;
    serde_json::from_value(data).map_err(|e| ClientError::Protocol(e.to_string()))
}

/// Blocking wrapper around [`EnvClient`], mirroring Python's `SyncEnvClient`.
/// Owns a single-threaded tokio runtime.
pub struct BlockingEnvClient {
    rt: tokio::runtime::Runtime,
    inner: Option<EnvClient>,
}

impl BlockingEnvClient {
    pub fn connect(base_url: &str) -> Result<Self, ClientError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ClientError::Connection(e.to_string()))?;
        let inner = rt.block_on(EnvClient::connect(base_url))?;
        Ok(Self {
            rt,
            inner: Some(inner),
        })
    }

    pub fn reset(&mut self, req: ResetRequest) -> Result<StepResponse, ClientError> {
        let inner = self.inner.as_mut().ok_or(ClientError::Closed)?;
        self.rt.block_on(inner.reset(req))
    }

    pub fn step(&mut self, action: Value) -> Result<StepResponse, ClientError> {
        let inner = self.inner.as_mut().ok_or(ClientError::Closed)?;
        self.rt.block_on(inner.step(action))
    }

    pub fn state(&mut self) -> Result<Value, ClientError> {
        let inner = self.inner.as_mut().ok_or(ClientError::Closed)?;
        self.rt.block_on(inner.state())
    }

    pub fn close(&mut self) -> Result<(), ClientError> {
        match self.inner.take() {
            Some(inner) => self.rt.block_on(inner.close()),
            None => Ok(()),
        }
    }
}

impl Drop for BlockingEnvClient {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
