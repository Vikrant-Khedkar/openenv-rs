mod http;
mod ws;

use std::net::SocketAddr;
use std::sync::Arc;

use openenv_core::{ConcurrencyConfig, DynEnvironment, Environment};
use openenv_mcp::ToolRegistry;
use tokio::sync::Mutex;

pub use http::create_router;

pub type EnvFactory = Arc<dyn Fn() -> Box<dyn DynEnvironment> + Send + Sync>;

#[derive(Clone)]
pub struct ServerState {
    pub factory: EnvFactory,
    /// Shared default instance backing the plain HTTP endpoints, matching
    /// Python's single-env HTTP mode. WS sessions get fresh factory instances.
    pub http_env: Arc<Mutex<Box<dyn DynEnvironment>>>,
    pub config: ConcurrencyConfig,
    pub sessions: Arc<tokio::sync::Semaphore>,
    pub mcp: Option<Arc<ToolRegistry>>,
}

pub struct EnvServer {
    state: ServerState,
}

impl EnvServer {
    pub fn new<E, F>(factory: F) -> Self
    where
        E: Environment,
        F: Fn() -> E + Send + Sync + 'static,
    {
        Self::with_config(factory, ConcurrencyConfig::default())
    }

    pub fn with_config<E, F>(factory: F, config: ConcurrencyConfig) -> Self
    where
        E: Environment,
        F: Fn() -> E + Send + Sync + 'static,
    {
        let factory: EnvFactory = Arc::new(move || openenv_core::DynEnv::boxed(factory()));
        let http_env = Arc::new(Mutex::new(factory()));
        Self {
            state: ServerState {
                factory,
                http_env,
                config,
                sessions: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_envs)),
                mcp: None,
            },
        }
    }

    /// Attach an MCP tool registry, served via `POST /mcp` and WS `mcp` messages.
    pub fn with_mcp(mut self, registry: ToolRegistry) -> Self {
        self.state.mcp = Some(Arc::new(registry));
        self
    }

    pub fn router(&self) -> axum::Router {
        create_router(self.state.clone())
    }

    pub async fn serve(self, addr: SocketAddr) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("openenv server listening on {addr}");
        axum::serve(listener, self.router()).await
    }

    /// Serve on 0.0.0.0:$PORT (default 8000) with tracing initialized,
    /// mirroring the Python `uvicorn` setup.
    pub async fn serve_default(self) -> std::io::Result<()> {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();
        let port: u16 = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8000);
        self.serve(SocketAddr::from(([0, 0, 0, 0], port))).await
    }
}

/// Concurrency config from env vars (`MAX_CONCURRENT_ENVS`, default 8).
pub fn config_from_env() -> ConcurrencyConfig {
    ConcurrencyConfig {
        max_concurrent_envs: std::env::var("MAX_CONCURRENT_ENVS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8),
        session_timeout: None,
    }
}

/// Dispatch a raw JSON-RPC payload to the server's MCP registry, mirroring
/// Python's `mcp_handler` error handling.
pub(crate) fn handle_mcp(
    state: &ServerState,
    raw: serde_json::Value,
) -> openenv_mcp::JsonRpcResponse {
    use openenv_mcp::{JsonRpcRequest, JsonRpcResponse, INTERNAL_ERROR, INVALID_REQUEST};

    let id = raw.get("id").cloned();
    let Some(registry) = &state.mcp else {
        return JsonRpcResponse::error(INTERNAL_ERROR, "Environment does not support MCP", id);
    };
    match serde_json::from_value::<JsonRpcRequest>(raw) {
        Ok(req) => registry.handle(req),
        Err(e) => JsonRpcResponse::error(INVALID_REQUEST, format!("Invalid request: {e}"), id),
    }
}

/// Standard entrypoint for env server binaries: serves on 0.0.0.0:$PORT
/// (default 8000), mirroring the Python `uvicorn` setup.
pub async fn serve_env<E, F>(factory: F) -> std::io::Result<()>
where
    E: Environment,
    F: Fn() -> E + Send + Sync + 'static,
{
    EnvServer::with_config(factory, config_from_env())
        .serve_default()
        .await
}
