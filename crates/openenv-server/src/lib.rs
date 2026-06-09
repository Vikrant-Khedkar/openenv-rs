mod http;

use std::net::SocketAddr;
use std::sync::Arc;

use openenv_core::{ConcurrencyConfig, DynEnvironment, Environment};
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
        let factory: EnvFactory = Arc::new(move || Box::new(factory()));
        let http_env = Arc::new(Mutex::new(factory()));
        Self {
            state: ServerState {
                factory,
                http_env,
                config,
                sessions: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_envs)),
            },
        }
    }

    pub fn router(&self) -> axum::Router {
        create_router(self.state.clone())
    }

    pub async fn serve(self, addr: SocketAddr) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("openenv server listening on {addr}");
        axum::serve(listener, self.router()).await
    }
}

/// Standard entrypoint for env server binaries: serves on 0.0.0.0:$PORT
/// (default 8000), mirroring the Python `uvicorn` setup.
pub async fn serve_env<E, F>(factory: F) -> std::io::Result<()>
where
    E: Environment,
    F: Fn() -> E + Send + Sync + 'static,
{
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let max_concurrent: usize = std::env::var("MAX_CONCURRENT_ENVS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    EnvServer::with_config(
        factory,
        ConcurrencyConfig {
            max_concurrent_envs: max_concurrent,
            session_timeout: None,
        },
    )
    .serve(SocketAddr::from(([0, 0, 0, 0], port)))
    .await
}
