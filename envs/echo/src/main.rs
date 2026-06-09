use openenv_server::{config_from_env, EnvServer};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    EnvServer::with_config(echo_env::EchoEnvironment::default, config_from_env())
        .with_mcp(echo_env::mcp_tools())
        .serve_default()
        .await
}
