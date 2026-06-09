#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(echo_env::EchoEnvironment::default).await
}
