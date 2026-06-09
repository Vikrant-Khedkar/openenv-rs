#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(wildfire_env::WildfireEnvironment::default).await
}
