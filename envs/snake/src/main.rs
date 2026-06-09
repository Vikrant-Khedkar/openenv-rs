#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(snake_env::SnakeEnvironment::default).await
}
