#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(connect4_env::Connect4Environment::default).await
}
