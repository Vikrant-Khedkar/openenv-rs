#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(chess_env::ChessEnvironment::default).await
}
