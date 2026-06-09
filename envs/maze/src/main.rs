#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(maze_env::MazeEnvironment::default).await
}
