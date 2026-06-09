#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(grid_world_env::GridWorldEnvironment::default).await
}
