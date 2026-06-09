#[tokio::main]
async fn main() -> std::io::Result<()> {
    openenv_server::serve_env(websearch_env::WebSearchEnvironment::default).await
}
