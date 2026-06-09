//! Wire-compat probe: the Rust client against any OpenEnv server (Rust or
//! Python). Usage: cargo run -p openenv-client --example interop_probe -- <base_url> [mcp]

use openenv_client::EnvClient;
use openenv_core::ResetRequest;
use serde_json::json;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "http://localhost:8000".into());
    let mcp_mode = args.iter().any(|a| a == "mcp");

    let mut client = EnvClient::connect(&base).await.expect("connect");

    let r = client.reset(ResetRequest::default()).await.expect("reset");
    assert!(!r.done, "reset should not be done: {r:?}");
    println!("reset: {}", r.observation);

    let state = client.state().await.expect("state");
    println!("state: {state}");

    if mcp_mode {
        // Python echo_env is MCP-based: list tools, call echo_message, and
        // step with a CallToolAction payload.
        let tools = client
            .mcp(json!({"jsonrpc": "2.0", "method": "tools/list", "id": 1}))
            .await
            .expect("tools/list");
        println!("tools: {tools}");

        let call = client
            .mcp(json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo_message", "arguments": {"message": "from rust"}},
                "id": 2
            }))
            .await
            .expect("tools/call");
        println!("tools/call: {call}");

        let s = client
            .step(json!({
                "type": "call_tool",
                "tool_name": "echo_message",
                "arguments": {"message": "step from rust"}
            }))
            .await
            .expect("step");
        println!("step: {}", s.observation);
    } else {
        let s = client
            .step(json!({"message": "from rust"}))
            .await
            .expect("step");
        assert_eq!(s.observation["response"], "from rust");
        println!("step: {}", s.observation);
    }

    client.close().await.expect("close");
    println!("RUST CLIENT -> SERVER AT {base}: OK");
}
