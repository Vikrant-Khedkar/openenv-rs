use openenv_client::EnvClient;
use openenv_core::ResetRequest;
use serde_json::json;

/// Full Docker round-trip: requires the echo-env image built locally:
///   docker build -f envs/echo/Dockerfile -t echo-env .
/// Run with: cargo test -p openenv-client --test docker -- --ignored
#[tokio::test]
#[ignore = "requires docker and a locally built echo-env image"]
async fn docker_echo_round_trip() {
    let mut client = EnvClient::from_docker_image("echo-env").await.unwrap();

    let r = client.reset(ResetRequest::default()).await.unwrap();
    assert!(!r.done);

    let s = client.step(json!({"message": "via docker"})).await.unwrap();
    assert_eq!(s.observation["response"], "via docker");

    client.close().await.unwrap();
}
