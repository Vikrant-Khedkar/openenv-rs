use echo_like::EchoLikeEnv;
use openenv_client::{BlockingEnvClient, ClientError, EnvClient};
use openenv_core::ResetRequest;
use openenv_server::EnvServer;
use serde_json::json;

mod echo_like {
    use openenv_core::{EnvError, Environment, ResetRequest};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, JsonSchema)]
    pub struct A {
        pub message: String,
    }

    #[derive(Serialize, JsonSchema)]
    pub struct O {
        pub response: String,
        pub done: bool,
        pub reward: Option<f64>,
    }

    #[derive(Serialize, JsonSchema)]
    pub struct S {
        pub step_count: u64,
    }

    #[derive(Default)]
    pub struct EchoLikeEnv {
        step_count: u64,
    }

    impl Environment for EchoLikeEnv {
        type Action = A;
        type Observation = O;
        type State = S;

        fn reset(&mut self, _req: ResetRequest) -> Result<O, EnvError> {
            self.step_count = 0;
            Ok(O {
                response: "ready".into(),
                done: false,
                reward: None,
            })
        }

        fn step(&mut self, action: A) -> Result<O, EnvError> {
            self.step_count += 1;
            Ok(O {
                response: action.message,
                done: false,
                reward: Some(1.0),
            })
        }

        fn state(&self) -> S {
            S {
                step_count: self.step_count,
            }
        }
    }
}

async fn spawn_server() -> String {
    let server = EnvServer::with_config(
        EchoLikeEnv::default,
        openenv_core::ConcurrencyConfig {
            max_concurrent_envs: 8,
            session_timeout: None,
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = server.router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn async_client_round_trip() {
    let base = spawn_server().await;
    let mut client = EnvClient::connect(&base).await.unwrap();

    let r = client.reset(ResetRequest::default()).await.unwrap();
    assert_eq!(r.observation["response"], "ready");
    assert!(!r.done);

    let s = client.step(json!({"message": "hello rust"})).await.unwrap();
    assert_eq!(s.observation["response"], "hello rust");
    assert_eq!(s.reward, Some(1.0));

    let state = client.state().await.unwrap();
    assert_eq!(state, json!({"step_count": 1}));

    client.close().await.unwrap();
}

#[tokio::test]
async fn async_client_surfaces_server_errors() {
    let base = spawn_server().await;
    let mut client = EnvClient::connect(&base).await.unwrap();
    client.reset(ResetRequest::default()).await.unwrap();

    let err = client.step(json!({"nope": 1})).await.unwrap_err();
    match err {
        ClientError::Server { code, .. } => assert_eq!(code, "VALIDATION_ERROR"),
        other => panic!("expected server error, got {other:?}"),
    }
}

#[test]
fn blocking_client_round_trip() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let base = rt.block_on(spawn_server());
    let _guard = rt.enter();

    std::thread::spawn(move || {
        let mut client = BlockingEnvClient::connect(&base).unwrap();
        let r = client.reset(ResetRequest::default()).unwrap();
        assert_eq!(r.observation["response"], "ready");
        let s = client.step(json!({"message": "sync"})).unwrap();
        assert_eq!(s.observation["response"], "sync");
        client.close().unwrap();
    })
    .join()
    .unwrap();
}
