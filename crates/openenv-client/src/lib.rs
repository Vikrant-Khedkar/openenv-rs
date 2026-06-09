mod client;
mod error;

pub use client::{BlockingEnvClient, EnvClient};
pub use error::ClientError;
pub use openenv_core::StepResponse as StepResult;

/// Convert an http(s) base URL to the ws(s) URL of the `/ws` endpoint,
/// mirroring Python's `convert_to_ws_url`.
pub fn ws_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let converted = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if base.starts_with("ws://") || base.starts_with("wss://") {
        base.to_string()
    } else {
        format!("ws://{base}")
    };
    if converted.ends_with("/ws") {
        converted
    } else {
        format!("{converted}/ws")
    }
}

#[cfg(test)]
mod tests {
    use super::ws_url;

    #[test]
    fn ws_url_conversion() {
        assert_eq!(ws_url("http://localhost:8000"), "ws://localhost:8000/ws");
        assert_eq!(ws_url("https://host/"), "wss://host/ws");
        assert_eq!(ws_url("ws://host/ws"), "ws://host/ws");
        assert_eq!(ws_url("localhost:9000"), "ws://localhost:9000/ws");
    }
}
