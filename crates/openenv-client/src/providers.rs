use std::collections::HashMap;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::ClientError;

/// Launches and manages env server containers, mirroring Python's
/// `ContainerProvider` (start → base_url, health-poll, stop).
pub trait ContainerProvider: Send {
    fn start_container(
        &mut self,
        image: &str,
        port: Option<u16>,
        env_vars: &HashMap<String, String>,
    ) -> Result<String, ClientError>;

    fn stop_container(&mut self) -> Result<(), ClientError>;
}

/// Runs containers on the local Docker daemon via the `docker` CLI,
/// exactly like Python's `LocalDockerProvider`.
#[derive(Debug, Default)]
pub struct LocalDockerProvider {
    container_name: Option<String>,
}

impl LocalDockerProvider {
    pub fn new() -> Self {
        Self::default()
    }

    fn docker(args: &[&str]) -> Result<String, ClientError> {
        let out = Command::new("docker")
            .args(args)
            .output()
            .map_err(|e| ClientError::Connection(format!("failed to run docker: {e}")))?;
        if !out.status.success() {
            return Err(ClientError::Connection(format!(
                "docker {} failed: {}",
                args.first().unwrap_or(&""),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

impl ContainerProvider for LocalDockerProvider {
    fn start_container(
        &mut self,
        image: &str,
        port: Option<u16>,
        env_vars: &HashMap<String, String>,
    ) -> Result<String, ClientError> {
        let port = match port {
            Some(p) => p,
            None => free_port()?,
        };
        let name = container_name(image);

        let port_map = format!("{port}:8000");
        let mut args: Vec<String> = vec![
            "run".into(),
            "-d".into(),
            "--rm".into(),
            "-p".into(),
            port_map,
            "--name".into(),
            name.clone(),
        ];
        for (k, v) in env_vars {
            args.push("-e".into());
            args.push(format!("{k}={v}"));
        }
        args.push(image.into());

        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        Self::docker(&arg_refs)?;
        self.container_name = Some(name);
        Ok(format!("http://localhost:{port}"))
    }

    fn stop_container(&mut self) -> Result<(), ClientError> {
        if let Some(name) = self.container_name.take() {
            Self::docker(&["stop", &name])?;
        }
        Ok(())
    }
}

impl Drop for LocalDockerProvider {
    fn drop(&mut self) {
        let _ = self.stop_container();
    }
}

fn free_port() -> Result<u16, ClientError> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| ClientError::Connection(e.to_string()))?;
    Ok(listener
        .local_addr()
        .map_err(|e| ClientError::Connection(e.to_string()))?
        .port())
}

pub(crate) fn container_name(image: &str) -> String {
    let clean: String = image
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}-{ts}", clean.trim_matches('-'))
}

/// Poll `{base_url}/health` every 500ms until it returns 200, mirroring
/// Python's `wait_for_ready`.
pub async fn wait_for_ready(base_url: &str, timeout: Duration) -> Result<(), ClientError> {
    let client = reqwest::Client::new();
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        if Instant::now() >= deadline {
            return Err(ClientError::Connection(format!(
                "server at {base_url} not ready within {}s",
                timeout.as_secs_f64()
            )));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_name_is_sanitized() {
        let name = container_name("ghcr.io/foo/echo-env:latest");
        assert!(name.starts_with("ghcr-io-foo-echo-env-latest-"));
    }

    #[tokio::test]
    async fn wait_for_ready_times_out_fast() {
        let err = wait_for_ready("http://127.0.0.1:1", Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not ready"));
    }
}
