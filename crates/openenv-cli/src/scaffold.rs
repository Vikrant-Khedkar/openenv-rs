use std::path::Path;

use anyhow::{bail, Result};

/// Generate a new environment crate at `dir`, the counterpart of
/// `openenv init`'s template. Standalone by default: depends on openenv-rs
/// crates via git.
pub fn init(dir: &Path, name: &str) -> Result<()> {
    if dir.exists() {
        bail!("{} already exists", dir.display());
    }
    let snake = name.replace('-', "_");
    let pascal: String = snake
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect();

    std::fs::create_dir_all(dir.join("src"))?;

    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "{name}-env"
version = "0.1.0"
edition = "2021"

[dependencies]
openenv-core = {{ git = "https://github.com/Vikrant-Khedkar/openenv-rs" }}
openenv-server = {{ git = "https://github.com/Vikrant-Khedkar/openenv-rs" }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
schemars = "0.8"
tokio = {{ version = "1", features = ["full"] }}

[[bin]]
name = "{name}-env"
path = "src/main.rs"
"#
        ),
    )?;

    std::fs::write(
        dir.join("src/lib.rs"),
        format!(
            r#"use openenv_core::{{EnvError, Environment, EnvironmentMetadata, ResetRequest}};
use schemars::JsonSchema;
use serde::{{Deserialize, Serialize}};

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct {pascal}Action {{
    pub message: String,
}}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct {pascal}Observation {{
    pub response: String,
    pub done: bool,
    pub reward: Option<f64>,
}}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct {pascal}State {{
    pub episode_id: Option<String>,
    pub step_count: u64,
}}

#[derive(Default)]
pub struct {pascal}Environment {{
    episode_id: Option<String>,
    step_count: u64,
}}

impl Environment for {pascal}Environment {{
    type Action = {pascal}Action;
    type Observation = {pascal}Observation;
    type State = {pascal}State;

    fn reset(&mut self, req: ResetRequest) -> Result<Self::Observation, EnvError> {{
        self.episode_id = req.episode_id;
        self.step_count = 0;
        Ok({pascal}Observation {{
            response: "ready".into(),
            done: false,
            reward: None,
        }})
    }}

    fn step(&mut self, action: Self::Action) -> Result<Self::Observation, EnvError> {{
        self.step_count += 1;
        Ok({pascal}Observation {{
            response: action.message,
            done: false,
            reward: Some(1.0),
        }})
    }}

    fn state(&self) -> Self::State {{
        {pascal}State {{
            episode_id: self.episode_id.clone(),
            step_count: self.step_count,
        }}
    }}

    fn metadata(&self) -> EnvironmentMetadata {{
        EnvironmentMetadata::new("{snake}_env", "TODO: describe this environment")
    }}
}}
"#
        ),
    )?;

    std::fs::write(
        dir.join("src/main.rs"),
        format!(
            r#"#[tokio::main]
async fn main() -> std::io::Result<()> {{
    openenv_server::serve_env({snake}_env::{pascal}Environment::default).await
}}
"#
        ),
    )?;

    std::fs::write(
        dir.join("Dockerfile"),
        format!(
            r#"FROM rust:1.87-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/{name}-env /usr/local/bin/{name}-env
EXPOSE 8000
ENV PORT=8000
CMD ["{name}-env"]
"#
        ),
    )?;

    std::fs::write(
        dir.join("README.md"),
        format!("# {name}-env\n\nAn openenv-rs environment.\n\n```bash\ncargo run\ncurl -s localhost:8000/health\n```\n"),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_then_validate() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("my-cool");
        init(&dir, "my-cool").unwrap();
        crate::validate(&dir).unwrap();

        let lib = std::fs::read_to_string(dir.join("src/lib.rs")).unwrap();
        assert!(lib.contains("MyCoolEnvironment"));
        let main = std::fs::read_to_string(dir.join("src/main.rs")).unwrap();
        assert!(main.contains("my_cool_env::MyCoolEnvironment"));
    }

    #[test]
    fn init_refuses_existing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(init(tmp.path(), "x").is_err());
    }

    #[test]
    fn validate_rejects_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(crate::validate(tmp.path()).is_err());
    }
}
