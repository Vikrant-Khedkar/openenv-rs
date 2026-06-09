pub mod hf;
pub mod scaffold;

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Check that a directory looks like an openenv-rs environment crate.
pub fn validate(dir: &Path) -> Result<()> {
    let mut missing = vec![];
    for required in ["Cargo.toml", "src/lib.rs", "src/main.rs", "Dockerfile"] {
        if !dir.join(required).exists() {
            missing.push(required);
        }
    }
    if !missing.is_empty() {
        bail!("missing required files: {}", missing.join(", "));
    }
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml"))?;
    if !manifest.contains("openenv-server") {
        bail!("Cargo.toml does not depend on openenv-server");
    }
    Ok(())
}

/// `cargo run` the env server crate in `dir`, inheriting stdio.
pub fn serve(dir: &Path, port: u16) -> Result<()> {
    let status = Command::new("cargo")
        .args(["run", "--release"])
        .current_dir(dir)
        .env("PORT", port.to_string())
        .status()
        .context("failed to run cargo")?;
    if !status.success() {
        bail!("server exited with {status}");
    }
    Ok(())
}

/// `docker build` an env image. `context` is the build context (repo root for
/// in-workspace envs, the env dir for standalone ones).
pub fn build(dir: &Path, context: &Path, tag: &str) -> Result<()> {
    let dockerfile = dir.join("Dockerfile");
    let status = Command::new("docker")
        .args([
            "build",
            "-f",
            &dockerfile.to_string_lossy(),
            "-t",
            tag,
            &context.to_string_lossy(),
        ])
        .status()
        .context("failed to run docker")?;
    if !status.success() {
        bail!("docker build failed with {status}");
    }
    Ok(())
}
