use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "openenv", about = "OpenEnv-rs environment tooling", version)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scaffold a new environment crate
    Init {
        /// Environment name (kebab-case), also the target directory
        name: String,
        /// Parent directory to create the crate in
        #[arg(long, default_value = ".")]
        dir: PathBuf,
    },
    /// Validate an environment crate's structure
    Validate {
        #[arg(default_value = ".")]
        dir: PathBuf,
    },
    /// Run an environment server locally via cargo
    Serve {
        #[arg(default_value = ".")]
        dir: PathBuf,
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
    /// Build a Docker image for an environment
    Build {
        #[arg(default_value = ".")]
        dir: PathBuf,
        /// Docker build context (defaults to the env dir)
        #[arg(long)]
        context: Option<PathBuf>,
        #[arg(long)]
        tag: Option<String>,
    },
    /// Push an environment to a Hugging Face Space (Docker SDK)
    Push {
        /// Target repo, e.g. user/my-env
        repo_id: String,
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        /// Space secret KEY=VALUE (repeatable)
        #[arg(long = "secret")]
        secrets: Vec<String>,
        /// Space variable KEY=VALUE (repeatable)
        #[arg(long = "variable")]
        variables: Vec<String>,
    },
}

fn parse_kv(items: &[String]) -> Result<Vec<(String, String)>> {
    items
        .iter()
        .map(|item| {
            item.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .with_context(|| format!("expected KEY=VALUE, got '{item}'"))
        })
        .collect()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Init { name, dir } => {
            let target = dir.join(&name);
            openenv_cli::scaffold::init(&target, &name)?;
            println!("Created {}", target.display());
        }
        Cmd::Validate { dir } => {
            openenv_cli::validate(&dir)?;
            println!("OK: {} is a valid openenv-rs environment", dir.display());
        }
        Cmd::Serve { dir, port } => {
            openenv_cli::serve(&dir, port)?;
        }
        Cmd::Build { dir, context, tag } => {
            let name = dir
                .canonicalize()?
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "env".into());
            let tag = tag.unwrap_or(format!("{name}-env"));
            let context = context.unwrap_or_else(|| dir.clone());
            openenv_cli::build(&dir, &context, &tag)?;
            println!("Built image {tag}");
        }
        Cmd::Push {
            repo_id,
            dir,
            secrets,
            variables,
        } => {
            let Ok(token) = std::env::var("HF_TOKEN") else {
                bail!("HF_TOKEN is not set");
            };
            openenv_cli::validate(&dir)?;
            openenv_cli::hf::push(
                &dir,
                &repo_id,
                &token,
                &parse_kv(&secrets)?,
                &parse_kv(&variables)?,
            )?;
            println!("Pushed to https://huggingface.co/spaces/{repo_id}");
        }
    }
    Ok(())
}
