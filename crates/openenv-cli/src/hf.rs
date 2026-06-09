use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use base64::Engine;
use serde_json::json;

const HUB: &str = "https://huggingface.co";
const IGNORED_DIRS: [&str; 4] = [".git", "target", "__pycache__", ".venv"];

/// Push an env directory to a Hugging Face Space (Docker SDK), the
/// counterpart of `openenv push`. Needs HF_TOKEN.
pub fn push(
    dir: &Path,
    repo_id: &str,
    token: &str,
    secrets: &[(String, String)],
    variables: &[(String, String)],
) -> Result<()> {
    create_space(repo_id, token)?;

    let files = collect_files(dir)?;
    if files.is_empty() {
        bail!("no files to upload in {}", dir.display());
    }
    let payload = commit_payload("Upload via openenv-rs", dir, &files)?;
    let url = format!("{HUB}/api/spaces/{repo_id}/commit/main");
    let resp = ureq::post(&url)
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/x-ndjson")
        .send_string(&payload);
    check("commit", resp)?;

    for (key, value) in secrets {
        let resp = ureq::post(&format!("{HUB}/api/spaces/{repo_id}/secrets"))
            .set("Authorization", &format!("Bearer {token}"))
            .send_json(json!({"key": key, "value": value}));
        check("set secret", resp)?;
    }
    for (key, value) in variables {
        let resp = ureq::post(&format!("{HUB}/api/spaces/{repo_id}/variables"))
            .set("Authorization", &format!("Bearer {token}"))
            .send_json(json!({"key": key, "value": value}));
        check("set variable", resp)?;
    }
    Ok(())
}

fn create_space(repo_id: &str, token: &str) -> Result<()> {
    let (org, name) = repo_id
        .split_once('/')
        .context("repo_id must be 'owner/name'")?;
    let resp = ureq::post(&format!("{HUB}/api/repos/create"))
        .set("Authorization", &format!("Bearer {token}"))
        .send_json(json!({
            "type": "space",
            "name": name,
            "organization": org,
            "sdk": "docker",
            "private": false,
        }));
    match resp {
        Ok(_) => Ok(()),
        // 409 = already exists, fine for re-push
        Err(ureq::Error::Status(409, _)) => Ok(()),
        Err(e) => bail!("create space failed: {e}"),
    }
}

fn check(what: &str, resp: Result<ureq::Response, ureq::Error>) -> Result<()> {
    match resp {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            bail!("{what} failed with HTTP {code}: {body}")
        }
        Err(e) => bail!("{what} failed: {e}"),
    }
}

/// All uploadable files under `dir`, skipping VCS/build dirs.
pub fn collect_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = vec![];
    walk(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if !IGNORED_DIRS.contains(&name.as_str()) {
                walk(&path, out)?;
            }
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Build the NDJSON payload for the HF Hub commit API: a header line plus one
/// base64 file line per upload.
pub fn commit_payload(summary: &str, base: &Path, files: &[PathBuf]) -> Result<String> {
    let mut lines = vec![json!({
        "key": "header",
        "value": {"summary": summary, "description": ""},
    })
    .to_string()];

    for file in files {
        let rel = file
            .strip_prefix(base)
            .context("file outside base dir")?
            .to_string_lossy()
            .replace('\\', "/");
        let content = std::fs::read(file)?;
        lines.push(
            json!({
                "key": "file",
                "value": {
                    "path": rel,
                    "content": base64::engine::general_purpose::STANDARD.encode(&content),
                    "encoding": "base64",
                },
            })
            .to_string(),
        );
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn payload_has_header_and_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        std::fs::create_dir(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/lib.rs"), "pub fn x() {}").unwrap();
        std::fs::create_dir(tmp.path().join("target")).unwrap();
        std::fs::write(tmp.path().join("target/skip.bin"), "x").unwrap();

        let files = collect_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 2, "target/ must be ignored");

        let payload = commit_payload("msg", tmp.path(), &files).unwrap();
        let lines: Vec<Value> = payload
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["key"], "header");
        assert_eq!(lines[1]["key"], "file");
        assert_eq!(lines[1]["value"]["path"], "a.txt");
        assert_eq!(
            lines[1]["value"]["content"],
            base64::engine::general_purpose::STANDARD.encode("hello")
        );
    }
}
