use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One collected rollout, JSONL-serialized in the same shape as Python
/// OpenEnv's `EpisodeRecord` (TRL's SFTTrainer consumes `messages` directly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub episode_id: String,
    pub messages: Vec<Value>,
    pub reward: f64,
    pub done: bool,
    #[serde(default)]
    pub tool_trace: Vec<Value>,
    #[serde(default)]
    pub metrics: Map<String, Value>,
    #[serde(default)]
    pub verify_metrics: Map<String, Value>,
    #[serde(default)]
    pub artifacts: Map<String, Value>,
    #[serde(default)]
    pub extra: Map<String, Value>,
}

impl EpisodeRecord {
    pub fn new(messages: Vec<Value>, reward: f64, done: bool) -> Self {
        Self {
            episode_id: uuid::Uuid::new_v4().to_string(),
            messages,
            reward,
            done,
            tool_trace: vec![],
            metrics: Map::new(),
            verify_metrics: Map::new(),
            artifacts: Map::new(),
            extra: Map::new(),
        }
    }
}

pub type KeepFilter = Box<dyn Fn(&EpisodeRecord) -> bool>;

/// Drives repeated rollouts and writes kept episodes to a JSONL file,
/// mirroring Python's `CollectRunner` (filtering + checkpoint resume).
pub struct CollectRunner<F>
where
    F: FnMut(usize) -> Result<EpisodeRecord, String>,
{
    rollout: F,
    should_keep: Option<KeepFilter>,
}

impl<F> CollectRunner<F>
where
    F: FnMut(usize) -> Result<EpisodeRecord, String>,
{
    pub fn new(rollout: F) -> Self {
        Self {
            rollout,
            should_keep: None,
        }
    }

    pub fn with_filter(mut self, filter: impl Fn(&EpisodeRecord) -> bool + 'static) -> Self {
        self.should_keep = Some(Box::new(filter));
        self
    }

    /// Collect until `num_episodes` records are kept, appending each to
    /// `out_path` as it lands. If the file already has records (a resumed
    /// run), they count toward the target.
    pub fn collect(
        &mut self,
        num_episodes: usize,
        out_path: &Path,
    ) -> Result<Vec<EpisodeRecord>, String> {
        let mut records = read_jsonl(out_path)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out_path)
            .map_err(|e| e.to_string())?;

        let mut attempt = records.len();
        while records.len() < num_episodes {
            let record = (self.rollout)(attempt)?;
            attempt += 1;
            if let Some(filter) = &self.should_keep {
                if !filter(&record) {
                    continue;
                }
            }
            let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
            writeln!(file, "{line}").map_err(|e| e.to_string())?;
            records.push(record);
        }
        Ok(records)
    }
}

/// Read all episode records from a JSONL file (empty vec if absent).
pub fn read_jsonl(path: &Path) -> Result<Vec<EpisodeRecord>, String> {
    let Ok(file) = File::open(path) else {
        return Ok(vec![]);
    };
    BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line.map_err(|e| e.to_string())?;
            serde_json::from_str(&line).map_err(|e| e.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fake_rollout(i: usize) -> Result<EpisodeRecord, String> {
        let mut rec = EpisodeRecord::new(
            vec![
                json!({"role": "user", "content": format!("task {i}")}),
                json!({"role": "assistant", "content": "answer"}),
            ],
            if i % 2 == 0 { 1.0 } else { 0.0 },
            true,
        );
        rec.metrics.insert("steps".into(), json!(i));
        Ok(rec)
    }

    #[test]
    fn collects_and_writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodes.jsonl");

        let records = CollectRunner::new(fake_rollout).collect(3, &path).unwrap();
        assert_eq!(records.len(), 3);

        let read_back = read_jsonl(&path).unwrap();
        assert_eq!(read_back.len(), 3);
        assert_eq!(read_back[0].messages[0]["role"], "user");
    }

    #[test]
    fn filter_drops_zero_reward_episodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodes.jsonl");

        let records = CollectRunner::new(fake_rollout)
            .with_filter(|r| r.reward > 0.0)
            .collect(2, &path)
            .unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|r| r.reward > 0.0));
    }

    #[test]
    fn resume_counts_existing_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodes.jsonl");

        CollectRunner::new(fake_rollout).collect(2, &path).unwrap();

        let mut calls = 0;
        let records = CollectRunner::new(|i| {
            calls += 1;
            fake_rollout(i)
        })
        .collect(3, &path)
        .unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(calls, 1, "resume should only roll out the missing episode");
    }
}
