use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
pub enum MoveAction {
    #[serde(rename = "UP")]
    Up,
    #[serde(rename = "DOWN")]
    Down,
    #[serde(rename = "LEFT")]
    Left,
    #[serde(rename = "RIGHT")]
    Right,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GridWorldAction {
    pub action: MoveAction,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct GridWorldObservation {
    pub x: i64,
    pub y: i64,
    pub message: String,
    pub reward: f64,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct GridWorldState {
    pub episode_id: Option<String>,
    pub step_count: u64,
}

/// 5x5 grid: the agent starts at [0, 0] and must reach [4, 4].
pub struct GridWorldEnvironment {
    grid_size: i64,
    goal: (i64, i64),
    x: i64,
    y: i64,
    episode_id: String,
    step_count: u64,
}

impl Default for GridWorldEnvironment {
    fn default() -> Self {
        Self {
            grid_size: 5,
            goal: (4, 4),
            x: 0,
            y: 0,
            episode_id: Uuid::new_v4().to_string(),
            step_count: 0,
        }
    }
}

impl Environment for GridWorldEnvironment {
    type Action = GridWorldAction;
    type Observation = GridWorldObservation;
    type State = GridWorldState;

    fn reset(&mut self, _req: ResetRequest) -> Result<GridWorldObservation, EnvError> {
        self.x = 0;
        self.y = 0;
        self.step_count = 0;
        self.episode_id = Uuid::new_v4().to_string();
        Ok(GridWorldObservation {
            x: 0,
            y: 0,
            message: "Welcome to Grid World! Goal is at [4, 4].".into(),
            reward: 0.0,
            done: false,
        })
    }

    fn step(&mut self, action: GridWorldAction) -> Result<GridWorldObservation, EnvError> {
        self.step_count += 1;
        match action.action {
            MoveAction::Up => self.x -= 1,
            MoveAction::Down => self.x += 1,
            MoveAction::Left => self.y -= 1,
            MoveAction::Right => self.y += 1,
        }
        self.x = self.x.clamp(0, self.grid_size - 1);
        self.y = self.y.clamp(0, self.grid_size - 1);

        let at_goal = (self.x, self.y) == self.goal;
        Ok(GridWorldObservation {
            x: self.x,
            y: self.y,
            message: if at_goal {
                "You found the goal!".into()
            } else {
                "Keep going...".into()
            },
            reward: if at_goal { 1.0 } else { -0.1 },
            done: at_goal,
        })
    }

    fn state(&self) -> GridWorldState {
        GridWorldState {
            episode_id: Some(self.episode_id.clone()),
            step_count: self.step_count,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            "grid_world_env",
            "5x5 grid navigation: reach the goal at [4, 4]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openenv_core::{DynEnv, DynEnvironment};
    use serde_json::json;

    #[test]
    fn reaches_goal() {
        let mut env = GridWorldEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        for _ in 0..4 {
            env.step(GridWorldAction {
                action: MoveAction::Down,
            })
            .unwrap();
        }
        let mut obs = None;
        for _ in 0..4 {
            obs = Some(
                env.step(GridWorldAction {
                    action: MoveAction::Right,
                })
                .unwrap(),
            );
        }
        let obs = obs.unwrap();
        assert!(obs.done);
        assert_eq!(obs.reward, 1.0);
        assert_eq!((obs.x, obs.y), (4, 4));
    }

    #[test]
    fn clamps_at_walls() {
        let mut env = GridWorldEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        let obs = env
            .step(GridWorldAction {
                action: MoveAction::Up,
            })
            .unwrap();
        assert_eq!((obs.x, obs.y), (0, 0));
        assert_eq!(obs.reward, -0.1);
    }

    #[test]
    fn wire_action_format_matches_python() {
        let mut env: Box<dyn DynEnvironment> = DynEnv::boxed(GridWorldEnvironment::default());
        env.reset(ResetRequest::default()).unwrap();
        let resp = env.step(json!({"action": "DOWN"})).unwrap();
        assert_eq!(resp.observation["x"], 1);
        assert_eq!(resp.reward, Some(-0.1));
    }
}
