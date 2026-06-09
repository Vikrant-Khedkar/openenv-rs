mod maze;

use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

pub use maze::{Maze, Status, DEFAULT_MAZE};

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct MazeAction {
    pub action: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MazeObservation {
    pub legal_actions: Vec<i64>,
    pub current_position: Vec<i64>,
    pub previous_position: Vec<i64>,
    pub done: bool,
    pub reward: f64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MazeState {
    pub episode_id: Option<String>,
    pub step_count: u64,
    pub done: bool,
    pub current_position: Vec<i64>,
    pub exit_cell: Vec<i64>,
    pub status: String,
}

/// Gridworld maze: reach the exit cell. Moves cost -0.05, revisits -0.25,
/// hitting a wall -0.75; the exit pays +10. Cumulative reward below
/// -0.5 * maze size loses the game.
pub struct MazeEnvironment {
    maze: Maze,
    start_cell: (i64, i64),
    grid: Vec<Vec<u8>>,
    episode_id: String,
    step_count: u64,
    done: bool,
}

impl Default for MazeEnvironment {
    fn default() -> Self {
        let grid: Vec<Vec<u8>> = DEFAULT_MAZE.iter().map(|r| r.to_vec()).collect();
        let maze = Maze::new(grid.clone(), (0, 0), None).expect("default maze is valid");
        Self {
            maze,
            start_cell: (0, 0),
            grid,
            episode_id: Uuid::new_v4().to_string(),
            step_count: 0,
            done: false,
        }
    }
}

impl MazeEnvironment {
    fn observation(&self, reward: f64, done: bool) -> MazeObservation {
        let legal_actions = if done {
            vec![]
        } else {
            self.maze.possible_actions(self.maze.current_cell)
        };
        let mut metadata = Map::new();
        metadata.insert("maze".into(), json!(self.grid));
        metadata.insert("status".into(), json!(self.maze.status.as_str()));
        metadata.insert(
            "exit_cell".into(),
            json!([self.maze.exit_cell.0, self.maze.exit_cell.1]),
        );
        metadata.insert("step".into(), json!(self.step_count));

        MazeObservation {
            legal_actions,
            current_position: vec![self.maze.current_cell.0, self.maze.current_cell.1],
            previous_position: vec![self.maze.previous_cell.0, self.maze.previous_cell.1],
            done,
            reward,
            metadata,
        }
    }
}

impl Environment for MazeEnvironment {
    type Action = MazeAction;
    type Observation = MazeObservation;
    type State = MazeState;

    fn reset(&mut self, req: ResetRequest) -> Result<MazeObservation, EnvError> {
        let start = match req.extra.get("start_cell") {
            Some(v) => {
                let cell: Vec<i64> = serde_json::from_value(v.clone())
                    .map_err(|e| EnvError::Validation(format!("invalid start_cell: {e}")))?;
                if cell.len() != 2 {
                    return Err(EnvError::Validation("start_cell must be [col, row]".into()));
                }
                (cell[0], cell[1])
            }
            None => self.start_cell,
        };
        self.maze.reset(start).map_err(EnvError::Validation)?;
        self.episode_id = Uuid::new_v4().to_string();
        self.step_count = 0;
        self.done = false;
        Ok(self.observation(0.0, false))
    }

    fn step(&mut self, action: MazeAction) -> Result<MazeObservation, EnvError> {
        self.step_count += 1;
        let (reward, status) = self.maze.step(action.action);
        self.done = matches!(status, Status::Win | Status::Lose);
        Ok(self.observation(reward, self.done))
    }

    fn state(&self) -> MazeState {
        MazeState {
            episode_id: Some(self.episode_id.clone()),
            step_count: self.step_count,
            done: self.done,
            current_position: vec![self.maze.current_cell.0, self.maze.current_cell.1],
            exit_cell: vec![self.maze.exit_cell.0, self.maze.exit_cell.1],
            status: self.maze.status.as_str().into(),
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("maze_env", "Gridworld maze: navigate to the exit cell")
    }
}

#[cfg(test)]
mod tests {
    use super::maze::*;
    use super::*;

    #[test]
    fn rewards_match_python_constants() {
        let mut env = MazeEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();

        let obs = env.step(MazeAction { action: MOVE_UP }).unwrap();
        assert_eq!(obs.reward, PENALTY_IMPOSSIBLE_MOVE);

        let obs = env.step(MazeAction { action: MOVE_RIGHT }).unwrap();
        assert_eq!(obs.reward, PENALTY_MOVE);
        assert_eq!(obs.current_position, vec![1, 0]);

        let obs = env.step(MazeAction { action: MOVE_LEFT }).unwrap();
        assert_eq!(obs.reward, PENALTY_MOVE);

        let obs = env.step(MazeAction { action: MOVE_RIGHT }).unwrap();
        assert_eq!(obs.reward, PENALTY_VISITED);
    }

    #[test]
    fn solvable_path_wins() {
        let mut env = MazeEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        // Hand-traced path through DEFAULT_MAZE from (0,0) to (7,7).
        let path = [
            MOVE_RIGHT, MOVE_RIGHT, MOVE_DOWN, MOVE_DOWN, MOVE_LEFT, MOVE_LEFT, MOVE_DOWN,
            MOVE_DOWN, MOVE_RIGHT, MOVE_DOWN, MOVE_DOWN, MOVE_RIGHT, MOVE_RIGHT, MOVE_DOWN,
            MOVE_RIGHT, MOVE_RIGHT, MOVE_RIGHT, MOVE_RIGHT,
        ];
        let mut last = None;
        for a in path {
            last = Some(env.step(MazeAction { action: a }).unwrap());
        }
        let obs = last.unwrap();
        assert!(obs.done, "expected win, state: {obs:?}");
        assert_eq!(obs.reward, REWARD_EXIT);
        assert_eq!(obs.metadata["status"], "win");
    }

    #[test]
    fn custom_start_cell_via_reset() {
        let mut env = MazeEnvironment::default();
        let mut req = ResetRequest::default();
        req.extra
            .insert("start_cell".into(), serde_json::json!([6, 0]));
        let obs = env.reset(req).unwrap();
        assert_eq!(obs.current_position, vec![6, 0]);
    }

    #[test]
    fn wandering_loses() {
        let mut env = MazeEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        let mut done = false;
        for i in 0..200 {
            let a = if i.is_multiple_of(2) {
                MOVE_RIGHT
            } else {
                MOVE_LEFT
            };
            let obs = env.step(MazeAction { action: a }).unwrap();
            if obs.done {
                assert_eq!(obs.metadata["status"], "lose");
                done = true;
                break;
            }
        }
        assert!(done, "expected the game to end in a loss");
    }
}
