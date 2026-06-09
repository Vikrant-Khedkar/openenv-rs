use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

pub const ROWS: usize = 6;
pub const COLUMNS: usize = 7;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Connect4Action {
    pub column: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct Connect4Observation {
    pub board: Vec<Vec<i8>>,
    pub legal_actions: Vec<usize>,
    pub done: bool,
    pub reward: f64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct Connect4State {
    pub episode_id: Option<String>,
    pub step_count: u64,
    pub board: Vec<Vec<i8>>,
    pub next_player: i8,
}

/// Two-player Connect-4 on a 6x7 board. 1 = current player, -1 = opponent.
/// An invalid move ends the game with reward -1.
pub struct Connect4Environment {
    board: [[i8; COLUMNS]; ROWS],
    next_player: i8,
    episode_id: String,
    step_count: u64,
}

impl Default for Connect4Environment {
    fn default() -> Self {
        Self {
            board: [[0; COLUMNS]; ROWS],
            next_player: 1,
            episode_id: Uuid::new_v4().to_string(),
            step_count: 0,
        }
    }
}

impl Connect4Environment {
    fn board_vec(&self) -> Vec<Vec<i8>> {
        self.board.iter().map(|r| r.to_vec()).collect()
    }

    fn legal_actions(&self) -> Vec<usize> {
        (0..COLUMNS).filter(|&c| self.board[0][c] == 0).collect()
    }

    fn observation(&self, reward: f64, done: bool) -> Connect4Observation {
        let mut metadata = Map::new();
        metadata.insert("next_player".into(), json!(self.next_player));
        Connect4Observation {
            board: self.board_vec(),
            legal_actions: self.legal_actions(),
            done,
            reward,
            metadata,
        }
    }

    fn check_win_or_draw(&self, row: usize, col: usize) -> (f64, bool) {
        let player = self.board[row][col];
        for (dr, dc) in [(1i64, 0i64), (0, 1), (1, 1), (1, -1)] {
            let mut count = 0;
            for step in -3i64..4 {
                let r = row as i64 + step * dr;
                let c = col as i64 + step * dc;
                if (0..ROWS as i64).contains(&r)
                    && (0..COLUMNS as i64).contains(&c)
                    && self.board[r as usize][c as usize] == player
                {
                    count += 1;
                    if count >= 4 {
                        return (1.0, true);
                    }
                } else {
                    count = 0;
                }
            }
        }
        let full = self.board.iter().all(|r| r.iter().all(|&c| c != 0));
        (0.0, full)
    }
}

impl Environment for Connect4Environment {
    type Action = Connect4Action;
    type Observation = Connect4Observation;
    type State = Connect4State;

    fn reset(&mut self, _req: ResetRequest) -> Result<Connect4Observation, EnvError> {
        self.board = [[0; COLUMNS]; ROWS];
        self.next_player = 1;
        self.episode_id = Uuid::new_v4().to_string();
        self.step_count = 0;
        Ok(self.observation(0.0, false))
    }

    fn step(&mut self, action: Connect4Action) -> Result<Connect4Observation, EnvError> {
        let col = action.column;
        let (reward, done) =
            if !(0..COLUMNS as i64).contains(&col) || self.board[0][col as usize] != 0 {
                (-1.0, true)
            } else {
                let col = col as usize;
                let row = (0..ROWS)
                    .rev()
                    .find(|&r| self.board[r][col] == 0)
                    .expect("column has space");
                self.board[row][col] = self.next_player;
                self.check_win_or_draw(row, col)
            };

        self.next_player = -self.next_player;
        self.step_count += 1;
        Ok(self.observation(reward, done))
    }

    fn state(&self) -> Connect4State {
        Connect4State {
            episode_id: Some(self.episode_id.clone()),
            step_count: self.step_count,
            board: self.board_vec(),
            next_player: self.next_player,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("connect4_env", "Two-player Connect-4 on a 6x7 board")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(env: &mut Connect4Environment, col: i64) -> Connect4Observation {
        env.step(Connect4Action { column: col }).unwrap()
    }

    #[test]
    fn vertical_win() {
        let mut env = Connect4Environment::default();
        env.reset(ResetRequest::default()).unwrap();
        for _ in 0..3 {
            play(&mut env, 0);
            play(&mut env, 1);
        }
        let obs = play(&mut env, 0);
        assert!(obs.done);
        assert_eq!(obs.reward, 1.0);
    }

    #[test]
    fn invalid_move_ends_game() {
        let mut env = Connect4Environment::default();
        env.reset(ResetRequest::default()).unwrap();
        let obs = play(&mut env, 99);
        assert!(obs.done);
        assert_eq!(obs.reward, -1.0);
    }

    #[test]
    fn full_column_is_illegal() {
        let mut env = Connect4Environment::default();
        env.reset(ResetRequest::default()).unwrap();
        for _ in 0..ROWS {
            play(&mut env, 3);
        }
        let obs = env.observation(0.0, false);
        assert!(!obs.legal_actions.contains(&3));
        let obs = play(&mut env, 3);
        assert!(obs.done);
        assert_eq!(obs.reward, -1.0);
    }

    #[test]
    fn players_alternate() {
        let mut env = Connect4Environment::default();
        env.reset(ResetRequest::default()).unwrap();
        play(&mut env, 0);
        play(&mut env, 0);
        assert_eq!(env.board[ROWS - 1][0], 1);
        assert_eq!(env.board[ROWS - 2][0], -1);
    }
}
