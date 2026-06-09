use std::collections::VecDeque;

use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const EMPTY: u8 = 0;
const WALL: u8 = 1;
const FRUIT: u8 = 2;
const BODY: u8 = 3;
const HEAD: u8 = 4;
const CHANNELS: usize = 5;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SnakeAction {
    /// 0 = noop (keep direction), 1 = turn left, 2 = turn right
    pub action: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SnakeObservation {
    pub grid: Vec<Vec<u8>>,
    /// H x W x C one-hot encoding of the grid
    pub observation: Vec<Vec<Vec<f32>>>,
    pub episode_score: f64,
    pub episode_steps: u64,
    pub episode_fruits: u64,
    pub episode_kills: u64,
    pub alive: bool,
    pub done: bool,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SnakeState {
    pub episode_id: Option<String>,
    pub step_count: u64,
}

/// Single-agent snake on a walled grid. Native Rust implementation with the
/// same wire shapes as OpenEnv's marlenv-backed snake_env (observer='snake':
/// relative turn actions; fruit reward 1.0).
pub struct SnakeEnvironment {
    height: usize,
    width: usize,
    snake_length: usize,
    max_episode_steps: u64,
    rng: StdRng,
    snake: VecDeque<(usize, usize)>,
    direction: (i64, i64),
    fruit: (usize, usize),
    alive: bool,
    score: f64,
    fruits: u64,
    steps: u64,
    episode_id: String,
}

impl Default for SnakeEnvironment {
    fn default() -> Self {
        Self::new(20, 20, 3, 1000)
    }
}

impl SnakeEnvironment {
    pub fn new(height: usize, width: usize, snake_length: usize, max_episode_steps: u64) -> Self {
        Self {
            height,
            width,
            snake_length,
            max_episode_steps,
            rng: StdRng::seed_from_u64(0),
            snake: VecDeque::new(),
            direction: (0, 1),
            fruit: (0, 0),
            alive: false,
            score: 0.0,
            fruits: 0,
            steps: 0,
            episode_id: String::new(),
        }
    }

    fn grid(&self) -> Vec<Vec<u8>> {
        let mut grid = vec![vec![EMPTY; self.width]; self.height];
        for x in 0..self.width {
            grid[0][x] = WALL;
            grid[self.height - 1][x] = WALL;
        }
        for row in grid.iter_mut() {
            row[0] = WALL;
            row[self.width - 1] = WALL;
        }
        if self.alive {
            grid[self.fruit.0][self.fruit.1] = FRUIT;
        }
        for (i, &(r, c)) in self.snake.iter().enumerate() {
            grid[r][c] = if i == 0 { HEAD } else { BODY };
        }
        grid
    }

    fn one_hot(&self, grid: &[Vec<u8>]) -> Vec<Vec<Vec<f32>>> {
        grid.iter()
            .map(|row| {
                row.iter()
                    .map(|&cell| {
                        let mut c = vec![0.0; CHANNELS];
                        c[cell as usize] = 1.0;
                        c
                    })
                    .collect()
            })
            .collect()
    }

    fn spawn_fruit(&mut self) {
        loop {
            let r = self.rng.gen_range(1..self.height - 1);
            let c = self.rng.gen_range(1..self.width - 1);
            if !self.snake.contains(&(r, c)) {
                self.fruit = (r, c);
                return;
            }
        }
    }

    fn observation(&self, reward: f64, done: bool) -> SnakeObservation {
        let grid = self.grid();
        SnakeObservation {
            observation: self.one_hot(&grid),
            grid,
            episode_score: self.score,
            episode_steps: self.steps,
            episode_fruits: self.fruits,
            episode_kills: 0,
            alive: self.alive,
            done,
            reward,
        }
    }
}

impl Environment for SnakeEnvironment {
    type Action = SnakeAction;
    type Observation = SnakeObservation;
    type State = SnakeState;

    fn reset(&mut self, req: ResetRequest) -> Result<SnakeObservation, EnvError> {
        if let Some(seed) = req.seed {
            self.rng = StdRng::seed_from_u64(seed);
        }
        let mid_r = self.height / 2;
        let start_c = self.width / 2;
        self.snake = (0..self.snake_length)
            .map(|i| (mid_r, start_c - i))
            .collect();
        self.direction = (0, 1);
        self.alive = true;
        self.score = 0.0;
        self.fruits = 0;
        self.steps = 0;
        self.episode_id = req.episode_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        self.spawn_fruit();
        Ok(self.observation(0.0, false))
    }

    fn step(&mut self, action: SnakeAction) -> Result<SnakeObservation, EnvError> {
        if !self.alive {
            return Ok(self.observation(0.0, true));
        }
        self.steps += 1;

        // Relative turns: (dr, dc) rotated 90° left/right.
        self.direction = match action.action {
            1 => (-self.direction.1, self.direction.0),
            2 => (self.direction.1, -self.direction.0),
            _ => self.direction,
        };

        let head = self.snake[0];
        let new_head = (
            (head.0 as i64 + self.direction.0) as usize,
            (head.1 as i64 + self.direction.1) as usize,
        );

        let hit_wall = new_head.0 == 0
            || new_head.0 >= self.height - 1
            || new_head.1 == 0
            || new_head.1 >= self.width - 1;
        let hit_body = self.snake.contains(&new_head);

        if hit_wall || hit_body {
            self.alive = false;
            return Ok(self.observation(0.0, true));
        }

        self.snake.push_front(new_head);
        let mut reward = 0.0;
        if new_head == self.fruit {
            reward = 1.0;
            self.score += 1.0;
            self.fruits += 1;
            self.spawn_fruit();
        } else {
            self.snake.pop_back();
        }

        let done = self.steps >= self.max_episode_steps;
        Ok(self.observation(reward, done))
    }

    fn state(&self) -> SnakeState {
        SnakeState {
            episode_id: Some(self.episode_id.clone()),
            step_count: self.steps,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            "snake_env",
            "Single-agent snake on a walled grid with relative turn actions",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_with_seed(env: &mut SnakeEnvironment, seed: u64) -> SnakeObservation {
        env.reset(ResetRequest {
            seed: Some(seed),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn reset_shape_is_correct() {
        let mut env = SnakeEnvironment::default();
        let obs = reset_with_seed(&mut env, 7);
        assert_eq!(obs.grid.len(), 20);
        assert_eq!(obs.grid[0].len(), 20);
        assert_eq!(obs.observation.len(), 20);
        assert_eq!(obs.observation[0][0].len(), CHANNELS);
        assert!(obs.alive);
        let heads = obs.grid.iter().flatten().filter(|&&c| c == HEAD).count();
        assert_eq!(heads, 1);
    }

    #[test]
    fn snake_moves_forward_on_noop() {
        let mut env = SnakeEnvironment::default();
        reset_with_seed(&mut env, 7);
        let head_before = env.snake[0];
        env.step(SnakeAction { action: 0 }).unwrap();
        let head_after = env.snake[0];
        assert_eq!(head_after, (head_before.0, head_before.1 + 1));
    }

    #[test]
    fn hitting_wall_dies() {
        let mut env = SnakeEnvironment::default();
        reset_with_seed(&mut env, 7);
        let mut last = None;
        for _ in 0..20 {
            last = Some(env.step(SnakeAction { action: 0 }).unwrap());
            if last.as_ref().unwrap().done {
                break;
            }
        }
        let obs = last.unwrap();
        assert!(obs.done);
        assert!(!obs.alive);
    }

    #[test]
    fn eating_fruit_grows_and_rewards() {
        let mut env = SnakeEnvironment::default();
        reset_with_seed(&mut env, 7);
        let len_before = env.snake.len();
        // Teleport the fruit directly in front of the head.
        let head = env.snake[0];
        env.fruit = (head.0, head.1 + 1);
        let obs = env.step(SnakeAction { action: 0 }).unwrap();
        assert_eq!(obs.reward, 1.0);
        assert_eq!(obs.episode_fruits, 1);
        assert_eq!(env.snake.len(), len_before + 1);
    }
}
