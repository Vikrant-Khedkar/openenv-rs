use std::collections::HashSet;

pub const MOVE_UP: i64 = 0;
pub const MOVE_DOWN: i64 = 1;
pub const MOVE_LEFT: i64 = 2;
pub const MOVE_RIGHT: i64 = 3;

pub const REWARD_EXIT: f64 = 10.0;
pub const PENALTY_MOVE: f64 = -0.05;
pub const PENALTY_VISITED: f64 = -0.25;
pub const PENALTY_IMPOSSIBLE_MOVE: f64 = -0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Win,
    Lose,
    Playing,
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Win => "win",
            Status::Lose => "lose",
            Status::Playing => "playing",
        }
    }
}

pub const DEFAULT_MAZE: [[u8; 8]; 8] = [
    [0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
];

/// Gridworld maze ported from OpenEnv's maze.py (itself derived from
/// erikdelange/Reinforcement-Learning-Maze, MIT). Cells are (col, row).
pub struct Maze {
    pub grid: Vec<Vec<u8>>,
    pub exit_cell: (i64, i64),
    pub current_cell: (i64, i64),
    pub previous_cell: (i64, i64),
    pub status: Status,
    total_reward: f64,
    minimum_reward: f64,
    visited: HashSet<(i64, i64)>,
}

impl Maze {
    pub fn new(
        grid: Vec<Vec<u8>>,
        start_cell: (i64, i64),
        exit_cell: Option<(i64, i64)>,
    ) -> Result<Self, String> {
        let nrows = grid.len() as i64;
        let ncols = grid[0].len() as i64;
        let exit_cell = exit_cell.unwrap_or((ncols - 1, nrows - 1));

        if exit_cell.0 < 0 || exit_cell.0 >= ncols || exit_cell.1 < 0 || exit_cell.1 >= nrows {
            return Err(format!("exit cell at {exit_cell:?} is not inside maze"));
        }
        if grid[exit_cell.1 as usize][exit_cell.0 as usize] == 1 {
            return Err(format!("exit cell at {exit_cell:?} is not free"));
        }

        let size = (nrows * ncols) as f64;
        let mut maze = Self {
            grid,
            exit_cell,
            current_cell: start_cell,
            previous_cell: start_cell,
            status: Status::Playing,
            total_reward: 0.0,
            minimum_reward: -0.5 * size,
            visited: HashSet::new(),
        };
        maze.reset(start_cell)?;
        Ok(maze)
    }

    pub fn reset(&mut self, start_cell: (i64, i64)) -> Result<(), String> {
        let nrows = self.grid.len() as i64;
        let ncols = self.grid[0].len() as i64;
        if start_cell.0 < 0 || start_cell.0 >= ncols || start_cell.1 < 0 || start_cell.1 >= nrows {
            return Err(format!("start cell at {start_cell:?} is not inside maze"));
        }
        if self.grid[start_cell.1 as usize][start_cell.0 as usize] == 1 {
            return Err(format!("start cell at {start_cell:?} is not free"));
        }
        if start_cell == self.exit_cell {
            return Err(format!(
                "start- and exit cell cannot be the same {start_cell:?}"
            ));
        }
        self.current_cell = start_cell;
        self.previous_cell = start_cell;
        self.total_reward = 0.0;
        self.visited.clear();
        self.status = Status::Playing;
        Ok(())
    }

    pub fn step(&mut self, action: i64) -> (f64, Status) {
        let reward = self.execute(action);
        self.total_reward += reward;
        self.status = self.compute_status();
        (reward, self.status)
    }

    fn execute(&mut self, action: i64) -> f64 {
        let possible = self.possible_actions(self.current_cell);
        if possible.is_empty() {
            return self.minimum_reward - 1.0;
        }
        if !possible.contains(&action) {
            return PENALTY_IMPOSSIBLE_MOVE;
        }

        let (mut col, mut row) = self.current_cell;
        match action {
            MOVE_LEFT => col -= 1,
            MOVE_UP => row -= 1,
            MOVE_RIGHT => col += 1,
            MOVE_DOWN => row += 1,
            _ => {}
        }
        self.previous_cell = self.current_cell;
        self.current_cell = (col, row);

        let reward = if self.current_cell == self.exit_cell {
            REWARD_EXIT
        } else if self.visited.contains(&self.current_cell) {
            PENALTY_VISITED
        } else {
            PENALTY_MOVE
        };
        self.visited.insert(self.current_cell);
        reward
    }

    pub fn possible_actions(&self, cell: (i64, i64)) -> Vec<i64> {
        let (col, row) = cell;
        let nrows = self.grid.len() as i64;
        let ncols = self.grid[0].len() as i64;
        let occupied = |r: i64, c: i64| self.grid[r as usize][c as usize] == 1;

        let mut actions = vec![MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN];
        if row == 0 || occupied(row - 1, col) {
            actions.retain(|&a| a != MOVE_UP);
        }
        if row == nrows - 1 || occupied(row + 1, col) {
            actions.retain(|&a| a != MOVE_DOWN);
        }
        if col == 0 || occupied(row, col - 1) {
            actions.retain(|&a| a != MOVE_LEFT);
        }
        if col == ncols - 1 || occupied(row, col + 1) {
            actions.retain(|&a| a != MOVE_RIGHT);
        }
        actions
    }

    fn compute_status(&self) -> Status {
        if self.current_cell == self.exit_cell {
            Status::Win
        } else if self.total_reward < self.minimum_reward {
            Status::Lose
        } else {
            Status::Playing
        }
    }
}
