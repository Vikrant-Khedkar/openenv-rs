use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const ASH: u8 = 0;
const FUEL: u8 = 1;
const BURNING: u8 = 2;
const BREAK: u8 = 3;
const WATER: u8 = 4;

const DIRS_8: [(&str, (i64, i64)); 9] = [
    ("N", (0, -1)),
    ("NE", (1, -1)),
    ("E", (1, 0)),
    ("SE", (1, 1)),
    ("S", (0, 1)),
    ("SW", (-1, 1)),
    ("W", (-1, 0)),
    ("NW", (-1, -1)),
    ("CALM", (0, 0)),
];

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WildfireAction {
    pub action: String,
    #[serde(default)]
    pub x: Option<i64>,
    #[serde(default)]
    pub y: Option<i64>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WildfireObservation {
    pub grid: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub step: u64,
    pub wind_dir: String,
    pub humidity: f64,
    pub burning_count: usize,
    pub burned_count: usize,
    pub reward_hint: f64,
    pub remaining_water: i64,
    pub remaining_breaks: i64,
    pub done: bool,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WildfireState {
    pub episode_id: String,
    pub step_count: u64,
    pub total_burned: usize,
    pub total_extinguished: usize,
    pub last_action: String,
    pub width: usize,
    pub height: usize,
    pub wind_dir: String,
    pub humidity: f64,
    pub remaining_water: i64,
    pub remaining_breaks: i64,
    pub grid: Vec<u8>,
    pub burn_timers: Vec<u32>,
}

/// Weather-aware wildfire simulation, ported from OpenEnv's wildfire_env.
/// Grid: 0 = ash, 1 = fuel, 2 = burning, 3 = firebreak, 4 = watered.
pub struct WildfireEnvironment {
    w: usize,
    h: usize,
    base_ignite_prob: f64,
    init_humidity: f64,
    init_sources: usize,
    max_steps: u64,
    init_water: i64,
    init_breaks: i64,
    burn_lifetime: u32,
    forced_wind: Option<String>,
    rng: StdRng,
    st: WildfireState,
}

impl Default for WildfireEnvironment {
    fn default() -> Self {
        Self::new(32, 32, 0.30, 0.25, 2, 3407, 128, 8, 50)
    }
}

impl WildfireEnvironment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: usize,
        base_ignite_prob: f64,
        humidity: f64,
        init_sources: usize,
        seed: u64,
        max_steps: u64,
        water_capacity: i64,
        break_capacity: i64,
    ) -> Self {
        let width = std::env::var("WILDFIRE_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(width);
        let height = std::env::var("WILDFIRE_HEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(height);
        let humidity = std::env::var("WILDFIRE_HUMIDITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(humidity);
        let forced_wind = std::env::var("WILDFIRE_WIND")
            .ok()
            .filter(|w| DIRS_8.iter().any(|(name, _)| name == w));

        Self {
            w: width,
            h: height,
            base_ignite_prob,
            init_humidity: humidity,
            init_sources,
            max_steps,
            init_water: water_capacity,
            init_breaks: break_capacity,
            burn_lifetime: 3,
            forced_wind,
            rng: StdRng::seed_from_u64(seed),
            st: WildfireState {
                episode_id: String::new(),
                step_count: 0,
                total_burned: 0,
                total_extinguished: 0,
                last_action: "reset".into(),
                width,
                height,
                wind_dir: "CALM".into(),
                humidity,
                remaining_water: water_capacity,
                remaining_breaks: break_capacity,
                grid: vec![],
                burn_timers: vec![],
            },
        }
    }

    fn idx(&self, x: i64, y: i64) -> usize {
        (y * self.w as i64 + x) as usize
    }

    fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && (x as usize) < self.w && y >= 0 && (y as usize) < self.h
    }

    fn burning_count(&self) -> usize {
        self.st.grid.iter().filter(|&&v| v == BURNING).count()
    }

    fn burned_count(&self) -> usize {
        self.st.grid.iter().filter(|&&v| v == ASH).count()
    }

    fn saved_cells(&self) -> usize {
        self.st.grid.iter().filter(|&&v| v != ASH).count()
    }

    fn is_done(&self) -> bool {
        self.burning_count() == 0 || self.st.step_count >= self.max_steps
    }

    fn apply_water(&mut self, x: i64, y: i64) -> f64 {
        if !self.in_bounds(x, y) {
            return -0.05;
        }
        if self.st.remaining_water <= 0 {
            return -0.5;
        }
        let i = self.idx(x, y);
        let reward = match self.st.grid[i] {
            BURNING => {
                self.st.grid[i] = WATER;
                self.st.burn_timers[i] = 0;
                self.st.total_extinguished += 1;
                0.25
            }
            FUEL => {
                self.st.grid[i] = WATER;
                self.st.burn_timers[i] = 0;
                -0.10
            }
            WATER => -0.05,
            _ => -0.05,
        };
        self.st.remaining_water -= 1;
        reward
    }

    fn apply_break(&mut self, x: i64, y: i64) -> f64 {
        if !self.in_bounds(x, y) {
            return -0.05;
        }
        let i = self.idx(x, y);
        let reward = match self.st.grid[i] {
            FUEL | WATER => {
                self.st.grid[i] = BREAK;
                self.st.burn_timers[i] = 0;
                0.15
            }
            BURNING => {
                self.st.grid[i] = BREAK;
                self.st.burn_timers[i] = 0;
                -0.02
            }
            BREAK => -0.01,
            _ => -0.02,
        };
        self.st.remaining_breaks -= 1;
        reward
    }

    fn spread_fire(&mut self) -> usize {
        let (w, h) = (self.w, self.h);
        let mut new_grid = self.st.grid.clone();
        let mut newly_burned = 0;

        let neighbors = [
            (-1i64, 0i64),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        let (wx, wy) = DIRS_8
            .iter()
            .find(|(name, _)| *name == self.st.wind_dir)
            .map(|(_, d)| *d)
            .unwrap_or((0, 0));

        let base = self.base_ignite_prob;
        let humidity_factor = 1.0 - self.st.humidity;
        let mut ignite_flags = vec![false; w * h];

        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let i = self.idx(x, y);
                if self.st.grid[i] != BURNING {
                    continue;
                }
                self.st.burn_timers[i] += 1;

                for (dx, dy) in neighbors {
                    let (nx, ny) = (x + dx, y + dy);
                    if !self.in_bounds(nx, ny) {
                        continue;
                    }
                    let ni = self.idx(nx, ny);
                    if self.st.grid[ni] != FUEL {
                        continue;
                    }

                    let wind_mult = if (dx, dy) == (wx, wy) {
                        2.0
                    } else if (dx, dy) == (-wx, -wy) {
                        0.5
                    } else {
                        1.0
                    };
                    let diag_mult = if dx != 0 && dy != 0 { 0.6 } else { 1.0 };
                    let p = (base * humidity_factor * wind_mult * diag_mult).clamp(0.0, 1.0);
                    if self.rng.gen::<f64>() < p {
                        ignite_flags[ni] = true;
                    }
                }
            }
        }

        for i in 0..self.st.grid.len() {
            let cell = self.st.grid[i];
            if cell == BURNING {
                if self.st.burn_timers[i] >= self.burn_lifetime {
                    new_grid[i] = ASH;
                    newly_burned += 1;
                } else {
                    new_grid[i] = BURNING;
                }
            } else if ignite_flags[i] && new_grid[i] == FUEL {
                new_grid[i] = BURNING;
                self.st.burn_timers[i] = 0;
            } else if cell == WATER {
                self.st.burn_timers[i] += 1;
                if self.st.burn_timers[i] >= 6 {
                    new_grid[i] = FUEL;
                }
            }
        }

        self.st.grid = new_grid;
        newly_burned
    }

    fn observation(&self, reward_hint: f64, done: bool, reward: f64) -> WildfireObservation {
        WildfireObservation {
            grid: self.st.grid.clone(),
            width: self.w,
            height: self.h,
            step: self.st.step_count,
            wind_dir: self.st.wind_dir.clone(),
            humidity: self.st.humidity,
            burning_count: self.burning_count(),
            burned_count: self.burned_count(),
            reward_hint,
            remaining_water: self.st.remaining_water,
            remaining_breaks: self.st.remaining_breaks,
            done,
            reward,
        }
    }
}

impl Environment for WildfireEnvironment {
    type Action = WildfireAction;
    type Observation = WildfireObservation;
    type State = WildfireState;

    fn reset(&mut self, req: ResetRequest) -> Result<WildfireObservation, EnvError> {
        if let Some(seed) = req.seed {
            self.rng = StdRng::seed_from_u64(seed);
        }
        let (w, h) = (self.w, self.h);
        let mut grid = vec![FUEL; w * h];

        let wind_dir = match &self.forced_wind {
            Some(wind) => wind.clone(),
            None => DIRS_8[self.rng.gen_range(0..DIRS_8.len())].0.to_string(),
        };
        let humidity = (self.init_humidity + self.rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);

        for _ in 0..self.init_sources {
            let x = self.rng.gen_range(0..w);
            let y = self.rng.gen_range(0..h);
            grid[y * w + x] = BURNING;
        }

        self.st = WildfireState {
            episode_id: Uuid::new_v4().to_string(),
            step_count: 0,
            total_burned: 0,
            total_extinguished: 0,
            last_action: "reset".into(),
            width: w,
            height: h,
            wind_dir,
            humidity,
            remaining_water: self.init_water,
            remaining_breaks: self.init_breaks,
            grid,
            burn_timers: vec![0; w * h],
        };

        Ok(self.observation(0.0, false, 0.0))
    }

    fn step(&mut self, action: WildfireAction) -> Result<WildfireObservation, EnvError> {
        let mut reward = 0.0;

        match (action.action.as_str(), action.x, action.y) {
            ("water", Some(x), Some(y)) if self.st.remaining_water > 0 => {
                reward += self.apply_water(x, y);
            }
            ("break", Some(x), Some(y)) if self.st.remaining_breaks > 0 => {
                reward += self.apply_break(x, y);
            }
            ("wait", _, _) => {}
            _ => reward -= 0.05,
        }

        let prev_burning = self.burning_count() as i64;
        let prev_burned = self.burned_count() as i64;

        let newly_burned = self.spread_fire();
        let new_burning = self.burning_count() as i64;
        let now_burned = self.burned_count() as i64;

        self.st.total_burned += newly_burned;
        self.st.step_count += 1;
        self.st.last_action = action.action;

        let spread_delta = new_burning - prev_burning;
        let burned_delta = now_burned - prev_burned;

        if spread_delta > 0 {
            reward -= 0.15 * spread_delta as f64;
        } else if spread_delta < 0 {
            reward += 0.10 * spread_delta.abs() as f64;
        }
        if burned_delta > 0 {
            reward -= 0.05 * burned_delta as f64;
        }
        reward -= 0.01;

        let done = self.is_done();
        if done {
            let total = (self.w * self.h) as f64;
            let saved_ratio = self.saved_cells() as f64 / total;
            let burned_ratio = now_burned as f64 / total;
            if self.burning_count() == 0 {
                reward += 0.5 + 0.5 * saved_ratio;
            }
            reward += 0.2 * (1.0 - burned_ratio);
        }

        Ok(self.observation(reward, done, reward))
    }

    fn state(&self) -> WildfireState {
        self.st.clone()
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            "wildfire_env",
            "Weather-aware wildfire suppression on a grid: water drops and firebreaks vs wind-driven spread",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_places_fires() {
        let mut env = WildfireEnvironment::default();
        let obs = env.reset(ResetRequest::default()).unwrap();
        assert_eq!(obs.grid.len(), 32 * 32);
        assert!(obs.burning_count > 0 && obs.burning_count <= 2);
        assert_eq!(obs.remaining_water, 8);
        assert_eq!(obs.remaining_breaks, 50);
    }

    #[test]
    fn water_extinguishes_burning_cell() {
        let mut env = WildfireEnvironment::default();
        let obs = env.reset(ResetRequest::default()).unwrap();
        let i = obs.grid.iter().position(|&v| v == BURNING).unwrap();
        let (x, y) = ((i % 32) as i64, (i / 32) as i64);
        env.step(WildfireAction {
            action: "water".into(),
            x: Some(x),
            y: Some(y),
        })
        .unwrap();
        assert_eq!(env.state().total_extinguished, 1);
    }

    #[test]
    fn episode_terminates() {
        let mut env = WildfireEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        let mut done = false;
        for _ in 0..200 {
            let obs = env
                .step(WildfireAction {
                    action: "wait".into(),
                    x: None,
                    y: None,
                })
                .unwrap();
            if obs.done {
                done = true;
                break;
            }
        }
        assert!(done, "fire should burn out or hit max_steps within 200");
    }

    #[test]
    fn invalid_action_penalized() {
        let mut env = WildfireEnvironment::default();
        env.reset(ResetRequest::default()).unwrap();
        let obs = env
            .step(WildfireAction {
                action: "bogus".into(),
                x: None,
                y: None,
            })
            .unwrap();
        assert!(obs.reward < 0.0);
    }
}
