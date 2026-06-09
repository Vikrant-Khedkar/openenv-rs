use std::collections::HashMap;

use openenv_core::{EnvError, Environment, EnvironmentMetadata, ResetRequest};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position, Role};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ChessAction {
    /// UCI move, e.g. "e2e4" or "e7e8q"
    pub r#move: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ChessObservation {
    pub fen: String,
    pub legal_moves: Vec<String>,
    pub is_check: bool,
    pub result: Option<String>,
    pub done: bool,
    pub reward: f64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ChessState {
    pub episode_id: Option<String>,
    pub step_count: u64,
    pub fen: String,
    pub current_player: String,
    pub move_history: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opponent {
    Random,
    None,
}

/// Chess environment on shakmaty. The agent plays one color against a random
/// opponent (or self-play with `Opponent::None`). Upstream's moonfish engine
/// opponent/eval is replaced by a random opponent and material evaluation.
pub struct ChessEnvironment {
    opponent: Opponent,
    max_moves: u64,
    agent_color_setting: Option<Color>,
    gamma: f64,
    rng: StdRng,
    board: Chess,
    agent_color: Color,
    agent_move_count: u64,
    repetitions: HashMap<String, u32>,
    st: ChessState,
}

impl Default for ChessEnvironment {
    fn default() -> Self {
        Self::new(Opponent::Random, 500, None, 0.99)
    }
}

fn fen_of(pos: &Chess) -> String {
    Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string()
}

fn epd_of(pos: &Chess) -> String {
    let fen = fen_of(pos);
    fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

/// Simple material evaluation in centipawns from White's perspective,
/// replacing moonfish's PSQT board_evaluation.
fn material_evaluation(pos: &Chess) -> i64 {
    let board = pos.board();
    let value = |role: Role| match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 0,
    };
    let mut score = 0i64;
    for (_, piece) in board.clone() {
        let v = value(piece.role);
        score += if piece.color == Color::White { v } else { -v };
    }
    score
}

impl ChessEnvironment {
    pub fn new(opponent: Opponent, max_moves: u64, agent_color: Option<Color>, gamma: f64) -> Self {
        let mut env = Self {
            opponent,
            max_moves,
            agent_color_setting: agent_color,
            gamma,
            rng: StdRng::seed_from_u64(0xC4E55),
            board: Chess::default(),
            agent_color: Color::White,
            agent_move_count: 0,
            repetitions: HashMap::new(),
            st: ChessState {
                episode_id: None,
                step_count: 0,
                fen: fen_of(&Chess::default()),
                current_player: "white".into(),
                move_history: vec![],
            },
        };
        env.reset(ResetRequest::default()).expect("initial reset");
        env
    }

    fn record_position(&mut self) {
        *self.repetitions.entry(epd_of(&self.board)).or_insert(0) += 1;
    }

    fn is_threefold_repetition(&self) -> bool {
        self.repetitions
            .get(&epd_of(&self.board))
            .is_some_and(|&n| n >= 3)
    }

    fn reward_and_done(&self) -> (f64, bool) {
        if self.board.is_checkmate() {
            let winner = !self.board.turn();
            return if winner == self.agent_color {
                (1.0, true)
            } else {
                (-1.0, true)
            };
        }
        if self.board.is_stalemate()
            || self.board.is_insufficient_material()
            || self.board.halfmoves() >= 100
            || self.is_threefold_repetition()
            || self.st.step_count >= self.max_moves
        {
            return (0.0, true);
        }
        (0.0, false)
    }

    fn result_string(&self) -> String {
        if self.board.is_checkmate() {
            if self.board.turn() == Color::Black {
                "1-0".into()
            } else {
                "0-1".into()
            }
        } else {
            "1/2-1/2".into()
        }
    }

    fn legal_moves_uci(&self) -> Vec<String> {
        self.board
            .legal_moves()
            .iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect()
    }

    fn push_move(&mut self, m: &shakmaty::Move) {
        self.board.play_unchecked(m);
        self.st.step_count += 1;
        self.st
            .move_history
            .push(m.to_uci(CastlingMode::Standard).to_string());
        self.st.current_player = color_name(self.board.turn()).into();
        self.st.fen = fen_of(&self.board);
        self.record_position();
    }

    fn make_opponent_move(&mut self) {
        let moves = self.board.legal_moves();
        if moves.is_empty() {
            return;
        }
        match self.opponent {
            Opponent::Random => {
                let m = moves[self.rng.gen_range(0..moves.len())].clone();
                self.push_move(&m);
            }
            Opponent::None => {}
        }
    }

    fn observation(&self, reward: f64, done: bool) -> ChessObservation {
        let mut metadata = Map::new();
        metadata.insert("evaluation".into(), json!(material_evaluation(&self.board)));
        metadata.insert("fullmove_number".into(), json!(self.board.fullmoves()));
        metadata.insert("halfmove_clock".into(), json!(self.board.halfmoves()));

        if done && self.agent_move_count > 0 {
            let total = self.agent_move_count;
            let discounted: Vec<f64> = (0..total)
                .map(|t| self.gamma.powi((total - 1 - t) as i32) * reward)
                .collect();
            metadata.insert("discounted_rewards".into(), json!(discounted));
            metadata.insert("gamma".into(), json!(self.gamma));
        }

        ChessObservation {
            fen: fen_of(&self.board),
            legal_moves: self.legal_moves_uci(),
            is_check: self.board.is_check(),
            result: done.then(|| self.result_string()),
            done,
            reward,
            metadata,
        }
    }
}

impl Environment for ChessEnvironment {
    type Action = ChessAction;
    type Observation = ChessObservation;
    type State = ChessState;

    fn reset(&mut self, req: ResetRequest) -> Result<ChessObservation, EnvError> {
        if let Some(seed) = req.seed {
            self.rng = StdRng::seed_from_u64(seed);
        }
        self.board = match req.extra.get("fen").and_then(|v| v.as_str()) {
            Some(fen) => fen
                .parse::<Fen>()
                .map_err(|e| EnvError::Validation(format!("invalid FEN: {e}")))?
                .into_position(CastlingMode::Standard)
                .map_err(|e| EnvError::Validation(format!("invalid position: {e}")))?,
            None => Chess::default(),
        };

        let episode_id = Uuid::new_v4().to_string();
        self.agent_color = match self.agent_color_setting {
            Some(c) => c,
            None => {
                if self.rng.gen_bool(0.5) {
                    Color::White
                } else {
                    Color::Black
                }
            }
        };

        self.st = ChessState {
            episode_id: Some(episode_id),
            step_count: 0,
            current_player: color_name(self.board.turn()).into(),
            fen: fen_of(&self.board),
            move_history: vec![],
        };
        self.agent_move_count = 0;
        self.repetitions.clear();
        self.record_position();

        if self.opponent != Opponent::None && self.agent_color == Color::Black {
            self.make_opponent_move();
        }

        let (_, done) = self.reward_and_done();
        Ok(self.observation(0.0, done))
    }

    fn step(&mut self, action: ChessAction) -> Result<ChessObservation, EnvError> {
        let uci: UciMove = match action.r#move.parse() {
            Ok(u) => u,
            Err(_) => return Ok(self.observation(-0.1, false)),
        };
        let m = match uci.to_move(&self.board) {
            Ok(m) => m,
            Err(_) => return Ok(self.observation(-0.1, false)),
        };
        if !self.board.is_legal(&m) {
            return Ok(self.observation(-0.1, false));
        }

        self.push_move(&m);
        self.agent_move_count += 1;

        let (mut reward, mut done) = self.reward_and_done();
        if !done && self.opponent != Opponent::None {
            self.make_opponent_move();
            (reward, done) = self.reward_and_done();
        }

        Ok(self.observation(reward, done))
    }

    fn state(&self) -> ChessState {
        self.st.clone()
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new(
            "chess_env",
            "Chess vs a random opponent (shakmaty rules, material evaluation)",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_env() -> ChessEnvironment {
        ChessEnvironment::new(Opponent::None, 500, Some(Color::White), 0.99)
    }

    #[test]
    fn reset_gives_startpos() {
        let mut env = white_env();
        let obs = env.reset(ResetRequest::default()).unwrap();
        assert!(obs.fen.starts_with("rnbqkbnr/pppppppp"));
        assert_eq!(obs.legal_moves.len(), 20);
        assert!(!obs.done);
    }

    #[test]
    fn illegal_move_penalized_not_terminal() {
        let mut env = white_env();
        env.reset(ResetRequest::default()).unwrap();
        let obs = env
            .step(ChessAction {
                r#move: "e2e5".into(),
            })
            .unwrap();
        assert_eq!(obs.reward, -0.1);
        assert!(!obs.done);
        let obs = env
            .step(ChessAction {
                r#move: "garbage".into(),
            })
            .unwrap();
        assert_eq!(obs.reward, -0.1);
    }

    #[test]
    fn fools_mate_is_agent_loss() {
        // Self-play: play out fool's mate; final mating move is by Black while
        // the env scores from the agent's (White) perspective.
        let mut env = white_env();
        env.reset(ResetRequest::default()).unwrap();
        for m in ["f2f3", "e7e5", "g2g4"] {
            let obs = env.step(ChessAction { r#move: m.into() }).unwrap();
            assert!(!obs.done);
        }
        let obs = env
            .step(ChessAction {
                r#move: "d8h4".into(),
            })
            .unwrap();
        assert!(obs.done);
        assert_eq!(obs.reward, -1.0);
        assert_eq!(obs.result.as_deref(), Some("0-1"));
        assert!(obs.metadata.contains_key("discounted_rewards"));
    }

    #[test]
    fn custom_fen_reset() {
        let mut env = white_env();
        let mut req = ResetRequest::default();
        req.extra.insert(
            "fen".into(),
            json!("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        );
        let obs = env.reset(req).unwrap();
        assert!(obs.fen.contains("4P3"));
    }

    #[test]
    fn random_opponent_replies() {
        let mut env = ChessEnvironment::new(Opponent::Random, 500, Some(Color::White), 0.99);
        env.reset(ResetRequest {
            seed: Some(42),
            ..Default::default()
        })
        .unwrap();
        env.step(ChessAction {
            r#move: "e2e4".into(),
        })
        .unwrap();
        // Agent moved and opponent replied: 2 half-moves.
        assert_eq!(env.state().step_count, 2);
        assert_eq!(env.state().current_player, "white");
    }
}
