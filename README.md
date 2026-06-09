# openenv-rs

A Rust rebuild of [HuggingFace OpenEnv](https://github.com/huggingface/OpenEnv) — Gym-style agentic environments served over HTTP + WebSocket, **wire-compatible** with the Python implementation.

Interop is proven both directions against the real Python `openenv` package (`tests/interop/run.sh`):

- Python `GenericEnvClient` ↔ Rust echo server: reset / step / state over WS ✅
- Rust `EnvClient` ↔ Python `echo_env` (FastMCP) server: reset / state / MCP `tools/list` / `tools/call` / `step(CallToolAction)` ✅

## Crates

| Crate | What it does |
|---|---|
| `openenv-core` | `Environment` trait, protocol types, `split_observation` wire serialization |
| `openenv-server` | axum server: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `GET /metadata`, `GET /health`, `POST /mcp`, `GET /web`, `GET /ws` (sessions, capacity, error codes) |
| `openenv-client` | Async WS client + blocking wrapper + `LocalDockerProvider` (`from_docker_image`) |
| `openenv-mcp` | MCP tool registry: JSON-RPC `tools/list` / `tools/call` over HTTP and WS |
| `openenv-rubrics` | `Rubric` trait, trajectory rubrics with exponential discounting, `LlmJudge` |
| `openenv-harness` | `CollectRunner` → `EpisodeRecord` JSONL (TRL-compatible `messages` column) |
| `openenv-cli` | `openenv init / validate / serve / build / push` (push targets HF Spaces) |

## Environments

| Env | Action | Notes |
|---|---|---|
| `echo` | `{message}` | + MCP tools `echo_message`, `echo_with_length` |
| `grid-world` | `{action: UP\|DOWN\|LEFT\|RIGHT}` | 5x5 grid, goal at [4,4] |
| `connect4` | `{column}` | invalid move = -1 and game over |
| `maze` | `{action: 0..3}` | exact port of the reward model (-0.05/-0.25/-0.75/+10) |
| `snake` | `{action: 0..2}` | native Rust snake, marlenv-shaped observations |
| `wildfire` | `{action: water\|break\|wait, x, y}` | line-by-line port of the spread model |
| `chess` | `{move: "e2e4"}` | shakmaty rules; random opponent replaces moonfish |
| `websearch` | `{query}` | Serper.dev API, needs `SERPER_API_KEY` |

Not ported (locked to Python-only ecosystems): atari (ale-py), browsergym/openapp (playwright), carla, dm_control (mujoco), jupyter/coding/opencode/terminus (E2B SDK), openspiel, sumo, unity, textarena, finrl, kernrl (triton), reasoning_gym, and the smolagents/transformers-based envs. The Gradio UI is replaced by a static HTML console at `/web`.

## Quick start

```bash
cargo run -p echo-env            # serves on 0.0.0.0:8000
curl -s localhost:8000/health    # {"status":"healthy"}
curl -s -X POST localhost:8000/reset -H 'content-type: application/json' -d '{}'
curl -s -X POST localhost:8000/step -H 'content-type: application/json' \
  -d '{"action": {"message": "hello"}}'
open http://localhost:8000/web   # browser console
```

Rust client:

```rust
use openenv_client::EnvClient;
use openenv_core::ResetRequest;
use serde_json::json;

let mut env = EnvClient::connect("http://localhost:8000").await?;
let r = env.reset(ResetRequest::default()).await?;
let s = env.step(json!({"message": "hello"})).await?;
env.close().await?;
```

Docker (build from repo root, launch via provider):

```bash
docker build -f envs/echo/Dockerfile -t echo-env .
cargo test -p openenv-client --test docker -- --ignored
```

CLI:

```bash
cargo install --path crates/openenv-cli
openenv init my-env && cd my-env
openenv validate && openenv serve
HF_TOKEN=hf_xxx openenv push user/my-env --secret API_KEY=xyz
```

## Wire protocol

Same as Python OpenEnv. WebSocket `/ws` messages:

```json
{"type": "reset", "data": {"seed": 42, "episode_id": "ep-1"}}
{"type": "step", "data": {"message": "hello"}}
{"type": "state"}
{"type": "mcp", "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1}}
{"type": "close"}
```

Responses: `{"type": "observation", "data": {"observation": {...}, "reward": null, "done": false}}`, `{"type": "state", "data": {...}}`, `{"type": "mcp", "data": {jsonrpc response}}`, `{"type": "error", "data": {"message", "code"}}` with codes `INVALID_JSON`, `UNKNOWN_TYPE`, `VALIDATION_ERROR`, `EXECUTION_ERROR`, `CAPACITY_REACHED`.

Note `state`/`close` carry no `data` field — Python's models reject extra fields there. Observations follow Python's `serialize_observation`: `done`/`reward` are lifted into the envelope and `metadata` is stripped from the payload.

## License

BSD-3-Clause, same as upstream OpenEnv.
