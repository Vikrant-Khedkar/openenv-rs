# openenv-rs

A Rust rebuild of [HuggingFace OpenEnv](https://github.com/huggingface/OpenEnv) — Gym-style agentic environments served over HTTP + WebSocket, **wire-compatible** with the Python implementation. Python clients talk to Rust servers and vice versa.

## Crates

| Crate | What it does |
|---|---|
| `openenv-core` | `Environment` trait, protocol types (Action/Observation/State), errors |
| `openenv-server` | axum HTTP + WebSocket env server: `/reset`, `/step`, `/state`, `/schema`, `/health`, `/metadata`, `/mcp`, `/ws` |
| `openenv-client` | Async WebSocket client + blocking wrapper + container providers (local Docker) |
| `openenv-mcp` | MCP tool registry: JSON-RPC `tools/list` / `tools/call` over HTTP and WS |
| `openenv-rubrics` | Reward computation: `Rubric` trait, LLM judge |
| `openenv-harness` | Rollout collection → `EpisodeRecord` JSONL (TRL-compatible) |
| `openenv-cli` | `openenv init / validate / serve / build / push` |

## Environments

Ported: `echo`, `grid-world`, `connect4`, `maze`, `snake`, `wildfire`, `chess` (shakmaty), `websearch`.

Not ported (locked to Python-only ecosystems): atari (ale-py), browsergym/openapp (playwright), carla, dm_control (mujoco), jupyter/coding/opencode/terminus (E2B SDK), openspiel, sumo, unity, textarena, finrl, kernrl (triton), and the smolagents/transformers-based envs.

## Quick start

```bash
cargo run -p echo-env            # serves on 0.0.0.0:8000
curl -s localhost:8000/health    # {"status":"healthy"}
curl -s -X POST localhost:8000/reset -H 'content-type: application/json' -d '{}'
curl -s -X POST localhost:8000/step -H 'content-type: application/json' \
  -d '{"action": {"message": "hello"}}'
```

## Wire protocol

Same as Python OpenEnv. WebSocket messages:

```json
{"type": "reset", "data": {"seed": 42}}
{"type": "step", "data": {"message": "hello"}}
{"type": "state", "data": {}}
{"type": "close", "data": {}}
```

Responses: `{"type": "observation", "data": {"observation": {...}, "reward": null, "done": false}}`, `{"type": "state", ...}`, `{"type": "error", "data": {"message": "...", "code": "VALIDATION_ERROR"}}`.

## License

BSD-3-Clause, same as upstream OpenEnv.
