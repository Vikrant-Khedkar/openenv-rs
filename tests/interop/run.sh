#!/usr/bin/env bash
# Wire-compat proof, both directions, against the real Python OpenEnv.
# Prereqs: uv, a clone of https://github.com/huggingface/OpenEnv
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OPENENV_PY="${OPENENV_PY:-/tmp/OpenEnv}"
RUST_PORT=18432
PY_PORT=18433

[ -d "$OPENENV_PY" ] || { echo "Python OpenEnv not found at $OPENENV_PY (set OPENENV_PY)"; exit 1; }

if [ ! -d "$HERE/.venv" ]; then
  uv venv "$HERE/.venv"
  uv pip install --python "$HERE/.venv/bin/python" "$OPENENV_PY"
fi

cleanup() { kill "${RUST_PID:-}" "${PY_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== direction 1: Python client -> Rust echo server ==="
cargo build -q -p echo-env
PORT=$RUST_PORT "$ROOT/target/debug/echo-env" & RUST_PID=$!
sleep 1
"$HERE/.venv/bin/python" "$HERE/python_client_vs_rust.py" "http://localhost:$RUST_PORT"
kill $RUST_PID

echo "=== direction 2: Rust client -> Python echo server ==="
(cd "$OPENENV_PY" && PYTHONPATH="$OPENENV_PY/src:$OPENENV_PY" \
  "$HERE/.venv/bin/python" -m uvicorn envs.echo_env.server.app:app --port $PY_PORT) & PY_PID=$!
sleep 4
cargo run -q -p openenv-client --example interop_probe -- "http://localhost:$PY_PORT" mcp
kill $PY_PID

echo "=== interop: ALL OK ==="
