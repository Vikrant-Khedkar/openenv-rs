"""Wire-compat proof: the real Python OpenEnv client against a Rust echo server.

Usage: python python_client_vs_rust.py http://localhost:8000
"""

import asyncio
import sys

from openenv.core.generic_client import GenericEnvClient


async def main(base_url: str) -> None:
    async with GenericEnvClient(base_url=base_url) as env:
        r = await env.reset()
        assert r.done is False, r
        assert "response" in r.observation, r

        s = await env.step({"message": "interop"})
        assert s.observation["response"] == "interop", s
        assert s.reward == 1.0, s
        assert s.done is False, s

        state = await env.state()
        assert state["step_count"] == 1, state

        rpc = await env.call_mcp("tools/list") if hasattr(env, "call_mcp") else None
        if rpc is not None:
            names = [t["name"] for t in rpc["tools"]]
            assert "echo_message" in names, rpc

    print("PYTHON CLIENT -> RUST SERVER: OK")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
