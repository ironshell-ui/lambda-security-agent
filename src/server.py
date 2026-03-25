"""Lambda Security Agent — A2A Server Entry Point."""

import argparse
import json
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


class PiBenchCompatMiddleware:
    """ASGI middleware for Pi-bench evaluator compatibility."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        original_headers = None

        async def patched_send(message):
            nonlocal original_headers
            if message["type"] == "http.response.start":
                original_headers = message
                return
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                try:
                    data = json.loads(body)
                    if "result" in data and isinstance(data["result"], dict):
                        data["result"] = self._patch(data["result"])
                    body = json.dumps(data).encode()
                except Exception:
                    pass
                if original_headers:
                    headers = dict(original_headers.get("headers", []))
                    new_headers = [(k, v) for k, v in original_headers.get("headers", [])
                                   if k != b"content-length"]
                    new_headers.append((b"content-length", str(len(body)).encode()))
                    original_headers["headers"] = new_headers
                    await send(original_headers)
                    original_headers = None
                await send({**message, "body": body})
            else:
                await send(message)

        await self.app(scope, receive, patched_send)

    def _patch(self, result):
        all_parts = []
        for art in result.get("artifacts", []):
            for part in art.get("parts", []):
                all_parts.append(part)
        status_msg = result.get("status", {}).get("message", {})
        if status_msg:
            for part in status_msg.get("parts", []):
                all_parts.append(part)
        if all_parts:
            converted = []
            for part in all_parts:
                if part.get("kind") == "data" and isinstance(part.get("data"), dict):
                    d = part["data"]
                    if d.get("type") == "tool_call":
                        converted.append({
                            "kind": "tool_call",
                            "name": d.get("name", ""),
                            "arguments": d.get("arguments", {}),
                            "callId": d.get("callId", ""),
                        })
                        continue
                    if d.get("type") == "state_change":
                        converted.append({
                            "kind": "state_change",
                            "entity": d.get("entity", ""),
                            "state": d.get("state", ""),
                        })
                        continue
                converted.append(part)
            result["message"] = {"role": "agent", "parts": converted}
        return result


def main():
    parser = argparse.ArgumentParser(description="Lambda Security Purple Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Lambda Security Agent",
        description="Defense-in-depth agent with 7-stage security pipeline. "
                    "Policy compliance, adversarial defense, error recovery.",
        url=f"http://localhost:{args.port}",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(id="policy", name="Policy Compliance",
                       description="7 policy surface verification with RESTRAINT mode",
                       tags=["security", "policy"]),
            AgentSkill(id="adversarial", name="Adversarial Defense",
                       description="71 OWASP-aligned threat patterns with encoding decode",
                       tags=["security", "adversarial"]),
            AgentSkill(id="recovery", name="Error Recovery",
                       description="Pre-execution tool validation with metacognitive recovery",
                       tags=["security", "recovery"]),
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = server.build()
    app = PiBenchCompatMiddleware(app)

    print(f"Lambda Security Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
