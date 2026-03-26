"""Lambda Security Agent — A2A Server Entry Point."""

import argparse
import json
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


class PythonA2ACompatMiddleware:
    """ASGI middleware that converts python-a2a format to a2a-sdk JSON-RPC format.

    python-a2a sends: {"content": {"text": "...", "type": "text"}, "role": "user", "message_id": "..."}
    a2a-sdk expects: {"jsonrpc": "2.0", "method": "message/send", "id": "...", "params": {"message": {...}}}
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only intercept POST to root (where A2A messages go)
        if scope.get("method") != "POST" or path not in ("/", ""):
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_parts = []
        async def buffered_receive():
            msg = await receive()
            if msg["type"] == "http.request":
                body_parts.append(msg.get("body", b""))
            return msg

        # Read the full body
        msg = await buffered_receive()
        raw_body = b"".join(body_parts)

        try:
            data = json.loads(raw_body)
        except Exception:
            # Not JSON, pass through
            async def replay_receive():
                return {"type": "http.request", "body": raw_body, "more_body": False}
            await self.app(scope, replay_receive, send)
            return

        # Check if this is python-a2a format (has "content" + "role" but no "jsonrpc")
        if "content" in data and "role" in data and "jsonrpc" not in data:
            # Convert to JSON-RPC format
            import uuid
            text = ""
            content = data.get("content", {})
            if isinstance(content, dict):
                text = content.get("text", "")
            elif isinstance(content, str):
                text = content

            msg_id = data.get("message_id", str(uuid.uuid4()))

            jsonrpc_body = json.dumps({
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": msg_id,
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": text}],
                        "messageId": msg_id,
                    }
                }
            }).encode()

            async def replay_receive():
                return {"type": "http.request", "body": jsonrpc_body, "more_body": False}

            # Also need to convert the response back to python-a2a format
            original_headers = None

            async def convert_send(message):
                nonlocal original_headers
                if message["type"] == "http.response.start":
                    original_headers = message
                    return
                if message["type"] == "http.response.body":
                    resp_body = message.get("body", b"")
                    try:
                        resp_data = json.loads(resp_body)
                        # Extract text from JSON-RPC response
                        result = resp_data.get("result", {})
                        text_out = ""

                        # Try artifacts
                        for art in result.get("artifacts", []):
                            for part in art.get("parts", []):
                                if isinstance(part, dict):
                                    if part.get("kind") == "text":
                                        text_out += part.get("text", "")
                                    elif "text" in part:
                                        text_out += part["text"]

                        # Try status message
                        if not text_out:
                            status_msg = result.get("status", {}).get("message", {})
                            if status_msg:
                                for part in status_msg.get("parts", []):
                                    if isinstance(part, dict) and "text" in part:
                                        text_out += part["text"]

                        if text_out:
                            # Return python-a2a format
                            pa2a_resp = json.dumps({
                                "content": {"text": text_out, "type": "text"},
                                "role": "agent",
                                "message_id": msg_id,
                            }).encode()
                            resp_body = pa2a_resp
                    except Exception:
                        pass

                    if original_headers:
                        new_headers = [(k, v) for k, v in original_headers.get("headers", [])
                                       if k != b"content-length"]
                        new_headers.append((b"content-length", str(len(resp_body)).encode()))
                        original_headers["headers"] = new_headers
                        await send(original_headers)
                        original_headers = None
                    await send({**message, "body": resp_body})
                else:
                    await send(message)

            await self.app(scope, replay_receive, convert_send)
            return

        # Regular JSON-RPC request, pass through with response patching
        async def replay_receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

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
                    new_headers = [(k, v) for k, v in original_headers.get("headers", [])
                                   if k != b"content-length"]
                    new_headers.append((b"content-length", str(len(body)).encode()))
                    original_headers["headers"] = new_headers
                    await send(original_headers)
                    original_headers = None
                await send({**message, "body": body})
            else:
                await send(message)

        await self.app(scope, replay_receive, patched_send)

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
    parser.add_argument("--card-url", default=None, help="Public URL for agent card")
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Lambda Security Agent",
        description="Defense-in-depth agent with 7-stage security pipeline. "
                    "Policy compliance, adversarial defense, error recovery.",
        url=args.card_url or f"http://localhost:{args.port}",
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
    app = PythonA2ACompatMiddleware(app)

    print(f"Lambda Security Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
