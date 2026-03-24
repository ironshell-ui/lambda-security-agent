"""Lambda Security Agent — A2A Server Entry Point."""

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Lambda Security Purple Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Lambda Security Agent",
        description="Defense-in-depth agent with 7-stage security pipeline. "
                    "Handles policy compliance, adversarial defense, and error recovery.",
        url=f"http://localhost:{args.port}",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(id="policy", name="Policy Compliance",
                       description="7 policy surface verification", tags=["security", "policy"]),
            AgentSkill(id="adversarial", name="Adversarial Defense",
                       description="OWASP-aligned threat detection with encoding decode", tags=["security", "adversarial"]),
            AgentSkill(id="recovery", name="Error Recovery",
                       description="Tool response validation with pre-execution detection", tags=["security", "recovery"]),
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    task_store = InMemoryTaskStore()
    executor = Executor()
    handler = DefaultRequestHandler(agent_card=agent_card, task_store=task_store, executor=executor)
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    print(f"Lambda Security Agent on {args.host}:{args.port}")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
