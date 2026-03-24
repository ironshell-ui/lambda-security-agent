"""Agent — Handles messages using the security pipeline + LLM."""

import asyncio
import os
import re
import traceback

import httpx
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart, TaskState
from a2a.utils import new_agent_text_message

from security_agent import (
    decode_and_sanitize,
    classify_threats,
    check_entropy,
    check_policy,
    validate_output,
    validate_tool_response,
    make_decision,
    Verdict,
    SYSTEM_PROMPT,
)

LLM_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8317/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("GOOGLE_API_KEY", "fake"))
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
MAX_RETRIES = 2


class Agent:
    def __init__(self):
        self.history: list[dict] = []

    async def call_llm(self, messages: list[dict]) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        f"{LLM_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                                 "Content-Type": "application/json"},
                        json={"model": LLM_MODEL, "messages": messages,
                              "max_tokens": 2000, "temperature": 0.0},
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"].get("content", "")
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        return f"I apologize, I'm having technical difficulties."
                    await asyncio.sleep(1)
        return "I'm unable to process your request at this time."

    def _extract_text(self, message: Message) -> str:
        text = ""
        if message.parts:
            for part in message.parts:
                root = getattr(part, 'root', part)
                if hasattr(root, 'text') and root.text:
                    text += root.text + " "
        return text.strip()

    async def run(self, message: Message, updater: TaskUpdater,
                  context_id: str = "", task_id: str = "") -> None:
        try:
            user_text = self._extract_text(message)

            # ── Security Pipeline: Input ──
            cleaned, decode_result = decode_and_sanitize(user_text)
            classify_result = classify_threats(cleaned)
            entropy_result = check_entropy(cleaned)
            policy_input = check_policy(cleaned)

            benign_signals = [
                bool(re.search(r"(help|please|thank|how|can\s+you|what|schedule|check)", cleaned, re.I)),
                len(cleaned) < 300,
                classify_result.verdict == Verdict.SAFE,
            ]
            is_benign = sum(benign_signals) >= 2

            input_checks = [decode_result, classify_result, entropy_result, policy_input]
            input_decision = make_decision(input_checks, is_benign_likely=is_benign)

            if input_decision["action"] == "block":
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=
                        "I'm unable to process that request as it appears to conflict with "
                        "our security policies. How can I help you with a legitimate request?"))],
                    name="response")
                return

            # ── LLM Response ──
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in self.history[-10:]:
                messages.append(h)
            messages.append({"role": "user", "content": cleaned})

            llm_response = await self.call_llm(messages)

            # ── Security Pipeline: Output ──
            output_pii = validate_output(llm_response)
            output_policy = check_policy(llm_response)
            output_decision = make_decision([output_pii, output_policy])

            if output_decision["action"] == "block":
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content":
                    "Your response contained sensitive data. Provide a safe version."})
                llm_response = await self.call_llm(messages)

                if validate_output(llm_response).verdict == Verdict.BLOCK:
                    llm_response = ("I apologize, but I'm unable to provide that information "
                                    "due to our privacy and security policies.")

            self.history.append({"role": "user", "content": cleaned})
            self.history.append({"role": "assistant", "content": llm_response})

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=llm_response))],
                name="response")

        except Exception as e:
            print(f"[Agent] Error: {traceback.format_exc()}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="I encountered an error. Please try again."))],
                name="error")
