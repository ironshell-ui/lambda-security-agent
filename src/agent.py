"""Agent — Handles messages using the security pipeline + LLM."""

import asyncio
import json
import os
import traceback

import httpx
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart, DataPart, Role
from a2a.utils import get_message_text, new_agent_text_message

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
        self.turn_count = 0

    async def call_llm(self, messages: list[dict], max_tokens: int = 2000) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        f"{LLM_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                                 "Content-Type": "application/json"},
                        json={"model": LLM_MODEL, "messages": messages,
                              "max_tokens": max_tokens, "temperature": 0.0},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"].get("content", "")
                    if content:
                        return content
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        return f"I apologize, I'm having technical difficulties. Error: {str(e)[:100]}"
                    await asyncio.sleep(1)
        return "I'm unable to process your request at this time."

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        try:
            user_text = get_message_text(message) or ""
            self.turn_count += 1

            # ── Security Pipeline: Input ──
            cleaned, decode_result = decode_and_sanitize(user_text)
            classify_result = classify_threats(cleaned)
            entropy_result = check_entropy(cleaned)
            policy_input = check_policy(cleaned)

            # Benign detection for RESTRAINT mode
            import re
            benign_signals = [
                bool(re.search(r"(help|please|thank|how|can\s+you|what|schedule|check|report)", cleaned, re.I)),
                len(cleaned) < 300,
                classify_result.verdict == Verdict.SAFE,
            ]
            is_benign = sum(benign_signals) >= 2

            input_checks = [decode_result, classify_result, entropy_result, policy_input]
            input_decision = make_decision(input_checks, is_benign_likely=is_benign)

            if input_decision["action"] == "block":
                response_text = (
                    "I'm unable to process that request as it appears to conflict with "
                    "our security policies. How can I help you with a legitimate request?"
                )
                await updater.update_status(
                    new_agent_text_message(response_text,
                                          context_id=updater._context_id,
                                          task_id=updater._task_id),
                )
                return

            # ── Check for tool responses in message parts ──
            tool_responses = {}
            if message.parts:
                for part in message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'data'):
                        data = part.root.data if hasattr(part.root, 'data') else None
                        if isinstance(data, dict):
                            if "tool_results" in data:
                                for tr in data["tool_results"]:
                                    tool_responses[tr.get("tool_name", "unknown")] = tr.get("content", "")
                            elif "tool_result" in data:
                                tr = data["tool_result"]
                                tool_responses[tr.get("tool_name", "unknown")] = tr.get("content", "")

            # ── Pre-execution tool validation (AVER) ──
            for tool_name, tool_resp in tool_responses.items():
                tool_check = validate_tool_response(tool_name, str(tool_resp))
                if tool_check.verdict == Verdict.BLOCK:
                    response_text = (
                        f"I detected an anomaly in the {tool_name} response: "
                        f"{tool_check.reason}. "
                        f"Diagnosis: The tool response contains indicators of an error — "
                        f"specifically {', '.join(tool_check.evidence[:2])}. "
                        f"Recovery: Let me try a different approach to get accurate results."
                    )
                    await updater.update_status(
                        new_agent_text_message(response_text,
                                              context_id=updater._context_id,
                                              task_id=updater._task_id),
                    )
                    return

            # ── LLM Response ──
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in self.history[-10:]:  # Keep last 10 turns
                messages.append(h)
            messages.append({"role": "user", "content": cleaned})

            llm_response = await self.call_llm(messages)

            # ── Security Pipeline: Output ──
            output_pii = validate_output(llm_response)
            output_policy = check_policy(llm_response)
            output_decision = make_decision([output_pii, output_policy])

            if output_decision["action"] == "block":
                # Regenerate safe response
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content":
                    "Your previous response contained sensitive information or a policy violation. "
                    "Please provide a safe response without PII, credentials, or policy violations."})
                llm_response = await self.call_llm(messages)

                # Final check
                recheck = validate_output(llm_response)
                if recheck.verdict == Verdict.BLOCK:
                    llm_response = (
                        "I apologize, but I'm unable to provide that specific information "
                        "due to our privacy and security policies. How else can I assist you?"
                    )

            # ── Store history ──
            self.history.append({"role": "user", "content": cleaned})
            self.history.append({"role": "assistant", "content": llm_response})

            # ── Send response ──
            await updater.update_status(
                new_agent_text_message(llm_response,
                                      context_id=updater._context_id,
                                      task_id=updater._task_id),
            )

        except Exception as e:
            print(f"[Agent] Error: {traceback.format_exc()}")
            await updater.update_status(
                new_agent_text_message(
                    f"I encountered an error processing your request. Please try again.",
                    context_id=updater._context_id,
                    task_id=updater._task_id),
            )
