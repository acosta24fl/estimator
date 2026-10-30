"""Claude engine: structured outputs into Pydantic models, with prompt caching.

Every call puts the skill prompt (and, from stage 2 on, the Story Bible) in
cached system blocks, so repeat calls in a session read them from cache.
If Claude's answer fails a guardrail validator, the validation error is fed
back and it gets another try.
"""
from __future__ import annotations

from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5-5"
T = TypeVar("T", bound=BaseModel)


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, model: str = MODEL, effort: str = "high", max_repairs: int = 2):
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_repairs = max_repairs
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def structured(self, skill_prompt: str, messages: list[dict], schema: type[T], context: str | None = None,
                   check: Callable[[T], list[str]] | None = None) -> T:
        """`check` runs cross-stage guardrails the schema alone can't see; any errors trigger a repair turn."""
        system = [{"type": "text", "text": skill_prompt}]
        if context:
            system.append({"type": "text", "text": context})
        system[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        convo = list(messages)
        for attempt in range(self.max_repairs + 1):
            try:
                resp = self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=16000,
                    system=system,
                    messages=convo,
                    output_format=schema,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except ValidationError as e:  # a guardrail validator rejected the answer
                if attempt == self.max_repairs:
                    raise EngineError(f"Claude's output kept failing validation:\n{e}") from e
                convo = _with_feedback(messages, str(e))
                continue
            except TypeError as e:
                if "authentication" not in str(e).lower():
                    raise
                raise EngineError("No Anthropic credentials found. Set ANTHROPIC_API_KEY (or run `ant auth login`), or use --from-file.") from e
            except anthropic.AuthenticationError as e:
                raise EngineError("No valid Anthropic credentials. Set ANTHROPIC_API_KEY (or run `ant auth login`), or use --from-file.") from e
            except anthropic.RateLimitError as e:
                raise EngineError(f"Rate limited; retry later. ({e.message})") from e
            except anthropic.APIStatusError as e:
                raise EngineError(f"Claude API error {e.status_code}: {e.message}") from e
            except anthropic.APIConnectionError as e:
                raise EngineError("Couldn't reach the Claude API. Check your network.") from e

            u = resp.usage
            self.usage["input"] += u.input_tokens
            self.usage["output"] += u.output_tokens
            self.usage["cache_read"] += u.cache_read_input_tokens or 0
            self.usage["cache_write"] += u.cache_creation_input_tokens or 0
            if resp.stop_reason == "refusal":
                raise EngineError(f"Claude declined this request ({getattr(resp.stop_details, 'category', None)}).")
            if resp.stop_reason == "max_tokens":
                raise EngineError("Answer was cut off at max_tokens; try a smaller batch.")
            if resp.parsed_output is None:
                raise EngineError("Claude returned no structured output.")
            problems = check(resp.parsed_output) if check else []
            if not problems:
                return resp.parsed_output
            if attempt == self.max_repairs:
                raise EngineError("Claude's output kept failing guardrails:\n- " + "\n- ".join(problems))
            convo = _with_feedback(messages, "\n".join(problems))
        raise EngineError("unreachable")


def _with_feedback(messages: list[dict], problems: str) -> list[dict]:
    """Append guardrail feedback to the last user turn (keeps strict user/assistant alternation)."""
    *head, last = messages
    note = f"\n\nA previous answer to this was rejected by the pipeline's guardrails. Avoid these problems:\n{problems}"
    return head + [{"role": "user", "content": last["content"] + note}]
