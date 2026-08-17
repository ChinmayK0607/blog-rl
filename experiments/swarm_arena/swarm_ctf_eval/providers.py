from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

from .schema import Scenario


@dataclass
class Generation:
    text: str
    latency_seconds: float
    usage: dict[str, Any]


class Provider(Protocol):
    def generate(self, scenario: Scenario, messages: list[dict[str, str]]) -> Generation: ...


class OracleProvider:
    """A perfect deterministic provider used to validate the harness."""

    def generate(self, scenario: Scenario, messages: list[dict[str, str]]) -> Generation:
        del messages
        message = ""
        if scenario.required_message_facts:
            message = "; ".join(" ".join(fact) for fact in scenario.required_message_facts)
        answer = scenario.acceptable_actions[0]
        index = scenario.legal_actions.index(answer)
        value = {"message": message, "action_id": f"A{index}"}
        return Generation(json.dumps(value), 0.0, {"completion_tokens": 0})


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "local",
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: float = 180.0,
        enable_thinking: bool = False,
        seed: int | None = None,
        response_format_factory: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.response_format_factory = response_format_factory

    def generate(self, scenario: Scenario, messages: list[dict[str, str]]) -> Generation:
        del scenario
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.response_format_factory is not None:
            payload["response_format"] = self.response_format_factory(messages)
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"inference HTTP {exc.code}: {detail}") from exc
        elapsed = time.perf_counter() - started
        text = body["choices"][0]["message"].get("content") or ""
        return Generation(text=text, latency_seconds=elapsed, usage=body.get("usage", {}))
