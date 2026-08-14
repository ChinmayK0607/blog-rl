from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import torch
import xgrammar as xgr
from renderers import Qwen3Renderer, Qwen3RendererConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm.v1.structured_output.utils import choice_as_grammar

from .live_rl_rollout import ChoiceCompletion, PolicyEndpoint
from .safety_supervisor import canonical_sha256
from .structured_protocol import protocol_choices


class HFChoiceGenerator:
    """Truthful constrained actor using the same Transformers forward family as training."""

    def __init__(
        self,
        model_path: str,
        initial_adapter: Path,
        *,
        adapter_names: tuple[str, ...],
        attention: str = "flash_attention_2",
        device: str = "cuda",
        max_tokens: int = 128,
    ) -> None:
        from transformers.utils import import_utils

        import_utils._torchvision_available = False
        from peft import PeftModel

        if not adapter_names or len(set(adapter_names)) != len(adapter_names):
            raise ValueError("HF actor adapter names must be nonempty and unique")
        if max_tokens < 1:
            raise ValueError("HF actor max_tokens must be positive")
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.attention = attention
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.renderer = Qwen3Renderer(
            self.tokenizer,
            Qwen3RendererConfig(enable_thinking=False),
        )
        config = AutoConfig.from_pretrained(model_path)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=int(config.vocab_size),
        )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )
        self.vocab_size = int(config.vocab_size)
        self._token_ids = torch.arange(self.vocab_size, dtype=torch.int64)
        self._word_indices = self._token_ids // 32
        self._bit_indices = self._token_ids % 32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attention,
        ).to(self.device)
        first, *remaining = adapter_names
        self.model = PeftModel.from_pretrained(
            base_model,
            initial_adapter,
            adapter_name=first,
            is_trainable=False,
        )
        for name in remaining:
            self.model.load_adapter(initial_adapter, adapter_name=name, is_trainable=False)
        self.model.eval()
        self._adapter_paths = {
            name: str(initial_adapter.resolve()) for name in adapter_names
        }
        self._lock = asyncio.Lock()
        self._group_requests: dict[str, asyncio.Task[ChoiceCompletion]] | None = None

    async def __aenter__(self) -> HFChoiceGenerator:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    @asynccontextmanager
    async def coalesced_request_group(self) -> AsyncIterator[None]:
        if self._group_requests is not None:
            raise RuntimeError("inference request groups cannot be nested")
        self._group_requests = {}
        try:
            yield
        finally:
            self._group_requests = None

    async def replace_adapter(self, name: str, path: Path) -> None:
        expected = str(path.resolve())
        async with self._lock:
            if self._adapter_paths.get(name) == expected:
                return
            if name in self._adapter_paths:
                active = next(
                    candidate
                    for candidate in self._adapter_paths
                    if candidate != name
                )
                self.model.set_adapter(active)
                self.model.delete_adapter(name)
            self.model.load_adapter(path, adapter_name=name, is_trainable=False)
            self._adapter_paths[name] = expected

    def _allowed_token_ids(self, matcher: Any, bitmask: torch.Tensor) -> list[int]:
        xgr.reset_token_bitmask(bitmask)
        matcher.fill_next_token_bitmask(bitmask)
        words = bitmask[0].to(torch.int64)
        accepted = (
            (words[self._word_indices] >> self._bit_indices) & 1
        ).to(torch.bool)
        return self._token_ids[accepted].tolist()

    def _last_token_logits(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Any | None,
    ) -> tuple[torch.Tensor, Any]:
        backbone = self.model.get_base_model()
        with torch.inference_mode():
            output = backbone.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            hidden = output.last_hidden_state[:, -1, :].to(dtype=torch.bfloat16)
            weight = backbone.lm_head.weight.to(dtype=torch.bfloat16)
            with torch.autocast("cuda", enabled=False):
                logits = torch.mm(hidden, weight.t(), out_dtype=torch.float32)[0]
        return logits, output.past_key_values

    async def _complete(
        self,
        endpoint: PolicyEndpoint,
        choices: tuple[str, ...],
        prompt_ids: tuple[int, ...],
        *,
        seed: int,
        request_sha256: str,
    ) -> ChoiceCompletion:
        async with self._lock:
            if endpoint.model_name not in self._adapter_paths:
                raise ValueError(f"HF actor has no adapter named {endpoint.model_name}")
            self.model.set_adapter(endpoint.model_name)
            compiled = self.compiler.compile_grammar(choice_as_grammar(list(choices)))
            matcher = xgr.GrammarMatcher(compiled)
            bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
            random = torch.Generator(device=self.device).manual_seed(seed)
            token_input = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.device
            )
            logits, cache = self._last_token_logits(
                token_input,
                past_key_values=None,
            )
            completion_ids: list[int] = []
            completion_logprobs: list[float] = []
            allowed_rows: list[tuple[int, ...]] = []
            distribution_rows: list[tuple[tuple[int, float], ...]] = []
            for _ in range(self.max_tokens):
                allowed = self._allowed_token_ids(matcher, bitmask)
                if not allowed:
                    raise RuntimeError("HF actor choice grammar produced an empty mask")
                legal_ids = torch.tensor(
                    allowed,
                    dtype=torch.long,
                    device=self.device,
                )
                legal_logprobs = torch.log_softmax(logits[legal_ids], dim=0)
                sampled_index = int(
                    torch.multinomial(
                        legal_logprobs.exp(),
                        1,
                        generator=random,
                    ).item()
                )
                token_id = allowed[sampled_index]
                completion_ids.append(token_id)
                completion_logprobs.append(float(legal_logprobs[sampled_index]))
                allowed_rows.append(tuple(allowed))
                distribution_rows.append(
                    tuple(
                        (candidate, float(value))
                        for candidate, value in zip(
                            allowed,
                            legal_logprobs.tolist(),
                            strict=True,
                        )
                    )
                )
                if not matcher.accept_token(token_id):
                    raise RuntimeError("HF actor sampled a grammar-rejected token")
                if matcher.is_terminated():
                    break
                next_input = torch.tensor(
                    [[token_id]], dtype=torch.long, device=self.device
                )
                logits, cache = self._last_token_logits(
                    next_input,
                    past_key_values=cache,
                )
            else:
                raise RuntimeError("HF actor did not terminate within max_tokens")
            decoded = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
            if decoded not in choices:
                raise RuntimeError("HF actor completion does not decode to a legal choice")
            return ChoiceCompletion(
                prompt_ids=prompt_ids,
                completion_ids=tuple(completion_ids),
                logprobs=tuple(completion_logprobs),
                allowed_token_ids=tuple(allowed_rows),
                text=decoded,
                request_sha256=request_sha256,
                transport_attempts=1,
                serving_allowed_logprobs=tuple(distribution_rows),
            )

    async def generate(
        self,
        endpoint: PolicyEndpoint,
        messages: list[dict[str, str]],
        *,
        sampling_key: str,
    ) -> ChoiceCompletion:
        endpoint.validate(require_base_urls=False)
        choices = protocol_choices(messages)
        prompt_ids = tuple(
            self.renderer.render_ids(messages, add_generation_prompt=True)
        )
        digest = hashlib.sha256(sampling_key.encode()).digest()
        seed = int.from_bytes(digest[4:8], "big")
        request_sha256 = canonical_sha256(
            {
                "actor": "hf-choice-v1",
                "attention": self.attention,
                "max_tokens": self.max_tokens,
                "policy_id": endpoint.policy_id,
                "policy_revision": endpoint.revision,
                "sampling_key": sampling_key,
                "model_name": endpoint.model_name,
                "prompt_ids": prompt_ids,
                "choices": choices,
                "seed": seed,
            }
        )
        request = self._complete(
            endpoint,
            choices,
            prompt_ids,
            seed=seed,
            request_sha256=request_sha256,
        )
        if self._group_requests is None:
            return await request
        task = self._group_requests.get(request_sha256)
        if task is None:
            task = asyncio.create_task(request)
            self._group_requests[request_sha256] = task
        else:
            request.close()
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            if self._group_requests.get(request_sha256) is task:
                del self._group_requests[request_sha256]
            raise
