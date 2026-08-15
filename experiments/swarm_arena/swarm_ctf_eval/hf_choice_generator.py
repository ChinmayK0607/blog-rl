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
        use_kv_cache: bool = True,
        prime_model_config: Any | None = None,
        prime_actor_state_dir: Path | None = None,
        prime_matmul_precision: str = "high",
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
        self.use_kv_cache = use_kv_cache
        self._prime_backend = prime_model_config is not None
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
        self._prime_slots: dict[str, int] = {}
        self._prime_manager: Any | None = None
        if self._prime_backend:
            if use_kv_cache:
                raise ValueError("Prime actor requires full-prefix generation")
            if prime_actor_state_dir is None:
                raise ValueError("Prime actor requires an isolated actor-state directory")
            self.model = self._setup_prime_model(
                prime_model_config,
                prime_actor_state_dir,
                adapter_names,
                model_path,
                prime_matmul_precision,
            )
        else:
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
                self.model.load_adapter(
                    initial_adapter, adapter_name=name, is_trainable=False
                )
        self.model.eval()
        self._adapter_paths = {
            name: str(initial_adapter.resolve()) for name in adapter_names
        }
        self._lock = asyncio.Lock()
        self._group_requests: dict[str, asyncio.Task[ChoiceCompletion]] | None = None

    def _setup_prime_model(
        self,
        model_config: Any,
        state_dir: Path,
        adapter_names: tuple[str, ...],
        model_path: str,
        matmul_precision: str,
    ) -> Any:
        import tomli_w

        from prime_rl.trainer.model import setup_model
        from prime_rl.trainer.parallel_dims import get_parallel_dims
        from prime_rl.trainer.runs import setup_multi_run_manager
        from prime_rl.trainer.utils import setup_torch_distributed

        state_dir.mkdir(parents=True, exist_ok=False)
        for name in adapter_names:
            control = state_dir / f"run_{name}" / "control"
            control.mkdir(parents=True)
            with (control / "orch.toml").open("wb") as handle:
                tomli_w.dump(
                    {
                        "batch_size": 1,
                        "group_size": 1,
                        "max_steps": 1,
                        "model": {
                            "name": model_path,
                            "lora": {"name": name, "rank": 16, "alpha": 32},
                        },
                        "optim": {"lr": 0.000005},
                        "renderer": {"name": "qwen3", "enable_thinking": False},
                        "train": {"env": [{"id": "reverse-text"}]},
                    },
                    handle,
                )
        setup_torch_distributed()
        torch.set_float32_matmul_precision(matmul_precision)
        manager = setup_multi_run_manager(
            state_dir,
            len(adapter_names),
            self.device,
            model_config.lora,
        )
        model = setup_model(model_config, get_parallel_dims(model_config))
        manager.discover_runs()
        manager.synchronize_state()
        expected = {f"run_{name}" for name in adapter_names}
        if set(manager.id_2_idx) != expected:
            raise RuntimeError("Prime actor did not discover every isolated adapter slot")
        self._prime_manager = manager
        self._prime_slots = {
            name: manager.id_2_idx[f"run_{name}"] for name in adapter_names
        }
        return model

    async def __aenter__(self) -> HFChoiceGenerator:
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._prime_backend:
            import torch.distributed as dist

            dist.destroy_process_group()
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
            if self._prime_backend:
                assert self._prime_manager is not None
                digest = hashlib.sha256(
                    (path / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
                self._prime_manager.load_adapter(
                    self._prime_slots[name], path, digest
                )
                self._adapter_paths[name] = expected
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
        if self._prime_backend:
            from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

            from prime_rl.trainer.models.layers.lora import set_lora_num_tokens

            counts = torch.zeros(
                len(self._prime_slots), dtype=torch.int32, device=self.device
            )
            counts[self._active_prime_slot] = input_ids.shape[1]
            set_lora_num_tokens(counts)
            position_ids = torch.arange(
                input_ids.shape[1], device=self.device
            ).unsqueeze(0)
            embedding_weight = self.model.model.embed_tokens.weight
            if not isinstance(embedding_weight, DTensor):
                raise RuntimeError("Prime actor expected an FSDP DTensor embedding")
            mesh = embedding_weight.device_mesh
            placements = [Replicate() for _ in range(mesh.ndim)]
            distributed_input_ids = distribute_tensor(
                input_ids, mesh, placements
            )
            distributed_position_ids = distribute_tensor(
                position_ids, mesh, placements
            )
            with torch.inference_mode():
                output = self.model.model(
                    input_ids=distributed_input_ids,
                    position_ids=distributed_position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state[:, -1, :]
                if isinstance(hidden, DTensor):
                    hidden = hidden.to_local()
                weight = self.model.lm_head.weight
                if isinstance(weight, DTensor):
                    weight = weight.to_local()
                hidden = hidden.to(torch.bfloat16)
                weight = weight.to(torch.bfloat16)
                with torch.autocast("cuda", enabled=False):
                    logits = torch.mm(
                        hidden, weight.t(), out_dtype=torch.float32
                    )[0]
            return logits, None
        backbone = self.model.get_base_model()
        with torch.inference_mode():
            output = backbone.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=self.use_kv_cache,
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
            if self._prime_backend:
                self._active_prime_slot = self._prime_slots[endpoint.model_name]
            else:
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
                if not self.use_kv_cache:
                    token_input = torch.cat((token_input, next_input), dim=1)
                    next_input = token_input
                    cache = None
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
                "actor": (
                    "prime-choice-v1" if self._prime_backend else "hf-choice-v1"
                ),
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
