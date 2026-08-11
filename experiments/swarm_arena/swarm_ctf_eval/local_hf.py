from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalHFArenaModel:
    """Lazy, GPU-only Transformers backend for the frozen arena evaluator."""

    model_id: str
    adapter_path: str | None = None
    max_new_tokens: int = 224

    def __post_init__(self) -> None:
        import torch
        from renderers import Qwen3Renderer, Qwen3RendererConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = self.adapter_path or self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.renderer = Qwen3Renderer(self.tokenizer, Qwen3RendererConfig(enable_thinking=False))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
        return self.respond_many([messages], [oracle_target])[0]

    def respond_many(
        self,
        prompts: list[list[dict[str, str]]],
        oracle_targets: list[str],
    ) -> list[str]:
        del oracle_targets
        import torch

        rendered = [self.renderer.render_ids(messages, add_generation_prompt=True) for messages in prompts]
        encoded = self.tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in rendered],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip() for row in output
        ]
