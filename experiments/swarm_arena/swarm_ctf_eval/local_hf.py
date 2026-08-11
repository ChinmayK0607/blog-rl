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
        del oracle_target
        import torch

        input_ids = torch.tensor(
            [self.renderer.render_ids(messages, add_generation_prompt=True)],
            dtype=torch.long,
            device=self.model.device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
