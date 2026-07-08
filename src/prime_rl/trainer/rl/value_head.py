import torch
from torch import Tensor, nn


class PPOValueHead(nn.Module):
    """Scalar critic head over causal transformer hidden states."""

    def __init__(self, hidden_size: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"expected hidden states shaped [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        return self.proj(hidden_states).squeeze(-1).float()
