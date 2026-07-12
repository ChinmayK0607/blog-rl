"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. For multimodal rollouts the branch also carries
the images it introduced (`branch.multi_modal_data`), rebuilt here into the flat `mm_kwargs` /
`mm_token_type_ids` the trainer forwards.
"""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.trace import Branch

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor, RoutedExperts
from prime_rl.utils.logger import get_logger


def _to_numpy(val) -> np.ndarray:
    """A renderer mm item value (torch tensor or numpy array) -> a contiguous numpy array."""
    if hasattr(val, "detach"):  # torch tensor
        val = val.detach().cpu().numpy()
    return np.ascontiguousarray(val)


def _encode_mm_kwargs(mm_items: dict[str, list[dict]]) -> dict[str, EncodedTensor] | None:
    """Concatenate the branch's per-image renderer items into the flat `mm_kwargs` the trainer
    forwards — one `EncodedTensor` per kwarg key (e.g. `pixel_values`, `image_grid_thw`), images
    cat'd along dim 0 in branch token order. Model-agnostic: the keys are whatever the processor
    emits. Returns None when there are no items."""
    bins: dict[str, list[np.ndarray]] = {}
    for items in mm_items.values():  # per modality
        for item in items:  # per image
            for key, val in item.items():
                bins.setdefault(key, []).append(_to_numpy(val))
    encoded: dict[str, EncodedTensor] = {}
    for key, arrs in bins.items():
        arr = np.concatenate(arrs, axis=0)
        encoded[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return encoded or None


def _encode_routed_experts(arr: np.ndarray | None, num_tokens: int) -> RoutedExperts | None:
    """The branch's router-replay array (`[tokens, layers, top_k]`) -> the transport
    `RoutedExperts` the trainer replays. Defensively realigns the token axis to `num_tokens`
    (the trainer asserts `routed_experts.shape[0] == len(token_ids)`): truncate if longer,
    zero-pad the tail if shorter. `Branch.routed_experts` already guarantees alignment, so this
    is a backstop."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.shape[0] > num_tokens:
        arr = arr[:num_tokens]
    elif arr.shape[0] < num_tokens:
        pad = np.zeros((num_tokens - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return RoutedExperts(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def trace_to_samples(
    trace: vf.Trace,
    *,
    env_name: str = "",
    mm_token_type_ids_mapping: dict[int, int] | None = None,
) -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. Errored rollouts are dropped upstream (`TrainSink.process_rollout`), so no error
    handling happens here. A branch carrying images also gets `mm_kwargs` (the concatenated
    pixel tensors) and `mm_token_type_ids` (the renderer's `mm_token_type_id_map` applied to
    the branch tokens). Branches with no sampled tokens (e.g. an openai client carrying none)
    yield nothing.
    """
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        mask = branch.sampled_mask
        if not any(mask):
            continue
        token_ids = branch.token_ids
        mm_kwargs: dict[str, EncodedTensor] | None = None
        mm_token_type_ids: list[int] | None = None
        mmd = branch.multi_modal_data
        if mmd is not None:
            mm_kwargs = _encode_mm_kwargs(mmd.mm_items)
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(t, 0) for t in token_ids]
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=mask,
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                env_name=env_name,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples


def _compaction_node_groups(nodes: list, token_budget: int) -> list[list[int]]:
    """Group sampled message nodes without ever cutting a message token span."""
    groups: list[list[int]] = []
    current: list[int] = []
    introduced = 0
    previous_sampled = -1
    for index, node in enumerate(nodes):
        introduced += len(node.token_ids)
        if not node.sampled or not any(node.mask):
            continue
        turn_tokens = sum(len(n.token_ids) for n in nodes[previous_sampled + 1 : index + 1])
        if current and introduced > token_budget:
            groups.append(current)
            current = []
            introduced = turn_tokens
        current.append(index)
        previous_sampled = index
    if current:
        groups.append(current)
    return groups


def trace_to_compacted_samples(
    trace: vf.Trace,
    *,
    token_budget: int,
    env_name: str = "",
    mm_token_type_ids_mapping: dict[int, int] | None = None,
) -> list[TrainingSample]:
    """Compile each branch into causal, message-aligned online segments.

    A segment includes the exact cumulative prefix through its final selected
    assistant node. Earlier sampled actions remain causal context but are masked
    from loss; only assistant nodes assigned to this segment are trainable.
    """
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        for sampled_nodes in _compaction_node_groups(branch.nodes, token_budget):
            selected = set(sampled_nodes)
            prefix_nodes = branch.nodes[: sampled_nodes[-1] + 1]
            token_ids: list[int] = []
            mask: list[bool] = []
            logprobs: list[float] = []
            routed_parts: list[np.ndarray] = []
            routed_template = next((n.routed_experts for n in prefix_nodes if n.routed_experts is not None), None)
            for index, node in enumerate(prefix_nodes):
                token_ids.extend(node.token_ids)
                node_mask = node.mask if index in selected else [False] * len(node.mask)
                mask.extend(node_mask)
                source_logprobs = Branch(index=0, nodes=[node]).logprobs
                logprobs.extend(source_logprobs if index in selected else [0.0] * len(node.token_ids))
                if routed_template is not None:
                    routed_parts.append(
                        node.routed_experts
                        if node.routed_experts is not None
                        else np.zeros(
                            (len(node.token_ids), *routed_template.shape[1:]), dtype=routed_template.dtype
                        )
                    )
            if not any(mask):
                continue
            prefix = Branch(index=branch.index, nodes=prefix_nodes)
            mm_kwargs = _encode_mm_kwargs(prefix.multi_modal_data.mm_items) if prefix.multi_modal_data else None
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(t, 0) for t in token_ids] if mm_kwargs is not None else None
            routed = np.concatenate(routed_parts, axis=0) if routed_parts else None
            samples.append(
                TrainingSample(
                    token_ids=token_ids,
                    mask=mask,
                    logprobs=logprobs,
                    temperatures=[],
                    env_name=env_name,
                    mm_kwargs=mm_kwargs,
                    mm_token_type_ids=mm_token_type_ids,
                    routed_experts=_encode_routed_experts(routed, len(token_ids)),
                )
            )
    if not samples:
        get_logger().warning(f"No trainable compacted samples (error={trace.has_error}, turns={trace.num_turns}).")
    return samples
