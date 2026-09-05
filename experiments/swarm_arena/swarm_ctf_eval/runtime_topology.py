from __future__ import annotations

from dataclasses import asdict, dataclass


def _unique_nonnegative(values: tuple[int, ...], *, label: str) -> None:
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} must contain non-negative GPU indices")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} cannot contain duplicate GPU indices")


@dataclass(frozen=True)
class RuntimeTopology:
    trainer_gpu_ids: tuple[int, ...]
    inference_gpu_ids: tuple[int, ...]
    rollout_ports: tuple[int, ...]

    def validate(self, *, visible_gpu_count: int | None = None) -> None:
        _unique_nonnegative(self.trainer_gpu_ids, label="trainer GPU IDs")
        _unique_nonnegative(self.inference_gpu_ids, label="inference GPU IDs")
        overlap = set(self.trainer_gpu_ids) & set(self.inference_gpu_ids)
        if overlap:
            raise ValueError(f"trainer and inference GPU IDs overlap: {sorted(overlap)}")
        if len(self.rollout_ports) != len(self.inference_gpu_ids):
            raise ValueError("each inference GPU requires exactly one rollout port")
        if any(not 1 <= port <= 65535 for port in self.rollout_ports):
            raise ValueError("rollout ports must be between 1 and 65535")
        if len(set(self.rollout_ports)) != len(self.rollout_ports):
            raise ValueError("rollout ports cannot contain duplicates")
        if visible_gpu_count is not None:
            assigned = set(self.trainer_gpu_ids) | set(self.inference_gpu_ids)
            expected = set(range(visible_gpu_count))
            if assigned != expected:
                raise ValueError(
                    "runtime topology must assign every visible GPU exactly once: "
                    f"assigned={sorted(assigned)}, visible={sorted(expected)}"
                )

    @property
    def base_urls(self) -> tuple[str, ...]:
        return tuple(f"http://127.0.0.1:{port}" for port in self.rollout_ports)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "trainer_gpu_ids": list(self.trainer_gpu_ids),
            "inference_gpu_ids": list(self.inference_gpu_ids),
            "rollout_ports": list(self.rollout_ports),
            "base_urls": list(self.base_urls),
        }


def runtime_topology(
    trainer_gpu_ids: list[int],
    inference_gpu_ids: list[int],
    rollout_ports: list[int],
    *,
    visible_gpu_count: int | None = None,
) -> RuntimeTopology:
    topology = RuntimeTopology(
        trainer_gpu_ids=tuple(trainer_gpu_ids),
        inference_gpu_ids=tuple(inference_gpu_ids),
        rollout_ports=tuple(rollout_ports),
    )
    topology.validate(visible_gpu_count=visible_gpu_count)
    return topology
