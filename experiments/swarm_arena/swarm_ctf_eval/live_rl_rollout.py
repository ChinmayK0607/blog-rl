from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Literal

import httpx
from renderers import Qwen3Renderer, Qwen3RendererConfig
from transformers import AutoTokenizer

from prime_rl.transport import TrainingSample

from .arena import Action, GameState, state_to_dict
from .arena_protocol import Broadcast, parse_action, parse_broadcast
from .episode import EpisodeConfig
from .episode_protocol import episode_action_prompt, episode_broadcast_prompt
from .multi_policy_contract import AgentPolicy
from .prime_multi_run_router import OwnedAgentSamples
from .prime_rl_bridge import RolloutDecision
from .rl_v3 import ArenaRLEnv
from .safety_supervisor import (
    BranchReplay,
    CreditGroupEvidence,
    ReplayTurn,
    canonical_sha256,
)
from .structured_protocol import (
    STRUCTURED_PROTOCOL_VERSION,
    completion_allowed_token_ids,
    protocol_choices,
)

Phase = Literal["BROADCAST", "ACT"]


@dataclass(frozen=True)
class PolicyEndpoint:
    policy_id: str
    revision: str
    model_name: str
    base_urls: tuple[str, ...]

    def validate(self) -> None:
        if not self.policy_id or not self.revision or not self.model_name:
            raise ValueError("policy endpoint fields cannot be empty")
        if not self.base_urls or any(not value.startswith("http") for value in self.base_urls):
            raise ValueError("policy endpoint requires HTTP inference URLs")


@dataclass(frozen=True)
class ChoiceCompletion:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    allowed_token_ids: tuple[tuple[int, ...], ...]
    text: str


@dataclass(frozen=True)
class BranchRollout:
    replay: BranchReplay
    decisions: tuple[RolloutDecision, ...]
    samples: tuple[tuple[str, TrainingSample], ...]


@dataclass(frozen=True)
class LiveCreditGroup:
    evidence: CreditGroupEvidence
    bindings: tuple[AgentPolicy, ...]
    owned_samples: tuple[OwnedAgentSamples, ...]


def protocol_constraint_sha256(phase: Phase) -> str:
    return canonical_sha256({"phase": phase, "version": STRUCTURED_PROTOCOL_VERSION})


def parity_gate_sha256(config: object) -> str:
    if not hasattr(config, "model_dump"):
        raise TypeError("parity gate config must support model_dump")
    return canonical_sha256(config.model_dump(mode="json"))


class VLLMChoiceGenerator:
    def __init__(self, tokenizer_path: str, *, timeout: float = 180.0) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.renderer = Qwen3Renderer(
            self.tokenizer,
            Qwen3RendererConfig(enable_thinking=False),
        )
        self.timeout = timeout
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def __aenter__(self) -> VLLMChoiceGenerator:
        return self

    async def __aexit__(self, *args: object) -> None:
        await asyncio.gather(*(client.aclose() for client in self._clients.values()))

    def _client(self, base_url: str) -> httpx.AsyncClient:
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=self.timeout,
            )
        return self._clients[base_url]

    async def generate(
        self,
        endpoint: PolicyEndpoint,
        messages: list[dict[str, str]],
        *,
        sampling_key: str,
    ) -> ChoiceCompletion:
        endpoint.validate()
        choices = protocol_choices(messages)
        prompt_ids = self.renderer.render_ids(messages, add_generation_prompt=True)
        digest = hashlib.sha256(sampling_key.encode()).digest()
        base_url = endpoint.base_urls[int.from_bytes(digest[:4], "big") % len(endpoint.base_urls)]
        seed = int.from_bytes(digest[4:8], "big")
        response = await self._client(base_url).post(
            "/inference/v1/generate",
            json={
                "model": endpoint.model_name,
                "token_ids": prompt_ids,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 128,
                    "logprobs": 1,
                    "seed": seed,
                    "structured_outputs": {"choice": list(choices)},
                },
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        completion_ids = list(choice["token_ids"])
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        choice_token_ids = [
            [*self.tokenizer.encode(value, add_special_tokens=False), eos_token_id]
            for value in choices
        ]
        allowed = completion_allowed_token_ids(completion_ids, choice_token_ids)
        return ChoiceCompletion(
            tuple(prompt_ids),
            tuple(completion_ids),
            tuple(item["logprob"] for item in choice["logprobs"]["content"]),
            tuple(tuple(row) for row in allowed),
            self.tokenizer.decode(completion_ids, skip_special_tokens=True),
        )


def _state_sha256(state: GameState) -> str:
    return canonical_sha256(state_to_dict(state))


def _sampling_key(game_id: str, agent_id: str, turn: int, phase: Phase) -> str:
    return f"{game_id}:{agent_id}:{turn}:{phase}"


def _policy_for_agent(
    agent_id: str,
    bindings: dict[str, AgentPolicy],
    policies: dict[str, PolicyEndpoint],
    *,
    replaced_agent: str | None,
    replacement_policy_id: str,
) -> PolicyEndpoint:
    policy_id = replacement_policy_id if agent_id == replaced_agent else bindings[agent_id].policy_id
    try:
        return policies[policy_id]
    except KeyError as error:
        raise ValueError(f"missing endpoint for policy {policy_id}") from error


def _training_sample(completion: ChoiceCompletion) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(completion.prompt_ids),
        prompt_mask=[False] * len(completion.prompt_ids),
        completion_ids=list(completion.completion_ids),
        completion_mask=[True] * len(completion.completion_ids),
        completion_logprobs=list(completion.logprobs),
        completion_temperatures=[1.0] * len(completion.completion_ids),
        env_name="swarm_arena_rl_v3",
        advantage=None,
        reward=None,
        completion_allowed_token_ids=[list(row) for row in completion.allowed_token_ids],
        training_mode="rl",
    )


async def rollout_branch(
    generator: VLLMChoiceGenerator,
    *,
    game_id: str,
    initial_state: GameState,
    config: EpisodeConfig,
    bindings: tuple[AgentPolicy, ...],
    policies: dict[str, PolicyEndpoint],
    replacement_policy_id: str,
    replaced_agent: str | None,
) -> BranchRollout:
    env = ArenaRLEnv(size=len(initial_state.nodes), config=config)
    env.reset_from_state(initial_state)
    binding_by_agent = {row.agent_id: row for row in bindings}
    decisions: list[RolloutDecision] = []
    samples: list[tuple[str, TrainingSample]] = []
    turns: list[ReplayTurn] = []
    trajectory_index = 0
    final = None

    for turn in range(initial_state.turn, initial_state.turn + config.horizon):
        pre_state_sha256 = _state_sha256(env._require_state())
        broadcast_contexts = env.observations()
        broadcast_jobs = []
        broadcast_metadata = []
        for index, agent_id in enumerate(sorted(binding_by_agent)):
            messages, _ = episode_broadcast_prompt(
                env,
                agent_id,
                permutation=int(hashlib.sha256(f"{game_id}:broadcast:{turn}:{index}".encode()).hexdigest()[:8], 16),
            )
            endpoint = _policy_for_agent(
                agent_id,
                binding_by_agent,
                policies,
                replaced_agent=replaced_agent,
                replacement_policy_id=replacement_policy_id,
            )
            key = _sampling_key(game_id, agent_id, turn, "BROADCAST")
            broadcast_jobs.append(generator.generate(endpoint, messages, sampling_key=key))
            broadcast_metadata.append((agent_id, endpoint, key))
        broadcast_completions = await asyncio.gather(*broadcast_jobs)
        broadcasts: dict[str, Broadcast] = {}
        for (agent_id, endpoint, key), completion in zip(
            broadcast_metadata, broadcast_completions, strict=True
        ):
            parsed = parse_broadcast(completion.text, env._require_state(), agent_id)
            if not parsed.valid or not isinstance(parsed.value, Broadcast):
                raise RuntimeError(f"structured broadcast failed for {agent_id}: {parsed.errors}")
            broadcast = parsed.value
            broadcasts[agent_id] = broadcast
            binding = binding_by_agent[agent_id]
            decisions.append(
                RolloutDecision(
                    game_id,
                    "actual" if replaced_agent is None else "replacement",
                    replaced_agent,
                    agent_id,
                    endpoint.policy_id,
                    endpoint.revision,
                    binding.team,
                    turn,
                    "BROADCAST",
                    trajectory_index,
                    completion.prompt_ids,
                    completion.completion_ids,
                    completion.logprobs,
                    protocol_constraint_sha256("BROADCAST"),
                    key,
                    canonical_sha256(broadcast_contexts[agent_id]),
                    canonical_sha256(broadcast.to_dict()),
                )
            )
            if replaced_agent is None and binding.trainable:
                samples.append((agent_id, _training_sample(completion)))
            trajectory_index += 1

        phase = env.broadcast_phase(broadcasts)
        action_contexts = env.action_observations()
        action_jobs = []
        action_metadata = []
        for index, agent_id in enumerate(sorted(binding_by_agent)):
            messages, displayed = episode_action_prompt(
                env,
                agent_id,
                permutation=int(hashlib.sha256(f"{game_id}:action:{turn}:{index}".encode()).hexdigest()[:8], 16),
            )
            endpoint = _policy_for_agent(
                agent_id,
                binding_by_agent,
                policies,
                replaced_agent=replaced_agent,
                replacement_policy_id=replacement_policy_id,
            )
            key = _sampling_key(game_id, agent_id, turn, "ACT")
            action_jobs.append(generator.generate(endpoint, messages, sampling_key=key))
            action_metadata.append((agent_id, endpoint, key, displayed))
        action_completions = await asyncio.gather(*action_jobs)
        actions: dict[str, Action] = {}
        for (agent_id, endpoint, key, displayed), completion in zip(
            action_metadata, action_completions, strict=True
        ):
            parsed = parse_action(completion.text, displayed)
            if not parsed.valid or not isinstance(parsed.value, Action):
                raise RuntimeError(f"structured action failed for {agent_id}: {parsed.errors}")
            action = parsed.value
            actions[agent_id] = action
            binding = binding_by_agent[agent_id]
            decisions.append(
                RolloutDecision(
                    game_id,
                    "actual" if replaced_agent is None else "replacement",
                    replaced_agent,
                    agent_id,
                    endpoint.policy_id,
                    endpoint.revision,
                    binding.team,
                    turn,
                    "ACT",
                    trajectory_index,
                    completion.prompt_ids,
                    completion.completion_ids,
                    completion.logprobs,
                    protocol_constraint_sha256("ACT"),
                    key,
                    canonical_sha256(action_contexts[agent_id]),
                    canonical_sha256(action.to_dict()),
                )
            )
            if replaced_agent is None and binding.trainable:
                samples.append((agent_id, _training_sample(completion)))
            trajectory_index += 1

        final = env.advance(actions)
        turns.append(
            ReplayTurn(
                turn,
                tuple(sorted(broadcasts.items())),
                tuple(sorted(phase.delivered.items())),
                tuple(sorted(actions.items())),
                pre_state_sha256,
                _state_sha256(env._require_state()),
            )
        )
        if final.terminated or final.truncated:
            break

    if final is None or not (final.terminated or final.truncated):
        raise RuntimeError("live branch did not reach a terminal state")
    return BranchRollout(
        BranchReplay(replaced_agent, tuple(turns), float(final.rewards["BLUE"])),
        tuple(decisions),
        tuple(samples),
    )


async def build_live_credit_group(
    generator: VLLMChoiceGenerator,
    *,
    game_id: str,
    seed: int,
    size: int,
    config: EpisodeConfig,
    bindings: tuple[AgentPolicy, ...],
    policies: tuple[PolicyEndpoint, ...],
    replacement_policy_id: str,
    run_lock_sha256: str,
    initial_state: GameState | None = None,
) -> LiveCreditGroup:
    if initial_state is None:
        bootstrap = ArenaRLEnv(seed=seed, size=size, config=config)
        bootstrap.reset(seed)
        initial_state = bootstrap._require_state().clone()
    else:
        initial_state = initial_state.clone()
        if len(initial_state.nodes) != size:
            raise ValueError("live credit-group size does not match its supplied initial state")
    policy_by_id = {row.policy_id: row for row in policies}
    if len(policy_by_id) != len(policies):
        raise ValueError("policy endpoint IDs must be unique")
    trainable_agents = sorted(
        row.agent_id for row in bindings if row.trainable and row.team == "BLUE"
    )
    if len(trainable_agents) != 4:
        raise ValueError("live credit group requires four trainable BLUE agents")
    branches = await asyncio.gather(
        rollout_branch(
            generator,
            game_id=game_id,
            initial_state=initial_state,
            config=config,
            bindings=bindings,
            policies=policy_by_id,
            replacement_policy_id=replacement_policy_id,
            replaced_agent=None,
        ),
        *(
            rollout_branch(
                generator,
                game_id=game_id,
                initial_state=initial_state,
                config=config,
                bindings=bindings,
                policies=policy_by_id,
                replacement_policy_id=replacement_policy_id,
                replaced_agent=agent_id,
            )
            for agent_id in trainable_agents
        ),
    )
    actual, *replacements = branches
    by_agent: dict[str, list[TrainingSample]] = {agent_id: [] for agent_id in trainable_agents}
    for agent_id, sample in actual.samples:
        by_agent[agent_id].append(sample)
    actual_decisions_by_agent = {
        agent_id: tuple(
            row.decision_id
            for row in actual.decisions
            if row.agent_id == agent_id
        )
        for agent_id in trainable_agents
    }
    binding_by_agent = {row.agent_id: row for row in bindings}
    owned = tuple(
        OwnedAgentSamples(
            game_id,
            agent_id,
            binding_by_agent[agent_id].policy_id,
            actual_decisions_by_agent[agent_id],
            tuple(by_agent[agent_id]),
        )
        for agent_id in trainable_agents
    )
    evidence = CreditGroupEvidence(
        run_lock_sha256,
        game_id,
        initial_state,
        _state_sha256(initial_state),
        config,
        actual.replay,
        tuple(row.replay for row in replacements),
        actual.decisions + tuple(
            decision for branch in replacements for decision in branch.decisions
        ),
        None,
    )
    return LiveCreditGroup(evidence, bindings, owned)
