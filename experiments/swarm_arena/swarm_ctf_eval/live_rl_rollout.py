from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal

import httpx
import torch
import xgrammar as xgr
from renderers import Qwen3Renderer, Qwen3RendererConfig
from transformers import AutoConfig, AutoTokenizer
from vllm.v1.structured_output.utils import choice_as_grammar

from prime_rl.transport import TrainingSample

from .arena import Action, GameState, state_to_dict
from .arena_protocol import Broadcast, parse_action, parse_broadcast
from .episode import EMPTY_BROADCAST, EpisodeConfig
from .episode_protocol import episode_action_prompt, episode_broadcast_prompt
from .multi_policy_contract import AgentPolicy
from .prime_multi_run_router import OwnedAgentSamples
from .prime_rl_bridge import RolloutDecision
from .rl_v3 import ArenaRLEnv
from .safety_supervisor import (
    BranchReplay,
    CreditGroupEvidence,
    MessageCreditGroupEvidence,
    ReplayTurn,
    SharedReturnGroupEvidence,
    SharedReturnReplicaEvidence,
    SharedReturnSpec,
    canonical_sha256,
)
from .structured_protocol import (
    STRUCTURED_PROTOCOL_VERSION,
    protocol_choices,
)

Phase = Literal["BROADCAST", "ACT"]


class XGrammarChoiceMask:
    """Reconstruct vLLM's exact choice-grammar mask along a sampled token path."""

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size,
        )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )
        self.vocab_size = vocab_size
        token_ids = torch.arange(vocab_size, dtype=torch.int64)
        self._token_ids = token_ids
        self._word_indices = token_ids // 32
        self._bit_indices = token_ids % 32

    def allowed_token_ids(
        self,
        choices: tuple[str, ...],
        completion_ids: list[int],
    ) -> list[list[int]]:
        compiled = self.compiler.compile_grammar(choice_as_grammar(list(choices)))
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        rows: list[list[int]] = []
        for token_id in completion_ids:
            xgr.reset_token_bitmask(bitmask)
            matcher.fill_next_token_bitmask(bitmask)
            words = bitmask[0].to(torch.int64)
            accepted = (
                (words[self._word_indices] >> self._bit_indices) & 1
            ).to(torch.bool)
            allowed = self._token_ids[accepted].tolist()
            if token_id not in allowed or not matcher.accept_token(token_id):
                raise ValueError(
                    f"completion token {token_id} is rejected by the serving choice grammar"
                )
            rows.append(allowed)
        if not matcher.is_terminated():
            raise ValueError("structured completion ended before its choice grammar terminated")
        return rows


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
    request_sha256: str


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


@dataclass(frozen=True)
class LiveMessageCreditGroup:
    evidence: MessageCreditGroupEvidence
    bindings: tuple[AgentPolicy, ...]
    owned_samples: tuple[OwnedAgentSamples, ...]


@dataclass(frozen=True)
class LiveSharedReturnGroup:
    evidence: SharedReturnGroupEvidence
    bindings: tuple[AgentPolicy, ...]
    owned_samples_by_replica: tuple[tuple[OwnedAgentSamples, ...], ...]


def protocol_constraint_sha256(phase: Phase) -> str:
    return canonical_sha256({"phase": phase, "version": STRUCTURED_PROTOCOL_VERSION})


def parity_gate_sha256(config: object) -> str:
    if not hasattr(config, "model_dump"):
        raise TypeError("parity gate config must support model_dump")
    return canonical_sha256(config.model_dump(mode="json"))


@asynccontextmanager
async def _coalesced_request_group(generator: object) -> AsyncIterator[None]:
    scope = getattr(generator, "coalesced_request_group", None)
    if scope is None:
        yield
        return
    async with scope():
        yield


class VLLMChoiceGenerator:
    def __init__(self, tokenizer_path: str, *, timeout: float = 180.0) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.renderer = Qwen3Renderer(
            self.tokenizer,
            Qwen3RendererConfig(enable_thinking=False),
        )
        config = AutoConfig.from_pretrained(tokenizer_path)
        self.choice_mask = XGrammarChoiceMask(self.tokenizer, int(config.vocab_size))
        self.timeout = timeout
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._group_requests: dict[str, asyncio.Task[ChoiceCompletion]] | None = None

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

    @asynccontextmanager
    async def coalesced_request_group(self) -> AsyncIterator[None]:
        if self._group_requests is not None:
            raise RuntimeError("inference request groups cannot be nested")
        self._group_requests = {}
        try:
            yield
        finally:
            self._group_requests = None

    async def _complete_request(
        self,
        *,
        base_url: str,
        request_body: dict[str, Any],
        choices: tuple[str, ...],
        prompt_ids: tuple[int, ...],
        request_sha256: str,
    ) -> ChoiceCompletion:
        response = await self._client(base_url).post(
            "/inference/v1/generate",
            json=request_body,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        completion_ids = list(choice["token_ids"])
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        try:
            allowed = self.choice_mask.allowed_token_ids(choices, completion_ids)
        except ValueError as error:
            matching_choices = [
                index for index, value in enumerate(choices) if value == decoded
            ]
            raise ValueError(
                "structured completion/trie mismatch: "
                + json.dumps(
                    {
                        "completion_ids": completion_ids,
                        "decoded": decoded,
                        "eos_token_id": eos_token_id,
                        "finish_reason": choice.get("finish_reason"),
                        "matching_choice_indices": matching_choices,
                        "matching_choice_token_ids": [
                            [
                                *self.tokenizer.encode(
                                    choices[index], add_special_tokens=False
                                ),
                                eos_token_id,
                            ]
                            for index in matching_choices
                        ],
                    },
                    sort_keys=True,
                )
            ) from error
        return ChoiceCompletion(
            prompt_ids,
            tuple(completion_ids),
            tuple(item["logprob"] for item in choice["logprobs"]["content"]),
            tuple(tuple(row) for row in allowed),
            decoded,
            request_sha256,
        )

    async def generate(
        self,
        endpoint: PolicyEndpoint,
        messages: list[dict[str, str]],
        *,
        sampling_key: str,
    ) -> ChoiceCompletion:
        endpoint.validate()
        choices = protocol_choices(messages)
        prompt_ids = tuple(self.renderer.render_ids(messages, add_generation_prompt=True))
        digest = hashlib.sha256(sampling_key.encode()).digest()
        base_url = endpoint.base_urls[int.from_bytes(digest[:4], "big") % len(endpoint.base_urls)]
        seed = int.from_bytes(digest[4:8], "big")
        request_body = {
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
        }
        request_sha256 = canonical_sha256(
            {
                "base_url": base_url,
                "policy_id": endpoint.policy_id,
                "policy_revision": endpoint.revision,
                "sampling_key": sampling_key,
                "request_body": request_body,
            }
        )
        request = self._complete_request(
            base_url=base_url,
            request_body=request_body,
            choices=choices,
            prompt_ids=prompt_ids,
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
    replacement_policy_id: str | None,
) -> PolicyEndpoint:
    if agent_id == replaced_agent:
        if replacement_policy_id is None:
            raise ValueError("replacement branch requires a replacement policy")
        policy_id = replacement_policy_id
    else:
        policy_id = bindings[agent_id].policy_id
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
    replacement_policy_id: str | None,
    replaced_agent: str | None,
    sampling_namespace: str,
    message_drop_agent: str | None = None,
    message_drop_turn: int | None = None,
    sample_phases: frozenset[Phase] = frozenset({"BROADCAST", "ACT"}),
    sample_turns: frozenset[int] | None = None,
) -> BranchRollout:
    if replaced_agent is not None and message_drop_agent is not None:
        raise ValueError("a branch cannot replace a policy and drop a message simultaneously")
    if (message_drop_agent is None) != (message_drop_turn is None):
        raise ValueError("message drop requires both a sender and an intervention turn")
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
            key = _sampling_key(sampling_namespace, agent_id, turn, "BROADCAST")
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
            branch = (
                "replacement"
                if replaced_agent is not None
                else "message_drop"
                if message_drop_agent is not None
                else "actual"
            )
            decisions.append(
                RolloutDecision(
                    game_id,
                    branch,
                    replaced_agent or message_drop_agent,
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
                    completion.request_sha256,
                    canonical_sha256(broadcast.to_dict()),
                    completion.allowed_token_ids,
                )
            )
            if (
                branch == "actual"
                and binding.trainable
                and "BROADCAST" in sample_phases
                and (sample_turns is None or turn in sample_turns)
            ):
                samples.append((agent_id, _training_sample(completion)))
            trajectory_index += 1

        delivered = dict(broadcasts)
        if message_drop_agent is not None and turn == message_drop_turn:
            if message_drop_agent not in delivered:
                raise ValueError(f"unknown message-drop sender: {message_drop_agent}")
            delivered[message_drop_agent] = EMPTY_BROADCAST
        phase = env.broadcast_phase(broadcasts, delivered_broadcasts=delivered)
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
            key = _sampling_key(sampling_namespace, agent_id, turn, "ACT")
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
            branch = (
                "replacement"
                if replaced_agent is not None
                else "message_drop"
                if message_drop_agent is not None
                else "actual"
            )
            decisions.append(
                RolloutDecision(
                    game_id,
                    branch,
                    replaced_agent or message_drop_agent,
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
                    completion.request_sha256,
                    canonical_sha256(action.to_dict()),
                    completion.allowed_token_ids,
                )
            )
            if (
                branch == "actual"
                and binding.trainable
                and "ACT" in sample_phases
                and (sample_turns is None or turn in sample_turns)
            ):
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
        BranchReplay(replaced_agent or message_drop_agent, tuple(turns), float(final.rewards["BLUE"])),
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
    sampling_namespace: str | None = None,
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
    resolved_sampling_namespace = sampling_namespace or game_id
    async with _coalesced_request_group(generator):
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
                sampling_namespace=resolved_sampling_namespace,
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
                    sampling_namespace=resolved_sampling_namespace,
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


async def build_live_message_credit_group(
    generator: VLLMChoiceGenerator,
    *,
    game_id: str,
    seed: int,
    size: int,
    config: EpisodeConfig,
    bindings: tuple[AgentPolicy, ...],
    policies: tuple[PolicyEndpoint, ...],
    run_lock_sha256: str,
    initial_state: GameState | None = None,
    sampling_namespace: str | None = None,
    intervention_turn: int | None = None,
) -> LiveMessageCreditGroup:
    """Build one actual rollout and four sender-message-drop counterfactuals."""
    if initial_state is None:
        bootstrap = ArenaRLEnv(seed=seed, size=size, config=config)
        bootstrap.reset(seed)
        initial_state = bootstrap._require_state().clone()
    else:
        initial_state = initial_state.clone()
        if len(initial_state.nodes) != size:
            raise ValueError("live message-credit size does not match its supplied initial state")
    policy_by_id = {row.policy_id: row for row in policies}
    if len(policy_by_id) != len(policies):
        raise ValueError("policy endpoint IDs must be unique")
    trainable_agents = sorted(
        row.agent_id for row in bindings if row.trainable and row.team == "BLUE"
    )
    if len(trainable_agents) != 4:
        raise ValueError("live message-credit group requires four trainable BLUE senders")
    resolved_sampling_namespace = sampling_namespace or game_id
    resolved_intervention_turn = (
        initial_state.turn if intervention_turn is None else intervention_turn
    )
    if resolved_intervention_turn != initial_state.turn:
        raise ValueError("bootstrap message credit currently supports the first turn only")
    async with _coalesced_request_group(generator):
        branches = await asyncio.gather(
            rollout_branch(
                generator,
                game_id=game_id,
                initial_state=initial_state,
                config=config,
                bindings=bindings,
                policies=policy_by_id,
                replacement_policy_id=None,
                replaced_agent=None,
                sampling_namespace=resolved_sampling_namespace,
                sample_phases=frozenset({"BROADCAST"}),
                sample_turns=frozenset({resolved_intervention_turn}),
            ),
            *(
                rollout_branch(
                    generator,
                    game_id=game_id,
                    initial_state=initial_state,
                    config=config,
                    bindings=bindings,
                    policies=policy_by_id,
                    replacement_policy_id=None,
                    replaced_agent=None,
                    sampling_namespace=resolved_sampling_namespace,
                    message_drop_agent=agent_id,
                    message_drop_turn=resolved_intervention_turn,
                    sample_phases=frozenset(),
                )
                for agent_id in trainable_agents
            ),
        )
    actual, *drops = branches
    by_agent: dict[str, list[TrainingSample]] = {agent_id: [] for agent_id in trainable_agents}
    for agent_id, sample in actual.samples:
        by_agent[agent_id].append(sample)
    decision_ids_by_agent = {
        agent_id: tuple(
            row.decision_id
            for row in actual.decisions
            if row.agent_id == agent_id
            and row.phase == "BROADCAST"
            and row.turn == resolved_intervention_turn
        )
        for agent_id in trainable_agents
    }
    binding_by_agent = {row.agent_id: row for row in bindings}
    owned = tuple(
        OwnedAgentSamples(
            game_id,
            agent_id,
            binding_by_agent[agent_id].policy_id,
            decision_ids_by_agent[agent_id],
            tuple(by_agent[agent_id]),
        )
        for agent_id in trainable_agents
    )
    evidence = MessageCreditGroupEvidence(
        run_lock_sha256,
        game_id,
        initial_state,
        _state_sha256(initial_state),
        config,
        resolved_intervention_turn,
        actual.replay,
        tuple(row.replay for row in drops),
        actual.decisions + tuple(
            decision for branch in drops for decision in branch.decisions
        ),
        None,
    )
    return LiveMessageCreditGroup(evidence, bindings, owned)


async def build_live_shared_return_group(
    generator: VLLMChoiceGenerator,
    *,
    group_id: str,
    seed: int,
    size: int,
    config: EpisodeConfig,
    spec: SharedReturnSpec,
    bindings: tuple[AgentPolicy, ...],
    policies: tuple[PolicyEndpoint, ...],
    run_lock_sha256: str,
    initial_state: GameState | None = None,
    sampling_namespace: str | None = None,
) -> LiveSharedReturnGroup:
    """Sample independent joint trajectories from one state for LOO team-return credit."""
    spec.validate()
    if initial_state is None:
        bootstrap = ArenaRLEnv(seed=seed, size=size, config=config)
        bootstrap.reset(seed)
        initial_state = bootstrap._require_state().clone()
    else:
        initial_state = initial_state.clone()
        if len(initial_state.nodes) != size:
            raise ValueError("live shared-return size does not match its supplied initial state")
    policy_by_id = {row.policy_id: row for row in policies}
    if len(policy_by_id) != len(policies):
        raise ValueError("policy endpoint IDs must be unique")
    trainable_agents = sorted(
        row.agent_id for row in bindings if row.trainable and row.team == "BLUE"
    )
    if len(trainable_agents) != 4:
        raise ValueError("live shared-return group requires four trainable BLUE agents")
    base_namespace = sampling_namespace or group_id
    absolute_turns = frozenset(
        initial_state.turn + offset for offset in spec.trainable_turn_offsets
    )
    phases = frozenset(spec.trainable_phases)
    replica_game_ids = tuple(
        f"{group_id}:replica-{index}" for index in range(spec.replicas)
    )
    replica_namespaces = tuple(
        f"{base_namespace}:replica-{index}" for index in range(spec.replicas)
    )
    async with _coalesced_request_group(generator):
        branches = await asyncio.gather(
            *(
                rollout_branch(
                    generator,
                    game_id=replica_game_ids[index],
                    initial_state=initial_state,
                    config=config,
                    bindings=bindings,
                    policies=policy_by_id,
                    replacement_policy_id=None,
                    replaced_agent=None,
                    sampling_namespace=replica_namespaces[index],
                    sample_phases=phases,
                    sample_turns=absolute_turns,
                )
                for index in range(spec.replicas)
            )
        )

    binding_by_agent = {row.agent_id: row for row in bindings}
    owned_by_replica = []
    replicas = []
    for index, branch in enumerate(branches):
        by_agent: dict[str, list[TrainingSample]] = {
            agent_id: [] for agent_id in trainable_agents
        }
        for agent_id, sample in branch.samples:
            by_agent[agent_id].append(sample)
        decision_ids_by_agent = {
            agent_id: tuple(
                row.decision_id
                for row in branch.decisions
                if row.agent_id == agent_id
                and row.phase in phases
                and row.turn in absolute_turns
            )
            for agent_id in trainable_agents
        }
        owned_by_replica.append(
            tuple(
                OwnedAgentSamples(
                    replica_game_ids[index],
                    agent_id,
                    binding_by_agent[agent_id].policy_id,
                    decision_ids_by_agent[agent_id],
                    tuple(by_agent[agent_id]),
                )
                for agent_id in trainable_agents
            )
        )
        replicas.append(
            SharedReturnReplicaEvidence(
                index,
                replica_game_ids[index],
                replica_namespaces[index],
                branch.replay,
                branch.decisions,
            )
        )
    evidence = SharedReturnGroupEvidence(
        run_lock_sha256,
        group_id,
        initial_state,
        _state_sha256(initial_state),
        config,
        spec,
        tuple(replicas),
        None,
    )
    return LiveSharedReturnGroup(evidence, bindings, tuple(owned_by_replica))
