import json
import tempfile
import unittest
from pathlib import Path

from swarm_ctf_eval.arena import (
    Action,
    AgentState,
    ArenaEnv,
    GameState,
    Node,
    WAIT,
    legal_actions,
    observe_node,
    redundant_agents,
    state_to_dict,
    step,
)
from swarm_ctf_eval.arena_generation import generate_state
from swarm_ctf_eval.arena_eval import OracleArenaModel, evaluate_case
from swarm_ctf_eval.arena_oracle import deterministic_policy, solve_joint_action
from swarm_ctf_eval.arena_protocol import (
    action_prompt,
    encode_action,
    parse_action,
    parse_broadcast,
)
from swarm_ctf_eval.arena_sft import generate_dataset, oracle_broadcast, write_dataset


def two_node_state() -> GameState:
    nodes = {
        "X": Node("X", ("Y",), "BLUE"),
        "Y": Node("Y", ("X",), "RED"),
    }
    agents = {}
    for team, position in (("BLUE", "X"), ("RED", "Y")):
        for index in range(4):
            aid = f"{team.lower()}-{index}"
            agents[aid] = AgentState(aid, team, position, 2)
    knowledge = {
        aid: {node_id: observe_node(node, 0) for node_id, node in nodes.items()}
        for aid in agents
    }
    state = GameState(0, nodes, agents, knowledge)
    state.validate()
    return state


class ArenaTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self) -> None:
        first = generate_state(17)
        second = generate_state(17)
        other = generate_state(18)
        self.assertEqual(state_to_dict(first), state_to_dict(second))
        self.assertNotEqual(state_to_dict(first), state_to_dict(other))
        first.validate()

    def test_probe_capture_is_complementary_not_duplicate(self) -> None:
        state = two_node_state()
        actions = {aid: WAIT for aid in state.agents}
        actions["blue-0"] = Action("PROBE", "Y")
        actions["blue-1"] = Action("CAPTURE", "Y")
        result = step(state, actions)
        self.assertEqual(result.state.nodes["Y"].owner, "BLUE")
        self.assertEqual(result.duplicate_targets["BLUE"], ())
        self.assertEqual(redundant_agents(state, actions, "BLUE"), ())

    def test_simultaneous_fortify_blocks_one_probe(self) -> None:
        state = two_node_state()
        actions = {aid: WAIT for aid in state.agents}
        actions["blue-0"] = Action("PROBE", "Y")
        actions["blue-1"] = Action("CAPTURE", "Y")
        actions["red-0"] = Action("FORTIFY", "Y")
        result = step(state, actions)
        self.assertEqual(result.state.nodes["Y"].owner, "RED")
        self.assertFalse(result.state.nodes["Y"].exposed)

    def test_resolution_does_not_depend_on_action_dict_order(self) -> None:
        state = generate_state(3)
        actions = {
            **deterministic_policy(state, "BLUE"),
            **deterministic_policy(state, "RED"),
        }
        forward = step(state, actions)
        reverse = step(state, dict(reversed(list(actions.items()))))
        self.assertEqual(state_to_dict(forward.state), state_to_dict(reverse.state))
        self.assertEqual(forward.rewards, reverse.rewards)
        self.assertEqual(forward.duplicate_targets, reverse.duplicate_targets)
        self.assertAlmostEqual(forward.rewards["BLUE"] + forward.rewards["RED"], 0.0)

    def test_invalid_action_is_rejected_and_penalized(self) -> None:
        state = two_node_state()
        baseline = step(state, {aid: WAIT for aid in state.agents})
        invalid = {aid: WAIT for aid in state.agents}
        invalid["blue-0"] = Action("CAPTURE", "NOT_A_NODE")
        result = step(state, invalid)
        self.assertEqual(result.invalid_agents, ("blue-0",))
        self.assertEqual(result.rewards["BLUE"], baseline.rewards["BLUE"] - 1.0)

    def test_exact_solver_returns_only_optimal_actions(self) -> None:
        state = generate_state(5)
        red = deterministic_policy(state, "RED")
        solution = solve_joint_action(state, "BLUE", red)
        self.assertGreater(solution.explored, 0)
        self.assertGreaterEqual(solution.optimal_count, len(solution.assignments))
        for assignment in solution.assignments:
            result = step(state, {**red, **dict(assignment)})
            self.assertAlmostEqual(result.rewards["BLUE"], solution.reward)

    def test_strict_protocol_rejects_extra_text_and_unsupported_facts(self) -> None:
        state = generate_state(9)
        aid = "blue-0"
        self.assertFalse(parse_broadcast('answer: {"facts":[],"intent":null,"request_resource":0}', state, aid).valid)
        hallucination = json.dumps(
            {
                "facts": [{"node": "ZZ", "owner": "RED", "status": "EXPOSED", "value": 3, "critical": False, "observed_turn": 0}],
                "intent": None,
                "request_resource": 0,
            }
        )
        self.assertEqual(parse_broadcast(hallucination, state, aid).errors, ("unsupported_fact",))

        known = next(iter(state.knowledge[aid].values()))
        stale_lie = json.dumps(
            {
                "facts": [{"node": known.node, "owner": known.owner, "status": known.status, "value": known.value, "critical": known.critical, "observed_turn": known.observed_turn + 1}],
                "intent": None,
                "request_resource": 0,
            }
        )
        self.assertEqual(parse_broadcast(stale_lie, state, aid).errors, ("unsupported_fact",))

    def test_action_protocol_accepts_exactly_displayed_action(self) -> None:
        state = generate_state(4)
        prompt, displayed = action_prompt(state, "blue-0", [], permutation=2)
        self.assertEqual(prompt[0]["role"], "system")
        target = encode_action(displayed[-1], displayed)
        parsed = parse_action(target, displayed)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.value, displayed[-1])
        self.assertFalse(parse_action(target + "\nthanks", displayed).valid)

    def test_sft_dataset_is_valid_deduplicated_and_seed_isolated(self) -> None:
        rows, manifest = generate_dataset(0, 3)
        self.assertEqual(manifest["num_examples"], len(rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset(rows, manifest, output)
            self.assertTrue((output / "manifest.json").is_file())
            written = sum(len((output / f"{split}.jsonl").read_text().splitlines()) for split in ("train", "validation", "test"))
            self.assertEqual(written, len(rows))

    def test_targeted_sft_covers_wait_scan_and_transfer(self) -> None:
        rows, _ = generate_dataset(0, 1, mechanics_per_kind=1, silence_examples=1)
        skills = {
            row["metadata"].get("targeted_skill")
            for row in rows
            if row["metadata"].get("generator_mode") == "targeted_mechanics"
        }
        self.assertEqual(skills, {"WAIT", "SCAN", "TRANSFER", "SILENCE"})
        silence = next(row for row in rows if row["metadata"].get("targeted_skill") == "SILENCE")
        self.assertEqual(
            json.loads(silence["messages"][-1]["content"]),
            {"facts": [], "intent": None, "request_resource": 0},
        )

    def test_every_generated_agent_has_wait_and_legal_policy_action(self) -> None:
        state = generate_state(21)
        for team in ("BLUE", "RED"):
            policy = deterministic_policy(state, team)
            for aid, action in policy.items():
                self.assertIn(WAIT, legal_actions(state, aid))
                self.assertIn(action, legal_actions(state, aid))

    def test_parallel_environment_runs_to_fixed_horizon(self) -> None:
        env = ArenaEnv(seed=44, horizon=2)
        observations = env.reset()
        self.assertEqual(len(observations), 8)
        for turn in range(2):
            actions = {agent_id: WAIT for agent_id in observations}
            observations, rewards, terminated, truncated, info = env.advance(actions)
            self.assertAlmostEqual(rewards["BLUE"] + rewards["RED"], 0.0)
            self.assertEqual(len(observations), 8)
            self.assertIn("team_value", info)
            if turn == 0:
                self.assertFalse(truncated)
        self.assertTrue(terminated or truncated)

    def test_oracle_has_no_protocol_false_negatives_and_reaches_optimum(self) -> None:
        state = generate_state(101)
        reference = deterministic_policy(state, "BLUE")
        shuffled = {
            aid: oracle_broadcast(state, aid, reference[aid])
            for aid in reference
        }
        row = evaluate_case(OracleArenaModel(), 101, 12, "balanced", shuffled)
        self.assertEqual(row["message_strict_rate"], 1.0)
        self.assertTrue(row["action_order_consistent"])
        for condition in row["conditions"]:
            self.assertEqual(condition["strict_action_rate"], 1.0)
            self.assertTrue(condition["optimal_outcome"])


if __name__ == "__main__":
    unittest.main()
