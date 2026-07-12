from symbolic_tool_calling_v1.curation import _select_stratified
from symbolic_tool_calling_v1.generator import generate_task


def test_stratified_selection_is_deterministic_and_balanced():
    candidates = []
    for successes in (1, 2, 3):
        for seed in range(5):
            candidates.append((generate_task(100 * successes + seed), successes))
    first = _select_stratified(candidates, 9, (1, 2, 3))
    second = _select_stratified(list(reversed(candidates)), 9, (1, 2, 3))
    assert first == second
    assert {successes: sum(value == successes for _, value in first) for successes in (1, 2, 3)} == {
        1: 3,
        2: 3,
        3: 3,
    }
