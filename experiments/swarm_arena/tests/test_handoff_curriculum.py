from swarm_ctf_eval import handoff_curriculum
from swarm_ctf_eval.arena import Action


def test_information_changes_choice_without_unlocking_actions() -> None:
    pair = handoff_curriculum.generate_pair(
        10_000_019,
        12,
        4,
        role_pair=("blue-0", "blue-1"),
    )
    assert pair is not None
    critical, decoy = pair
    audit = handoff_curriculum.matched_pair_audit(critical, decoy)
    assert all(audit.values())
    assert critical.minimum_advantage > 0
    assert decoy.minimum_advantage == 0
    assert all(
        certificate.dropped_information_sets == 1
        for certificate in critical.certificates
    )
    assert all(
        certificate.informed_information_sets == 2
        for certificate in critical.certificates
    )
    assert all(
        certificate.dropped_information_sets == 2
        for certificate in decoy.certificates
    )
    for certificate in critical.certificates:
        actions = dict(certificate.informed_actions)
        assert all(
            actions[world.label] == Action("CAPTURE", world.active_target).to_dict()
            for world in critical.worlds
        )
    separation = handoff_curriculum.exhaustive_receiver_target_separation(critical)
    assert separation["all_strictly_positive"] is True
    assert separation["minimum_advantage"] > 0
    assert separation["worlds"] == ["left_exposed", "right_exposed"]
    assert separation["opponent_styles"] == ["aggressive", "balanced", "defensive"]


def test_manifest_balances_all_ordered_roles_on_hard_maps() -> None:
    manifest = handoff_curriculum.generate_manifest(
        count=12,
        seed_start=12_000_041,
        sizes=(18, 20),
        horizons=(8, 10),
    )
    roles = {
        (pair["critical"]["sender"], pair["critical"]["receiver"])
        for pair in manifest["pairs"]
    }
    assert len(roles) == 12
    assert all(
        all(pair["matched_pair_audit"].values()) for pair in manifest["pairs"]
    )


def test_coprime_size_cycle_breaks_role_size_correlation() -> None:
    sizes = (12, 14, 16, 18, 20)
    manifest = handoff_curriculum.generate_manifest(
        count=60,
        seed_start=15_000_083,
        sizes=sizes,
        horizons=(4, 5, 6, 8, 10, 12, 14),
    )
    sizes_by_role: dict[tuple[str, str], set[int]] = {}
    for pair in manifest["pairs"]:
        critical = pair["critical"]
        role = (critical["sender"], critical["receiver"])
        sizes_by_role.setdefault(role, set()).add(critical["size"])

    assert len(sizes_by_role) == 12
    assert all(role_sizes == set(sizes) for role_sizes in sizes_by_role.values())
