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
