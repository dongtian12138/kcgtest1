import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
PROBE = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_multilayer_init_timeline_probe.py"
)
SOURCE = PROBE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _called_attributes() -> list[str]:
    return [
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_exact_dynamic_task_and_output_are_locked() -> None:
    assert 'TASK_ID = "DYN-A1-INIT-ROOTCAUSE"' in SOURCE
    assert (
        "artifacts/agent_control/tasks/DYN-A1-INIT-ROOTCAUSE/DIAGNOSTIC_T0_T7"
        in SOURCE
    )
    assert 'dynamic.get("diagnostic_runs_started") != 1' in SOURCE
    assert 'dynamic.get("diagnostic_runs_completed") != 0' in SOURCE


def test_all_frozen_input_hashes_are_exact() -> None:
    for digest in (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
        "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec",
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
        "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    ):
        assert digest in SOURCE


def test_timeline_contains_every_required_stage() -> None:
    for stage in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        assert f'timeline["{stage}"]' in SOURCE
    assert "fixed_receptacle" in SOURCE
    assert "body_assembly" in SOURCE
    assert "coupling_nut" in SOURCE
    assert "nut_body_relative_position_m" in SOURCE


def test_one_explicit_step_per_case_is_required() -> None:
    assert SOURCE.count("world.step(render=False)") == 1
    assert '"explicit_physics_step_count": 1' in SOURCE
    assert '"explicit_physics_step_count": 2' in SOURCE


def test_probe_never_writes_object_pose_after_start() -> None:
    called = _called_attributes()
    assert "set_world_pose" not in called
    assert "set_local_pose" not in called
    assert "set_default_state" not in called
    assert '"object_pose_write_after_physics_start_count": 0' in SOURCE


def test_collision_contrast_is_memory_only_and_restored() -> None:
    assert 'case_id="COLLISION_ENABLED_BASELINE"' in SOURCE
    assert 'case_id="COLLISION_DISABLED_IN_MEMORY"' in SOURCE
    assert "_set_collision_state(colliders, False)" in SOURCE
    assert "_set_collision_state(colliders, True)" in SOURCE
    assert "save_stage" not in SOURCE
    assert ".Save(" not in SOURCE
    assert '"source_asset_written": False' in SOURCE


def test_contact_ranking_contains_required_axes() -> None:
    assert '"maximum_penetration_m"' in SOURCE
    assert '"maximum_impulse_norm_n_s"' in SOURCE
    assert '"persistence_sample_count"' in SOURCE
    assert '"top_20"' in SOURCE


def test_only_authorized_root_cause_labels_are_emitted() -> None:
    for label in (
        "RESET_DEFAULT_STATE_MISMATCH",
        "INITIAL_COLLISION_DEPENETRATION",
        "DATUM_MEASUREMENT_ERROR",
    ):
        assert label in SOURCE


def test_rejected_offset_route_is_not_reimplemented() -> None:
    for forbidden in (
        "CreateContactOffsetAttr",
        "CreateRestOffsetAttr",
        "physxCollision:contactOffset",
        "physxCollision:restOffset",
    ):
        assert forbidden not in SOURCE
    assert '"rejected_hypothesis_retried": False' in SOURCE


def test_truth_is_posthoc_and_never_claims_dynamic_pass() -> None:
    assert '"control_consumed_contact_names": False' in SOURCE
    assert '"control_consumed_contact_normals": False' in SOURCE
    assert '"control_consumed_event_truth": False' in SOURCE
    assert '"dynamic_pass_claimed": False' in SOURCE
    assert '"diagnostic_only": True' in SOURCE
