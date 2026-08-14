"""Pure tests for the disabled D38999 pose-registry v2 migration."""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_pose_registry_v2_migration import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    audit_obj_vertex_symmetry,
    evaluate_pose_registry_v2_migration,
    evaluate_stage_authorization,
    load_pose_registry_v2_migration_contract,
    main,
    orientation_error_modulo_axial_symmetry,
    relative_yaw_hypotheses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "d38999_pose_registry_v2_migration.py"
)
ACTIVE_REGISTRY = (
    Path(__file__).parents[1]
    / "config"
    / "connector_pose_observation_v1.yaml"
)
E2E_RUNNER = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract():
    return load_pose_registry_v2_migration_contract()


def _mutated(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    output = tmp_path / "migration.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return output


def test_contract_is_cpu_only_disabled_and_not_wired_into_e2e():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    import_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_roots.add(node.module.split(".")[0])
    assert import_roots.isdisjoint(
        {
            "isaacsim",
            "omni",
            "pxr",
            "rclpy",
            "torch",
            "tensorrt",
            "onnx",
            "cv2",
            "open3d",
            "numpy",
        }
    )
    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled is False
    assert contract.status == "audit_confirmed_v2_designed_not_activated"
    assert all(value is False for value in contract.boundaries.values())
    assert "d38999_pose_registry_v2_migration" not in E2E_RUNNER.read_text(
        encoding="utf-8"
    )


def test_active_registry_is_hash_bound_but_not_modified():
    before = ACTIVE_REGISTRY.read_bytes()
    contract = _contract()
    assert contract.inputs["active_pose_registry_v1"].sha256 == _sha256(
        ACTIVE_REGISTRY
    )
    assert contract.active_declared_symmetry_class == "keyed_order_1"
    report = evaluate_pose_registry_v2_migration(contract, PROJECT_ROOT)
    assert ACTIVE_REGISTRY.read_bytes() == before
    assert report["active_registry"]["modified_by_migration"] is False
    assert report["active_registry"]["affected_symmetry_classes"] == {
        "d38999_26kj61sn_proxy_v1": "keyed_order_1",
        "d38999_20kj61pn_proxy_v1": "keyed_order_1",
    }


def test_exact_proxy_objs_pass_declared_vertex_orbit_audit():
    contract = _contract()
    for object_id, spec in contract.observed_symmetry.items():
        mesh = PROJECT_ROOT / contract.inputs[spec.geometry_input].path
        result = audit_obj_vertex_symmetry(
            mesh,
            orders=spec.required_verified_orders,
            decimal_places=contract.coordinate_decimal_places,
        )
        assert result
        assert all(result.values()), object_id
    assert contract.observed_symmetry["loose_plug"].order == 2
    assert contract.observed_symmetry["fixed_receptacle"].order == 2
    assert contract.observed_symmetry["coupling_nut"].order == 24


def test_proposed_v2_uses_equivalence_classes_and_separate_nut():
    contract = _contract()
    assert {model.role for model in contract.proposed_models.values()} == {
        "loose_plug",
        "fixed_receptacle",
    }
    assert {
        model.symmetry.order for model in contract.proposed_models.values()
    } == {2}
    assert all(
        model.symmetry.unique_key_geometry_present is False
        for model in contract.proposed_models.values()
    )
    assert contract.proposed_component.order == 24
    assert contract.proposed_component.object_id == "coupling_nut"


def test_modulo_orientation_metric_accepts_pi_for_order_two_only():
    identity = (0.0, 0.0, 0.0, 1.0)
    yaw_pi = (0.0, 0.0, 1.0, 0.0)
    modulo_two = orientation_error_modulo_axial_symmetry(
        yaw_pi, identity, symmetry_order=2
    )
    assert modulo_two.error_rad == pytest.approx(0.0)
    assert modulo_two.selected_symmetry_index == 1
    assert modulo_two.equivalent_yaw_offset_rad == pytest.approx(math.pi)
    keyed = orientation_error_modulo_axial_symmetry(
        yaw_pi, identity, symmetry_order=1
    )
    assert keyed.error_rad == pytest.approx(math.pi)


def test_pair_policy_retains_relative_yaw_branches():
    assert relative_yaw_hypotheses(2, 2) == pytest.approx((0.0, math.pi))
    assert len(relative_yaw_hypotheses(2, 3)) == 6


def test_readiness_report_distinguishes_design_from_runtime_and_control():
    report = evaluate_pose_registry_v2_migration(_contract(), PROJECT_ROOT)
    json.dumps(report, allow_nan=False)
    assert report["status"] == "AUDIT_CONFIRMED_V2_DESIGNED_NOT_ACTIVATED"
    assert report["gates"]["content_addressed_inputs_verified"] is True
    assert report["gates"]["active_v1_mismatch_confirmed"] is True
    assert report["gates"]["v2_migration_contract_parser_ready"] is True
    assert report["gates"]["production_v2_pose_parser_ready"] is False
    assert report["gates"]["hash_bound_unique_key_geometry_available"] is False
    assert report["gates"]["foundationpose_shadow_accuracy_qualified"] is False
    assert report["gates"]["v2_runtime_activated"] is False
    assert report["gates"]["vision_control_authorized"] is False
    assert report["key_geometry"][
        "polarization_N_label_counts_as_geometry_evidence"
    ] is False
    assert report["proposed_v2"][
        "loose_fixed_relative_yaw_hypothesis_count"
    ] == 2


@pytest.mark.parametrize(
    "stage",
    (
        "evaluation_or_preflight_pair_publication",
        "symmetry_invariant_pick_only",
        "keyed_assembly_full_workflow",
    ),
)
def test_stage_authorization_requires_activation_and_every_gate(stage):
    contract = _contract()
    required = contract.control_requirements[stage]
    missing = evaluate_stage_authorization(
        contract,
        stage=stage,
        passed_gates=(),
        migration_activated=False,
    )
    assert missing.authorized is False
    assert missing.missing_gates == required
    inactive = evaluate_stage_authorization(
        contract,
        stage=stage,
        passed_gates=required,
        migration_activated=False,
    )
    assert inactive.all_required_gates_passed is True
    assert inactive.authorized is False
    future = evaluate_stage_authorization(
        contract,
        stage=stage,
        passed_gates=required,
        migration_activated=True,
    )
    assert future.would_authorize_in_future_active_contract is True
    assert future.authorized is False


def test_cli_is_read_only_by_default(capsys):
    assert main(["--repository", str(PROJECT_ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "AUDIT_CONFIRMED_V2_DESIGNED_NOT_ACTIVATED"
    assert report["control"]["current_authorized"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda doc: doc.update({"enabled": True}), "must remain disabled"),
        (
            lambda doc: doc["current_mismatch"]["observed_objects"][
                "loose_plug"
            ].update({"unique_key_geometry_present": True}),
            "cannot claim unique key geometry",
        ),
        (
            lambda doc: doc["proposed_v2"]["model_registry"][0][
                "symmetry"
            ].update({"order": 1, "equivalent_yaw_period_rad": 2.0 * math.pi}),
            "differs from audited geometry",
        ),
        (
            lambda doc: doc["key_geometry_upgrade"].update(
                {"manually_drawn_or_guessed_key_allowed": True}
            ),
            "cannot be guessed",
        ),
        (
            lambda doc: doc["foundationpose_evaluation"].update(
                {"canonical_representative_may_authorize_control": True}
            ),
            "cannot authorize control",
        ),
        (
            lambda doc: doc["control_authorization"].update(
                {"current_authorized": True}
            ),
            "must remain unauthorized",
        ),
        (
            lambda doc: doc["boundaries"].update(
                {"robot_control_authorized": True}
            ),
            "must all remain false",
        ),
    ),
)
def test_contract_rejects_unsafe_claim_mutations(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_pose_registry_v2_migration_contract(_mutated(tmp_path, mutate))


def test_wrong_bound_input_hash_fails_before_reporting(tmp_path):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["inputs"]["active_pose_registry_v1"]["sha256"] = "a" * 64
    config = tmp_path / "wrong_hash.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    contract = load_pose_registry_v2_migration_contract(config)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_pose_registry_v2_migration(contract, PROJECT_ROOT)
