"""Pure tests for the disabled, independent visual-XY target adapter."""

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_visual_xy_control_adapter import (
    DEFAULT_CONFIG_PATH,
    DIAGNOSTICS_SCHEMA_VERSION,
    SCHEMA_VERSION,
    adapt_visual_xy_to_world_targets,
    load_visual_xy_control_adapter_contract,
    observe_and_adapt_visual_xy,
)
from kcg_connector.pose_provider import (
    POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
    PoseProviderPurpose,
    PoseProviderSample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kcg_connector/d38999_visual_xy_control_adapter.py"
)
E2E_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac/d38999_tabletop_pick_smoke.py"
)


def _contract():
    return load_visual_xy_control_adapter_contract(
        repository=PROJECT_ROOT
    )


def _diagnostics(
    *,
    loose_xy=(0.530, -0.200),
    fixed_xy=(0.560, 0.190),
    loose_timestamp=9.90,
    fixed_timestamp=9.91,
    confidence=0.90,
    error_bound=0.009,
):
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "estimator": (
            "ray_plane_registered_model_height_world_xy_only"
        ),
        "endpoints": {
            "loose_plug": {
                "estimated_world_xy_m": list(loose_xy),
                "timestamp_s": loose_timestamp,
                "confidence": confidence,
                "xy_error_bound_m": error_bound,
            },
            "fixed_receptacle": {
                "estimated_world_xy_m": list(fixed_xy),
                "timestamp_s": fixed_timestamp,
                "confidence": confidence,
                "xy_error_bound_m": error_bound,
            },
        },
    }


def _sample(*, diagnostics=None, truth_orientation=True):
    contract = _contract()
    return PoseProviderSample(
        schema_version=POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
        purpose=PoseProviderPurpose.PREFLIGHT,
        provider_id="d38999.masked_rgbd.xy",
        provider_version="v1",
        capture_id="capture-001",
        clock_domain=contract.required_clock_domain,
        control_frame=contract.required_control_frame,
        calibration_sha256=contract.required_calibration_sha256,
        pair=None,
        reference_truth_pair=None,
        full_6d=False,
        keyed_orientation_observed=False,
        uses_truth_position=False,
        uses_truth_orientation=truth_orientation,
        control_authorized=False,
        preflight_passed=False,
        diagnostics=(
            _diagnostics() if diagnostics is None else diagnostics
        ),
    )


def _adapt(sample=None, **kwargs):
    return adapt_visual_xy_to_world_targets(
        _contract(),
        _sample() if sample is None else sample,
        now_s=10.0,
        explicit_independent_probe_opt_in=True,
        orientation_source="sim_ground_truth",
        **kwargs,
    )


def test_contract_is_pure_disabled_hash_bound_and_not_wired_to_e2e():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "rclpy", "torch", "cv2", "open3d"}
    )

    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled_by_default is False
    assert contract.validation_trial_count == 5
    assert contract.validation_maximum_observed_xy_error_m == pytest.approx(
        0.006183311038175034
    )
    assert contract.gates.maximum_xy_error_bound_m == 0.010
    assert contract.boundaries["production_control_authorized"] is False
    assert all(path.is_file() for path in contract.input_paths.values())
    assert "d38999_visual_xy_control_adapter" not in E2E_PATH.read_text(
        encoding="utf-8"
    )


def test_bounded_xy_shifts_pick_and_assembly_targets_but_preserves_z():
    result = _adapt()
    assert result.eligible_for_independent_probe is True
    assert result.rejection_reasons == ()
    assert result.loose_translation_xy_m == pytest.approx((0.010, 0.010))
    assert result.fixed_translation_xy_m == pytest.approx((0.010, 0.005))
    assert result.world_targets["pregrasp_tcp"] == pytest.approx(
        (0.530, -0.200, 0.360)
    )
    assert result.world_targets["grasp_tcp"] == pytest.approx(
        (0.530, -0.200, 0.24848)
    )
    assert result.world_targets["transport_safe_tcp"] == pytest.approx(
        (0.560, 0.190, 0.3435)
    )
    assert result.world_targets["engage_tcp"] == pytest.approx(
        (0.560, 0.190, 0.31298)
    )
    evidence = result.to_mapping()
    assert evidence["translation_source"].startswith("vision_")
    assert evidence["orientation_source"] == "sim_ground_truth"
    assert evidence["uses_truth_orientation"] is True
    assert evidence["yaw_observed"] is False
    assert evidence["full_6d"] is False
    assert evidence["production_control_authorized"] is False
    assert evidence["collision_free_ik_verified"] is False
    json.dumps(evidence, allow_nan=False)


def test_registered_nominal_orientation_is_disclosed_without_full_6d():
    sample = _sample(truth_orientation=False)
    result = adapt_visual_xy_to_world_targets(
        _contract(),
        sample,
        now_s=10.0,
        explicit_independent_probe_opt_in=True,
        orientation_source="registered_nominal",
    )
    assert result.eligible_for_independent_probe is True
    assert result.orientation_source == "registered_nominal"
    assert result.uses_truth_orientation is False
    assert result.to_mapping()["full_6d"] is False


def test_pose_provider_protocol_is_called_only_for_preflight():
    class Provider:
        def __init__(self):
            self.calls = []

        def observe_pair(self, purpose, now_s):
            self.calls.append((purpose, now_s))
            return _sample()

    provider = Provider()
    result = observe_and_adapt_visual_xy(
        _contract(),
        provider,
        now_s=10.0,
        explicit_independent_probe_opt_in=True,
        orientation_source="sim_ground_truth",
    )
    assert result.eligible_for_independent_probe is True
    assert provider.calls == [(PoseProviderPurpose.PREFLIGHT, 10.0)]


def test_default_disabled_path_returns_no_consumable_targets():
    result = adapt_visual_xy_to_world_targets(
        _contract(),
        _sample(),
        now_s=10.0,
        explicit_independent_probe_opt_in=False,
        orientation_source="sim_ground_truth",
    )
    assert result.eligible_for_independent_probe is False
    assert result.world_targets == {}
    assert result.loose_translation_xy_m is None
    assert "explicit_independent_probe_opt_in_missing" in (
        result.rejection_reasons
    )


@pytest.mark.parametrize(
    ("diagnostics", "reason"),
    (
        (
            _diagnostics(confidence=0.79),
            "loose_plug:confidence_below_gate",
        ),
        (
            _diagnostics(error_bound=0.010001),
            "loose_plug:xy_error_bound_exceeds_10mm",
        ),
        (
            _diagnostics(loose_timestamp=9.74),
            "loose_plug:stale",
        ),
        (
            _diagnostics(loose_timestamp=10.021),
            "loose_plug:timestamp_too_far_in_future",
        ),
        (
            _diagnostics(loose_timestamp=9.90, fixed_timestamp=9.96),
            "endpoint_timestamp_skew_exceeds_gate",
        ),
        (
            _diagnostics(loose_xy=(0.484, -0.210)),
            "loose_plug:outside_bounded_observation_domain",
        ),
        (
            _diagnostics(fixed_xy=(0.566, 0.185)),
            "fixed_receptacle:outside_bounded_reachability_screen",
        ),
        (
            _diagnostics(loose_xy=(0.151, -0.210)),
            "loose_plug:footprint_outside_table",
        ),
    ),
)
def test_runtime_gates_fail_closed_without_world_targets(
    diagnostics, reason
):
    result = _adapt(_sample(diagnostics=diagnostics))
    assert result.eligible_for_independent_probe is False
    assert result.world_targets == {}
    assert reason in result.rejection_reasons


def test_10mm_confidence_freshness_and_domain_boundaries_are_inclusive():
    diagnostics = _diagnostics(
        loose_xy=(0.485, -0.245),
        fixed_xy=(0.565, 0.200),
        loose_timestamp=9.75,
        fixed_timestamp=9.75,
        confidence=0.80,
        error_bound=0.010,
    )
    result = _adapt(_sample(diagnostics=diagnostics))
    assert result.eligible_for_independent_probe is True


def test_provider_provenance_truth_position_and_orientation_disclosure_gate():
    wrong_calibration = replace(_sample(), calibration_sha256="0" * 64)
    result = _adapt(wrong_calibration)
    assert "rgbd_calibration_provenance_mismatch" in result.rejection_reasons

    truth_position = replace(_sample(), uses_truth_position=True)
    result = _adapt(truth_position)
    assert "truth_position_for_translation_forbidden" in (
        result.rejection_reasons
    )

    wrong_orientation = replace(_sample(), uses_truth_orientation=False)
    result = _adapt(wrong_orientation)
    assert "orientation_source_disclosure_mismatch" in (
        result.rejection_reasons
    )


def test_pose_provider_purpose_and_partial_pose_claims_are_revalidated():
    with pytest.raises(ValueError, match="purpose differs"):
        _adapt(replace(_sample(), purpose=PoseProviderPurpose.CONTROL))
    with pytest.raises(ValueError, match="full-6D provider"):
        _adapt(replace(_sample(), full_6d=True))


def test_diagnostics_schema_is_exact_and_finite():
    extra = _diagnostics()
    extra["truth_xy"] = [0.52, -0.21]
    with pytest.raises(ValueError, match="keys differ"):
        _adapt(_sample(diagnostics=extra))

    invalid = _diagnostics()
    invalid["endpoints"]["loose_plug"]["confidence"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _adapt(_sample(diagnostics=invalid))


def _mutated_config(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "visual_xy_adapter.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda doc: doc.update(enabled_by_default=True),
            "disabled by default",
        ),
        (
            lambda doc: doc["gates"].update(
                maximum_xy_error_bound_m=0.011
            ),
            "cannot loosen",
        ),
        (
            lambda doc: doc["orientation_policy"].update(full_6d=True),
            "orientation scope",
        ),
        (
            lambda doc: doc["boundaries"].update(
                production_control_authorized=True
            ),
            "boundaries changed",
        ),
        (
            lambda doc: doc["inputs"]["multisite_rgbd_report"].update(
                sha256="0" * 64
            ),
            "SHA-256 mismatch",
        ),
    ),
)
def test_contract_rejects_gate_scope_or_provenance_drift(
    tmp_path, mutate, message
):
    path = _mutated_config(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_visual_xy_control_adapter_contract(
            path, repository=PROJECT_ROOT
        )
