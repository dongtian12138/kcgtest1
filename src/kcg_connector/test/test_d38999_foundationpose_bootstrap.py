"""Pure tests for the disabled FoundationPose asset/bootstrap boundary."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_foundationpose_bootstrap import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    build_proxy_mesh_documents,
    evaluate_foundationpose_readiness,
    load_foundationpose_bootstrap_contract,
    validate_obj_document,
)
from kcg_connector.d38999_proxy import load_d38999_shell25j_proxy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "d38999_foundationpose_bootstrap.py"
)
E2E_RUNNER_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)
READINESS_CLI_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_foundationpose_readiness.py"
)


def _contract():
    return load_foundationpose_bootstrap_contract(DEFAULT_CONFIG_PATH)


def _write_mutated(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    output = tmp_path / "mutated.yaml"
    output.write_text(yaml.safe_dump(document), encoding="utf-8")
    return output


def test_contract_is_disabled_content_addressed_and_local_only():
    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled is False
    assert contract.model_version == "1.0.1_onnx"
    assert contract.model_storage_scope == "local_gitignored_artifact_only"
    assert set(contract.models) == {"refine_model", "score_model"}
    assert set(contract.meshes) == {
        "loose_body",
        "coupling_nut",
        "fixed_receptacle",
    }
    assert all(
        str(item.path).startswith("artifacts/")
        for item in contract.models.values()
    )
    assert all(value is False for value in contract.boundaries.values())
    assert "artifacts/" in (
        PROJECT_ROOT / ".gitignore"
    ).read_text(encoding="utf-8").splitlines()


def test_module_has_no_isaac_ros_tensorrt_or_gpu_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {
            "isaacsim",
            "omni",
            "pxr",
            "rclpy",
            "torch",
            "tensorrt",
            "onnx",
            "onnxruntime",
            "cv2",
            "open3d",
        }
    )


def test_bootstrap_is_not_imported_by_current_e2e_runner():
    source = E2E_RUNNER_PATH.read_text(encoding="utf-8")
    assert "d38999_foundationpose_bootstrap" not in source
    assert "d38999_foundationpose_bootstrap_v1.yaml" not in source
    assert READINESS_CLI_PATH.is_file()


def test_generated_obj_documents_are_deterministic_and_hash_pinned():
    contract = _contract()
    proxy = load_d38999_shell25j_proxy(
        PROJECT_ROOT / contract.inputs["proxy_config"].path
    )
    first = build_proxy_mesh_documents(
        proxy, radial_sections=contract.radial_sections
    )
    second = build_proxy_mesh_documents(
        proxy, radial_sections=contract.radial_sections
    )
    assert first == second
    assert set(first) == set(contract.meshes)
    for mesh_id, document in first.items():
        assert hashlib.sha256(document).hexdigest() == (
            contract.meshes[mesh_id].file.sha256
        )
        stats = validate_obj_document(document)
        assert stats.vertex_count >= 192
        assert stats.triangle_count >= 288
        assert stats.bounds_min_xyz_m[0] < 0.0
        assert stats.bounds_max_xyz_m[0] > 0.0


def test_nut_is_a_separate_mesh_and_all_yaw_claims_remain_blocked():
    contract = _contract()
    nut = contract.meshes["coupling_nut"]
    loose = contract.meshes["loose_body"]
    fixed = contract.meshes["fixed_receptacle"]
    assert nut.asset_prim_path.endswith("/LoosePlug/CouplingNut")
    assert loose.asset_prim_path.endswith("/LoosePlug/BodyAssembly")
    assert fixed.asset_prim_path.endswith("/FixedReceptacle")
    assert nut.rotational_symmetry_order == 24
    assert loose.rotational_symmetry_order == 2
    assert fixed.rotational_symmetry_order == 2
    assert all(
        not item.unique_polarization_key_present
        and not item.control_orientation_qualified
        for item in contract.meshes.values()
    )
    assert contract.observability[
        "equivalent_yaw_period_rad"
    ] == pytest.approx(3.141592653589793)


def test_current_local_artifact_report_never_claims_inference_or_control():
    # The large NVIDIA files are intentionally not a package dependency.  This
    # test accepts either missing or verified local artifacts, but never a pose
    # or control claim based on file presence alone.
    report = evaluate_foundationpose_readiness(_contract(), PROJECT_ROOT)
    json.dumps(report, allow_nan=False)
    assert report["gates"]["foundationpose_inference_ready"] is False
    assert report["gates"]["full_6d_keyed_pose_observable"] is False
    assert report["gates"]["vision_control_authorized"] is False
    assert report["claims"]["foundationpose_inference_performed"] is False
    assert report["claims"]["tensorrt_engine_build_performed"] is False
    assert (
        "proxy_unique_polarization_key_geometry_absent"
        in report["blockers"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document: document.update({"enabled": True}),
            "must remain disabled",
        ),
        (
            lambda document: document["model_bundle"].update(
                {"standalone_redistribution_allowed": True}
            ),
            "stand-alone model redistribution",
        ),
        (
            lambda document: document["model_bundle"].update(
                {"commit_or_package_allowed": True}
            ),
            "never be committed",
        ),
        (
            lambda document: document["mesh_bundle"]["objects"][
                "loose_body"
            ].update({"unique_polarization_key_present": True}),
            "cannot claim unique keyed orientation",
        ),
        (
            lambda document: document["mesh_bundle"]["objects"][
                "loose_body"
            ].update({"sha256": "0" * 64}),
            "non-zero lowercase SHA-256",
        ),
        (
            lambda document: document["boundaries"].update(
                {"foundationpose_inference_performed": True}
            ),
            "must be false",
        ),
    ),
)
def test_loader_fails_closed_when_claim_or_license_boundaries_are_weakened(
    tmp_path, mutate, message
):
    path = _write_mutated(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_foundationpose_bootstrap_contract(path)


def test_obj_validator_rejects_unsupported_or_invalid_geometry():
    with pytest.raises(ValueError, match="unsupported OBJ"):
        validate_obj_document(b"o x\nv 0 0 0\nl 1 1\n")
    with pytest.raises(ValueError, match="face index"):
        validate_obj_document(
            b"o x\nv 0 0 0\nv 1 0 0\nv 0 1 1\nf 1 2 4\n"
        )
