"""Fail-closed authorization for the one TASK-R12-006B local candidate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .d38999_keyed_v2_physical_model_contract import (
    PhysicalModelContract,
    WORKSPACE_ROOT,
)


TASK_ID = "TASK-R12-006B"
SOURCE_ASSET_REL = Path(
    "artifacts/kcg_connector/isaac/keyed_v3_physical_r12/candidates/"
    "r12_candidate_02/r12_candidate_02.usda"
)
SOURCE_SCENE_REL = SOURCE_ASSET_REL.parent / "scene.yaml"
CANDIDATE_ASSET_REL = Path(
    "artifacts/agent_control/tasks/TASK-R12-006B/candidate/"
    "task_r12_006b_local_candidate_01.usda"
)
CANDIDATE_SCENE_REL = CANDIDATE_ASSET_REL.parent / "scene.yaml"
BUILD_RESULT_REL = CANDIDATE_ASSET_REL.parent / "CANDIDATE_BUILD_RESULT.json"
EXPECTED_SOURCE_ASSET_SHA256 = (
    "5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995"
)
EXPECTED_MODIFIED_PRIMS = 2928
EXPECTED_MODIFIED_POINTS = 8784
EXPECTED_UNCHANGED_PROPERTIES = 618585
EXPECTED_WORLD_POINT_DELTA_M = 2.0e-5
WORLD_POINT_DELTA_TOLERANCE_M = 3.0e-8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class R12LocalCandidateAuthorization:
    model: PhysicalModelContract
    result_path: Path
    result_sha256: str
    source_asset: Path
    source_asset_sha256: str
    candidate_asset: Path
    candidate_asset_relative_path: str
    candidate_asset_sha256: str
    candidate_scene: Path
    candidate_scene_sha256: str

    def evidence(self) -> dict[str, Any]:
        return {
            "scope": "TASK-R12-006B_LOCAL_CANDIDATE_ONLY",
            "result_path": str(self.result_path),
            "result_sha256": self.result_sha256,
            "source_asset": str(self.source_asset),
            "source_asset_sha256": self.source_asset_sha256,
            "candidate_asset": str(self.candidate_asset),
            "candidate_asset_relative_path": self.candidate_asset_relative_path,
            "candidate_asset_sha256": self.candidate_asset_sha256,
            "candidate_scene": str(self.candidate_scene),
            "candidate_scene_sha256": self.candidate_scene_sha256,
            "candidate_number": 1,
            "candidate_limit": 1,
            "structural_parameter_count": 1,
            "structural_parameter": "socket_entry_F_diameter_mm",
            "structural_parameter_before_mm": 1.28,
            "structural_parameter_after_mm": 1.32,
            "formal_contract_file_modified": False,
            "in_memory_identity_output_path_override_only": True,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(repository: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty repository-relative text")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay below the repository")
    resolved = (repository / relative).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return resolved, str(relative)


def _require_close(actual: Any, expected: float, label: str, tolerance: float) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} is not numeric")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} changed: {actual}")


def authorize_task_r12_006b_local_candidate(
    *,
    model: PhysicalModelContract,
    result_path: Path | str,
    expected_result_sha256: str,
    scene_config: Path | str,
    repository_root: Path | str = WORKSPACE_ROOT,
) -> R12LocalCandidateAuthorization:
    """Authorize exactly one pinned local candidate without changing the frozen YAML."""

    repository = Path(repository_root).expanduser().resolve()
    result = Path(result_path).expanduser().resolve()
    scene = Path(scene_config).expanduser().resolve()
    if not _SHA256_RE.fullmatch(expected_result_sha256):
        raise ValueError("local candidate result SHA-256 is malformed")
    expected_result = (repository / BUILD_RESULT_REL).resolve()
    if result != expected_result or not result.is_file():
        raise ValueError("local candidate result path is not the TASK-R12-006B result")
    result_sha256 = _sha256(result)
    if result_sha256 != expected_result_sha256:
        raise ValueError("local candidate result SHA-256 differs from the command pin")
    document = json.loads(result.read_text(encoding="utf-8"))

    exact_values = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "candidate_number": 1,
        "candidate_limit": 1,
        "structural_parameter_count": 1,
        "modified_prim_count": EXPECTED_MODIFIED_PRIMS,
        "modified_point_count": EXPECTED_MODIFIED_POINTS,
        "expected_modified_prim_count": EXPECTED_MODIFIED_PRIMS,
        "expected_modified_point_count": EXPECTED_MODIFIED_POINTS,
        "compared_unchanged_property_count": EXPECTED_UNCHANGED_PROPERTIES,
        "passed": True,
        "formal_contract_modified": False,
        "mass_material_friction_elasticity_modified": False,
        "source_candidate_modified": False,
    }
    for key, expected in exact_values.items():
        if document.get(key) != expected:
            raise ValueError(f"local candidate result field changed: {key}")
    if document.get("argv") != [
        "tools/agent_control/build_r12_006b_candidate.py",
        "--run",
        "--socket-entry-f-diameter-mm",
        "1.32",
    ]:
        raise ValueError("local candidate builder argv changed")

    parameter = document.get("structural_parameter")
    if not isinstance(parameter, Mapping):
        raise ValueError("local candidate structural parameter is missing")
    if parameter.get("name") != "socket_entry_F_diameter_mm":
        raise ValueError("local candidate structural parameter name changed")
    _require_close(parameter.get("before_mm"), 1.28, "parameter.before_mm", 1.0e-12)
    _require_close(parameter.get("after_mm"), 1.32, "parameter.after_mm", 1.0e-12)
    if parameter.get("authorized_range_mm") != [1.24, 1.32]:
        raise ValueError("local candidate authorized diameter range changed")
    _require_close(
        parameter.get("derived_inner_radius_before_m"),
        0.000640,
        "parameter.derived_inner_radius_before_m",
        1.0e-12,
    )
    _require_close(
        parameter.get("derived_inner_radius_after_m"),
        0.000660,
        "parameter.derived_inner_radius_after_m",
        1.0e-12,
    )

    point_audit = document.get("point_geometry_audit")
    if not isinstance(point_audit, Mapping):
        raise ValueError("local candidate point geometry audit is missing")
    _require_close(
        point_audit.get("stage_meters_per_unit"),
        1.0,
        "point_audit.stage_meters_per_unit",
        1.0e-12,
    )
    _require_close(
        point_audit.get("target_point_scale"),
        0.001,
        "point_audit.target_point_scale",
        1.0e-10,
    )
    for key in ("world_delta_m_min", "world_delta_m_max"):
        _require_close(
            point_audit.get(key),
            EXPECTED_WORLD_POINT_DELTA_M,
            f"point_audit.{key}",
            WORLD_POINT_DELTA_TOLERANCE_M,
        )
    if float(point_audit["world_delta_m_min"]) > float(
        point_audit["world_delta_m_max"]
    ):
        raise ValueError("local candidate point delta range is inverted")
    if document.get("source_inventory") != document.get("candidate_inventory"):
        raise ValueError("local candidate prim/type/family inventory changed")

    source_asset, source_relative = _resolve_relative(
        repository, document.get("source_asset"), "source_asset"
    )
    candidate_asset, candidate_relative = _resolve_relative(
        repository, document.get("candidate_asset"), "candidate_asset"
    )
    source_scene, source_scene_relative = _resolve_relative(
        repository, document.get("source_scene"), "source_scene"
    )
    candidate_scene, candidate_scene_relative = _resolve_relative(
        repository, document.get("candidate_scene"), "candidate_scene"
    )
    if source_relative != str(SOURCE_ASSET_REL):
        raise ValueError("local candidate source asset is not frozen candidate2")
    if candidate_relative != str(CANDIDATE_ASSET_REL):
        raise ValueError("local candidate output path changed")
    if source_scene_relative != str(SOURCE_SCENE_REL):
        raise ValueError("local candidate source scene changed")
    if candidate_scene_relative != str(CANDIDATE_SCENE_REL) or scene != candidate_scene:
        raise ValueError("formal P1 scene is not the authorized local candidate scene")
    for path in (source_asset, candidate_asset, source_scene, candidate_scene):
        if not path.is_file():
            raise FileNotFoundError(f"authorized local candidate input is missing: {path}")

    source_sha256 = _sha256(source_asset)
    candidate_sha256 = _sha256(candidate_asset)
    source_scene_sha256 = _sha256(source_scene)
    candidate_scene_sha256 = _sha256(candidate_scene)
    if not (
        source_sha256
        == document.get("source_asset_sha256_before")
        == document.get("source_asset_sha256_after")
        == EXPECTED_SOURCE_ASSET_SHA256
    ):
        raise ValueError("frozen candidate2 asset SHA-256 changed")
    if candidate_sha256 != document.get("candidate_asset_sha256"):
        raise ValueError("local candidate asset SHA-256 changed")
    if source_scene_sha256 != document.get("source_scene_sha256"):
        raise ValueError("frozen candidate2 scene SHA-256 changed")
    if candidate_scene_sha256 != document.get("candidate_scene_sha256"):
        raise ValueError("local candidate scene SHA-256 changed")

    source_text = source_scene.read_text(encoding="utf-8")
    expected_scene_text = source_text.replace(
        f"  local_path: {SOURCE_ASSET_REL}",
        f"  local_path: {CANDIDATE_ASSET_REL}",
        1,
    )
    if expected_scene_text == source_text or candidate_scene.read_text(
        encoding="utf-8"
    ) != expected_scene_text:
        raise ValueError("local candidate scene differs by more than its asset path")

    identity = model.document["identity"]
    model_source = Path(str(identity["recommended_asset_directory"])) / str(
        identity["recommended_asset_name"]
    )
    if model_source != SOURCE_ASSET_REL:
        raise ValueError("input model is not the frozen candidate2 model view")
    local_document = deepcopy(dict(model.document))
    local_identity = local_document["identity"]
    local_identity["recommended_asset_directory"] = str(CANDIDATE_ASSET_REL.parent)
    local_identity["recommended_asset_name"] = CANDIDATE_ASSET_REL.name
    local_model = PhysicalModelContract(path=model.path, document=local_document)

    return R12LocalCandidateAuthorization(
        model=local_model,
        result_path=result,
        result_sha256=result_sha256,
        source_asset=source_asset,
        source_asset_sha256=source_sha256,
        candidate_asset=candidate_asset,
        candidate_asset_relative_path=candidate_relative,
        candidate_asset_sha256=candidate_sha256,
        candidate_scene=candidate_scene,
        candidate_scene_sha256=candidate_scene_sha256,
    )


__all__ = [
    "R12LocalCandidateAuthorization",
    "authorize_task_r12_006b_local_candidate",
]
