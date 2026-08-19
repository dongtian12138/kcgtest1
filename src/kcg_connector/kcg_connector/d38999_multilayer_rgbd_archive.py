"""Evidence-bound RGB-D archive adapter for the D38999 multilayer model.

The adapter reuses the existing raw-formal capture and formal archive writer.
It adds source locking, strict no-overwrite behavior, preflight validation and
file digests.  The only writable entry point in this task is explicitly for
offline fixtures; it cannot claim a dynamic camera capture.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.postgrasp_shadow_estimator import (
    FormalArchiveError,
    FormalView,
    write_formal_archive,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_rgbd_archive_contract_v1"
PROVENANCE_SCHEMA_VERSION = "kcg_d38999_multilayer_rgbd_archive_provenance_v1"
VIEW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

FROZEN_SOURCES = {
    "src/kcg_connector/kcg_connector/isaac_d38999_rgbd_runtime.py": (
        "081708beda588703de4c3da752ad698f7185a25accefc714b73e15b3e3b6195a"
    ),
    "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py": (
        "b164a83fa5039d39664cbfa6ac4bca1ca7976158f019f3fe174de71afaaa6a8e"
    ),
    "src/kcg_connector/test/test_postgrasp_raw_formal_capture.py": (
        "68a31608e4cabd9efcaa79515fbe838a2e39803cf0b3538da6d70568bb0b2809"
    ),
    "src/kcg_connector/test/test_postgrasp_shadow_estimator.py": (
        "8b0ef246c05baa4daa89b6e5f6c24631a2d023882aa2200d89aab83c59763b0e"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C1-PALM-CAMERA-INTERFACE/"
    "INTERFACE_MANIFEST.json": (
        "b1d5a8938a918898302a7b4790025dc94a97abbdb844fb020db195a0b160d990"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C2-WRIST-CAMERA-INTERFACE/"
    "INTERFACE_MANIFEST.json": (
        "09b4369e08a2556fbc35fb0e1ec4d07f7376a80982bbfb049f5eb5c481fe506f"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
    "VIEW_PLAN_MANIFEST.json": (
        "64eecf1a5e1ac04fd129453b042507e25ab2fbfe170fc035e7c00bcaba23921e"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verified_sources(root: Path) -> tuple[dict[str, str], ...]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen RGB-D archive source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"frozen RGB-D archive source hash mismatch: {relative}"
            )
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_rgbd_archive_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build the current offline archive contract without writing an archive."""

    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    palm = _json_mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C1-PALM-CAMERA-INTERFACE/INTERFACE_MANIFEST.json",
        "Palm interface manifest",
    )
    wrist = _json_mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C2-WRIST-CAMERA-INTERFACE/INTERFACE_MANIFEST.json",
        "wrist interface manifest",
    )
    view_plan = _json_mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/VIEW_PLAN_MANIFEST.json",
        "postgrasp view plan manifest",
    )
    runtime_source = (
        root
        / "src/kcg_connector/kcg_connector/isaac_d38999_rgbd_runtime.py"
    ).read_text(encoding="utf-8")
    archive_source = (
        root
        / "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py"
    ).read_text(encoding="utf-8")

    if (
        palm.get("status") != "OFFLINE_PASS"
        or palm.get("dynamic_camera_pass_claimed") is not False
        or wrist.get("status") != "OFFLINE_PASS"
        or wrist.get("dynamic_camera_pass_claimed") is not False
        or view_plan.get("status") != "STATIC_PASS"
        or view_plan.get("dynamic_status") != "PARKED"
        or view_plan.get("dynamic_readiness", {}).get(
            "current_multilayer_dynamic_views_proven"
        )
        != 0
    ):
        raise ValueError("upstream camera/view evidence was promoted or changed")
    if palm.get("observation_target") != wrist.get("observation_target"):
        raise ValueError("Palm and wrist multilayer targets differ")
    if (
        view_plan.get("camera_binding", {}).get("observation_target")
        != palm.get("observation_target")
    ):
        raise ValueError("C3 and camera multilayer targets differ")

    raw_signature = inspect.signature(
        __import__(
            "kcg_connector.isaac_d38999_rgbd_runtime",
            fromlist=["capture_d38999_rgbd_raw_formal"],
        ).capture_d38999_rgbd_raw_formal
    )
    if any(
        name in raw_signature.parameters
        for name in (
            "semantic",
            "object_pose",
            "loose_prim",
            "fixed_prim",
            "contact_report",
        )
    ):
        raise ValueError("raw formal capture signature admits truth inputs")
    for token in (
        '"semantic_annotator_used": False',
        '"endpoint_truth_read": False',
        'get_annotator("rgb")',
        '"distance_to_image_plane"',
    ):
        if token not in runtime_source:
            raise ValueError(f"raw formal capture token missing: {token}")
    for token in (
        "output.mkdir(parents=True, exist_ok=False)",
        '"formal_observation"',
        '"rgb.png"',
        '"depth.npy"',
        '"camera.json"',
        '"fk.json"',
        '"semantic"',
        '"object_truth"',
        '"contact_report"',
    ):
        if token not in archive_source:
            raise ValueError(f"formal archive token missing: {token}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "EVIDENCE_BOUND_RGBD_ARCHIVE_INTERFACE",
        "observation_target": dict(palm["observation_target"]),
        "camera_interfaces": {
            "Palm": {
                "role": palm["interface_role"],
                "camera_prim": palm["camera_prim"],
                "dynamic_capture_passed": False,
            },
            "Wrist": {
                "role": wrist["interface_role"],
                "camera_prim": wrist["camera_prim"],
                "dynamic_capture_passed": False,
            },
        },
        "raw_capture": {
            "function": "capture_d38999_rgbd_raw_formal",
            "channels_exactly": ["rgb", "distance_to_image_plane"],
            "semantic_annotator_used": False,
            "endpoint_truth_read": False,
            "camera_world_pose_read": False,
        },
        "archive_format": {
            "writer": "write_formal_archive",
            "root_manifest": "formal_manifest.json",
            "per_view_files_exactly": [
                "rgb.png",
                "depth.npy",
                "camera.json",
                "fk.json",
            ],
            "provenance_file": "archive_provenance.json",
            "existing_output_policy": "REFUSE_OVERWRITE",
            "file_sha256_required": True,
            "inf_background_depth_preserved": True,
        },
        "truth_firewall": {
            "forbidden_inputs": [
                "semantic",
                "id_to_labels",
                "registered_truth_xy_m",
                "object_truth",
                "contact_report",
                "collider_identity",
                "contact_normal",
            ],
            "object_pose_truth_allowed": False,
            "postrun_object_pose_write_allowed": False,
        },
        "current_readiness": {
            "offline_fixture_archive_allowed": True,
            "dynamic_capture_archives_available": 0,
            "dynamic_archive_write_authorized": False,
            "formal_dynamic_rgbd_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "render_capture_performed": False,
        "dynamic_rgbd_pass_claimed": False,
        "hardware_authorized": False,
    }


def _valid_timestamp(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _preflight_views(views: Sequence[FormalView]) -> tuple[FormalView, ...]:
    frozen = tuple(views)
    if not 1 <= len(frozen) <= 3:
        raise FormalArchiveError("offline archive requires one to three views")
    ids = []
    for view in frozen:
        if not isinstance(view, FormalView):
            raise FormalArchiveError("archive inputs must be FormalView values")
        if (
            VIEW_ID_PATTERN.fullmatch(view.view_id) is None
            or view.view_id in {"posthoc_semantic", "truth_restore"}
        ):
            raise FormalArchiveError("unsafe or reserved formal view id")
        ids.append(view.view_id)
        if not _valid_timestamp(view.timestamp_utc):
            raise FormalArchiveError("view timestamp must be timezone-aware ISO-8601")
        rgb = np.asarray(view.rgb)
        depth = np.asarray(view.depth)
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.shape != rgb.shape[:2]
            or np.isnan(depth).any()
            or np.any(np.isfinite(depth) & (depth < 0.0))
            or not np.any(np.isfinite(depth) & (depth > 0.0))
        ):
            raise FormalArchiveError("invalid offline RGB or depth array")
        if (
            int(view.camera.width) != rgb.shape[1]
            or int(view.camera.height) != rgb.shape[0]
        ):
            raise FormalArchiveError("camera resolution differs from RGB-D")
        for name, matrix in (
            ("T_WH", view.T_WH),
            ("T_WC", view.T_WC),
            ("T_HC", view.T_HC),
        ):
            if matrix is None and name == "T_HC":
                continue
            array = np.asarray(matrix, dtype=np.float64)
            if array.shape != (4, 4) or not np.all(np.isfinite(array)):
                raise FormalArchiveError(f"{name} must be finite 4x4")
    if len(set(ids)) != len(ids):
        raise FormalArchiveError("formal view ids must be unique")
    return frozen


def write_offline_multilayer_rgbd_archive(
    repository_root: str | Path,
    output_dir: str | Path,
    views: Sequence[FormalView],
    *,
    fixture_id: str,
) -> dict[str, Any]:
    """Write a digest-complete archive explicitly labelled OFFLINE_FIXTURE."""

    contract = build_multilayer_rgbd_archive_contract(repository_root)
    if VIEW_ID_PATTERN.fullmatch(fixture_id) is None:
        raise FormalArchiveError("fixture_id contains unsafe characters")
    frozen_views = _preflight_views(views)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"RGB-D archive output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-", dir=output.parent
    ) as temporary:
        archive = Path(temporary) / "archive"
        manifest = write_formal_archive(archive, frozen_views)
        expected = {"formal_manifest.json"}
        for view in frozen_views:
            expected.update(
                {
                    f"{view.view_id}/rgb.png",
                    f"{view.view_id}/depth.npy",
                    f"{view.view_id}/camera.json",
                    f"{view.view_id}/fk.json",
                }
            )
        actual = {
            str(path.relative_to(archive))
            for path in archive.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise FormalArchiveError("formal archive file set differs")
        if (
            manifest.get("role") != "formal_observation"
            or not {
                "semantic",
                "object_truth",
                "contact_report",
            }.issubset(set(manifest.get("forbidden_inputs", [])))
        ):
            raise FormalArchiveError("formal archive truth firewall differs")
        files = [
            {
                "path": relative,
                "sha256": _sha256(archive / relative),
                "size_bytes": (archive / relative).stat().st_size,
            }
            for relative in sorted(expected)
        ]
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "evidence_level": "OFFLINE_FIXTURE",
            "fixture_id": fixture_id,
            "view_count": len(frozen_views),
            "view_ids": [view.view_id for view in frozen_views],
            "files": files,
            "source_hashes": contract["sources"],
            "semantic_input_used": False,
            "object_pose_truth_used": False,
            "contact_truth_used": False,
            "dynamic_capture_claimed": False,
            "formal_dynamic_rgbd_pass_claimed": False,
            "hardware_authorized": False,
        }
        (archive / "archive_provenance.json").write_text(
            json.dumps(
                provenance,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(
                f"RGB-D archive output appeared during write: {output}"
            )
        archive.rename(output)
    return provenance


__all__ = [
    "FROZEN_SOURCES",
    "PROVENANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_multilayer_rgbd_archive_contract",
    "write_offline_multilayer_rgbd_archive",
]
