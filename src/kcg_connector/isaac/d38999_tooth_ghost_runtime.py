#!/usr/bin/env python3

"""Render-only finger occlusion control for the prepared twist probe.

This helper is intentionally opt-in and owns exactly three anonymous-session
``visibility=invisible`` opinions, one on each finger's first-link root.  USD
visibility is inherited by render descendants but does not disable their
rigid bodies or colliders.  The caller creates this object before
``world.reset()`` and keeps it alive until the synchronized capture has been
finalized.  ``restore`` removes the three opinions and is safe to call again
from a runner ``finally`` block.

Isaac/PXR types are injected so importing this module remains CPU-only.  The
success manifest binds this source, the prepared-tooth runner, the visibility
sidecar, the synchronized capture manifest and both physics files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "kcg_d38999_tooth_ghost_runtime_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_tooth_ghost_manifest_v1"
CAPTURE_SCHEMA_VERSION = "kcg_d38999_tooth_sync_capture_v3"
EXPECTED_ROBOT_ROOT = "/World/HandArm"
EXPECTED_FINGER_NAMES = ("f1Link1", "f2Link1", "f3Link1")
EXPECTED_FINGER_ROOT_SUFFIXES = tuple(
    f"/{name}" for name in EXPECTED_FINGER_NAMES
)
VISIBILITY_PROPERTY = "visibility"
VISIBILITY_SIDECAR_NAME = "visibility_sidecar.json"
MANIFEST_NAME = "manifest.json"


def sha256_file(path):
    """Hash an evidence input without loading a large file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MODULE_SOURCE_PATH = Path(__file__).resolve()
_MODULE_SHA256_AT_IMPORT = sha256_file(_MODULE_SOURCE_PATH)


def prepare_empty_output_directory(path):
    """Create a fresh output directory or reject mixed-run evidence."""

    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"tooth ghost output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def file_binding(path):
    """Return an absolute size-and-SHA binding for one immutable file."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def physics_trace_sha256(summary_path):
    """Hash the same parent/tooth state columns used by the RGB analyzer."""

    digest = hashlib.sha256()
    with Path(summary_path).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        columns = [
            name
            for name in (reader.fieldnames or ())
            if name != "segment_contact_records"
        ]
        if not columns:
            raise RuntimeError("tooth physics summary has no columns")
        digest.update(("\x1f".join(columns) + "\n").encode("utf-8"))
        rows = 0
        for row in reader:
            digest.update(
                ("\x1f".join(row[name] for name in columns) + "\n").encode(
                    "utf-8"
                )
            )
            rows += 1
    if rows == 0:
        raise RuntimeError("tooth physics summary has no rows")
    return digest.hexdigest()


def contact_trace_sha256(summary_path):
    """Hash the phase-local per-step contact-record count separately."""

    columns = (
        "global_step",
        "phase",
        "phase_step",
        "segment_contact_records",
    )
    digest = hashlib.sha256()
    with Path(summary_path).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise RuntimeError("tooth contact summary columns are incomplete")
        digest.update(("\x1f".join(columns) + "\n").encode("utf-8"))
        rows = 0
        for row in reader:
            digest.update(
                ("\x1f".join(row[name] for name in columns) + "\n").encode(
                    "utf-8"
                )
            )
            rows += 1
    if rows == 0:
        raise RuntimeError("tooth contact summary has no rows")
    return digest.hexdigest()


def discover_finger_roots(stage, robot_root=EXPECTED_ROBOT_ROOT):
    """Find only the three rigid-link roots directly below handbase_link.

    The imported USD also contains nested visual prims named ``f1Link1`` etc.
    Requiring the immediate parent's name to be ``handbase_link`` prevents a
    visibility opinion from landing on a visual child or collision proxy.
    """

    found = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(str(robot_root) + "/"):
            continue
        name = str(prim.GetName())
        parent = prim.GetParent()
        parent_name = str(parent.GetName()) if parent else ""
        if name in EXPECTED_FINGER_NAMES and parent_name == "handbase_link":
            if name in found:
                raise RuntimeError(f"duplicate finger root: {name}")
            found[name] = prim
    if set(found) != set(EXPECTED_FINGER_NAMES):
        missing = sorted(set(EXPECTED_FINGER_NAMES) - set(found))
        raise RuntimeError(
            "prepared robot does not expose exactly three finger roots: "
            + ",".join(missing)
        )
    return tuple(found[name] for name in EXPECTED_FINGER_NAMES)


def _is_anonymous_layer(layer):
    """Support both Sdf's boolean and its stable ``anon:`` identifier."""

    if bool(getattr(layer, "anonymous", False)):
        return True
    return str(getattr(layer, "identifier", "")).startswith("anon:")


def _imageable_snapshot(stage, roots, UsdGeom):
    """Record effective visibility for every imageable finger descendant."""

    prefixes = tuple(str(prim.GetPath()) for prim in roots)
    result = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in prefixes
        ):
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        result[path] = str(imageable.ComputeVisibility())
    if not result or any(prefix not in result for prefix in prefixes):
        raise RuntimeError("finger visibility snapshot is incomplete")
    return result


def _session_property_spec(layer, Sdf, prim_path):
    property_path = Sdf.Path(str(prim_path)).AppendProperty(
        VISIBILITY_PROPERTY
    )
    return layer.GetPropertyAtPath(property_path)


class D38999ToothGhostRuntime:
    """Own and audit three render-only finger visibility opinions."""

    def __init__(
        self,
        *,
        stage,
        robot_root,
        output_directory,
        runner_source_path,
        Sdf,
        UsdGeom,
    ):
        if str(robot_root) != EXPECTED_ROBOT_ROOT:
            raise ValueError("tooth ghost robot root differs from contract")
        self._stage = stage
        self._Sdf = Sdf
        self._UsdGeom = UsdGeom
        self._robot_root = str(robot_root)
        self._output = prepare_empty_output_directory(output_directory)
        self._source_path = _MODULE_SOURCE_PATH
        self._source_at_import = _MODULE_SHA256_AT_IMPORT
        self._source_at_start = sha256_file(self._source_path)
        self._runner_source_path = Path(runner_source_path).resolve()
        self._runner_source_at_start = sha256_file(self._runner_source_path)
        self._session_layer = stage.GetSessionLayer()
        if not _is_anonymous_layer(self._session_layer):
            raise RuntimeError(
                "tooth ghost requires an anonymous session layer"
            )
        self._finger_roots = discover_finger_roots(stage, self._robot_root)
        self._finger_paths = tuple(
            str(prim.GetPath()) for prim in self._finger_roots
        )
        if not all(
            path.endswith(suffix)
            for path, suffix in zip(
                self._finger_paths, EXPECTED_FINGER_ROOT_SUFFIXES
            )
        ):
            raise RuntimeError("tooth ghost finger path order differs")
        for path in self._finger_paths:
            if _session_property_spec(self._session_layer, Sdf, path):
                raise RuntimeError(
                    f"pre-existing session visibility opinion: {path}"
                )
        self._before_visibility = _imageable_snapshot(
            stage, self._finger_roots, UsdGeom
        )
        invisible = str(UsdGeom.Tokens.invisible)
        if any(
            self._before_visibility[path] == invisible
            for path in self._finger_paths
        ):
            raise RuntimeError(
                "a finger root is already effectively invisible"
            )
        self._restored = False
        self._cleanup = None
        self._authored_imageable_count = len(self._before_visibility)
        try:
            self._author_visibility()
        except Exception:
            # A constructor exception cannot leave cleanup to the runner,
            # because assignment to its local owner has not completed yet.
            self._clear_visibility_opinions(require_full_restore=False)
            raise

    @property
    def restored(self):
        """Return whether the visibility opinions were fully removed."""

        return self._restored

    def _with_session_edit(self, callback):
        previous = self._stage.GetEditTarget()
        self._stage.SetEditTarget(self._session_layer)
        try:
            if self._stage.GetEditTarget().GetLayer() != self._session_layer:
                raise RuntimeError("failed to select the session edit target")
            return callback()
        finally:
            self._stage.SetEditTarget(previous)

    def _author_visibility(self):
        def author():
            for prim in self._finger_roots:
                imageable = self._UsdGeom.Imageable(prim)
                if not imageable:
                    raise RuntimeError("finger root is not UsdGeom.Imageable")
                imageable.CreateVisibilityAttr().Set(
                    self._UsdGeom.Tokens.invisible
                )

        self._with_session_edit(author)
        after = _imageable_snapshot(
            self._stage, self._finger_roots, self._UsdGeom
        )
        invisible = str(self._UsdGeom.Tokens.invisible)
        if set(after) != set(self._before_visibility) or any(
            value != invisible for value in after.values()
        ):
            raise RuntimeError(
                "not every finger render descendant became invisible"
            )
        for path in self._finger_paths:
            if not _session_property_spec(
                self._session_layer, self._Sdf, path
            ):
                raise RuntimeError(
                    f"session visibility opinion was not authored: {path}"
                )

    def _clear_visibility_opinions(self, *, require_full_restore):
        errors = []

        def clear():
            for prim in self._finger_roots:
                try:
                    # Remove the session AttributeSpec itself.  Attribute.Clear
                    # can leave a type-only spec behind, which would make a
                    # later run inherit stale presentation metadata.
                    prim.RemoveProperty(VISIBILITY_PROPERTY)
                except Exception as exception:  # pragma: no cover - PXR edge
                    errors.append(
                        f"{prim.GetPath()}:"
                        f"{type(exception).__name__}:{exception}"
                    )

        self._with_session_edit(clear)
        opinions_removed = all(
            not _session_property_spec(
                self._session_layer, self._Sdf, path
            )
            for path in self._finger_paths
        )
        after = _imageable_snapshot(
            self._stage, self._finger_roots, self._UsdGeom
        )
        restored = after == self._before_visibility
        self._restored = bool(opinions_removed and restored and not errors)
        self._cleanup = {
            "effective_visibility_restored_to_pre_author_state": restored,
            "errors": errors,
            "session_visibility_opinions_removed": opinions_removed,
        }
        if require_full_restore and not self._restored:
            raise RuntimeError(
                "tooth ghost visibility cleanup failed: "
                + json.dumps(self._cleanup, sort_keys=True)
            )
        return dict(self._cleanup)

    def restore(self):
        """Remove only this helper's opinions; safe for runner ``finally``."""

        if self._restored:
            return dict(self._cleanup)
        return self._clear_visibility_opinions(require_full_restore=True)

    def active_report(self):
        """Return the pre-play authoring contract without final evidence."""

        return {
            "schema_version": SCHEMA_VERSION,
            "active": not self._restored,
            "authoring": {
                "authored_before_timeline_play": True,
                "authored_before_world_reset": True,
                "computed_descendants_invisible": True,
                "edit_layer": "anonymous_session_layer",
                "imageable_descendant_count": self._authored_imageable_count,
                "prim_paths": list(self._finger_paths),
                "properties": [VISIBILITY_PROPERTY],
                "robot_root": self._robot_root,
                "visibility_token": "invisible",
            },
            "mutation_audit": {
                "collision_api_writes": 0,
                "material_writes": 0,
                "object_pose_writes": 0,
                "physics_api_writes": 0,
                "xform_writes": 0,
            },
        }

    def finalize(
        self,
        *,
        capture_manifest_path,
        physics_report_path,
        physics_summary_path,
    ):
        """Restore visibility and hash-bind the completed ghost evidence."""

        capture_path = Path(capture_manifest_path).expanduser().resolve()
        physics_report_path = Path(physics_report_path).expanduser().resolve()
        physics_summary_path = (
            Path(physics_summary_path).expanduser().resolve()
        )
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (
            capture.get("schema_version") != CAPTURE_SCHEMA_VERSION
            or capture.get("passed") is not True
            or capture.get("cleanup", {}).get("object_pose_writes") != 0
            or capture.get("capture_source", {}).get(
                "unchanged_during_capture"
            )
            is not True
        ):
            raise RuntimeError("ghost synchronized capture contract failed")
        physics_evidence = capture.get("physics_evidence", {})
        if (
            physics_evidence.get("report_sha256")
            != sha256_file(physics_report_path)
            or physics_evidence.get("summary_sha256")
            != sha256_file(physics_summary_path)
        ):
            raise RuntimeError(
                "ghost physics files differ from capture binding"
            )

        # Importing the pure contract here binds the exact contact definition
        # used to produce the fingerprint without adding PXR dependencies.
        from kcg_connector import d38999_tooth_occlusion_control as contract

        physics_report = json.loads(
            physics_report_path.read_text(encoding="utf-8")
        )
        contact_fingerprint = contract.contact_dynamics_fingerprint(
            physics_report
        )
        cleanup = self.restore()
        source_at_finalize = sha256_file(self._source_path)
        runner_at_finalize = sha256_file(self._runner_source_path)
        source_unchanged = bool(
            self._source_at_import
            == self._source_at_start
            == source_at_finalize
        )
        runner_unchanged = bool(
            self._runner_source_at_start == runner_at_finalize
        )
        report = {
            **self.active_report(),
            "active": False,
            "bindings": {
                "capture_manifest_sha256": sha256_file(capture_path),
                "contact_dynamics_sha256": contact_fingerprint,
                "physics_contact_trace_sha256": contact_trace_sha256(
                    physics_summary_path
                ),
                "physics_report_sha256": sha256_file(physics_report_path),
                "physics_state_trace_sha256": physics_trace_sha256(
                    physics_summary_path
                ),
                "physics_summary_sha256": sha256_file(
                    physics_summary_path
                ),
            },
            "cleanup": cleanup,
            "passed": bool(
                self._restored and source_unchanged and runner_unchanged
            ),
            "source": {
                "path": str(self._source_path),
                "runner_path": str(self._runner_source_path),
                "runner_sha256_at_finalize": runner_at_finalize,
                "runner_sha256_at_start": self._runner_source_at_start,
                "runner_unchanged_during_run": runner_unchanged,
                "sha256_at_finalize": source_at_finalize,
                "sha256_at_import": self._source_at_import,
                "sha256_at_start": self._source_at_start,
                "unchanged_during_run": source_unchanged,
            },
        }
        # Re-run the independent pure contract before emitting an authorized
        # sidecar; this catches a missing provenance or zero-write field even
        # if the runtime report construction changes later.
        contract.validate_runtime_sidecar(report)
        sidecar_path = self._output / VISIBILITY_SIDECAR_NAME
        sidecar_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        contract_source = Path(contract.__file__).resolve()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "HASH_SIZE_SCHEMA_BOUND",
            "inputs": {
                "capture_manifest": file_binding(capture_path),
                "physics_report": file_binding(physics_report_path),
                "physics_summary": file_binding(physics_summary_path),
            },
            "outputs": {
                "visibility_sidecar": file_binding(sidecar_path),
            },
            "sources": {
                "contact_fingerprint_contract": file_binding(
                    contract_source
                ),
                "prepared_tooth_runner": file_binding(
                    self._runner_source_path
                ),
                "runtime": file_binding(self._source_path),
            },
        }
        manifest_path = self._output / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if report["passed"] is not True:
            raise RuntimeError("tooth ghost runtime provenance changed")
        return {"manifest": manifest, "report": report}


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "D38999ToothGhostRuntime",
    "EXPECTED_FINGER_NAMES",
    "EXPECTED_FINGER_ROOT_SUFFIXES",
    "EXPECTED_ROBOT_ROOT",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "VISIBILITY_SIDECAR_NAME",
    "contact_trace_sha256",
    "discover_finger_roots",
    "file_binding",
    "physics_trace_sha256",
    "prepare_empty_output_directory",
    "sha256_file",
]
