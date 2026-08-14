#!/usr/bin/env python3

"""Opt-in, read-mostly diagnostics for one-tooth D38999 nut jitter.

The probe observes all 24 CouplingNut segment transforms at the physics rate.
Its normal CSV is deliberately one row per step; a full 24-segment snapshot is
written only when a transform threshold is exceeded.  The optional Segment_00
normalization and display colors are authored only in the anonymous stage's
session layer before physics starts, so the checked-in USDA remains untouched.

This module intentionally has no Isaac Sim imports at module scope.  Smoke
scripts inject their already-loaded pxr types and PhysX contact reporter.
"""

from __future__ import annotations

import colorsys
import csv
import json
import math
from pathlib import Path
import re


SCHEMA_VERSION = "kcg_d38999_nut_tooth_jitter_probe_v1"
SEGMENT_COUNT = 24
TRANSLATION_THRESHOLD_M = 1.0e-6
ROTATION_THRESHOLD_RAD = 1.0e-5
_SEGMENT_RE = re.compile(r"(?:^|/)Segment_(\d{2})(?:$|/)")


def deterministic_segment_colors(count=SEGMENT_COUNT):
    """Return stable, visually distinct HSV-wheel colors keyed by segment."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("segment color count must be a positive integer")
    result = {}
    for index in range(count):
        # Fixed saturation/value keep IDs readable without making Segment_00
        # special; its identity comes only from the number and hue position.
        rgb = colorsys.hsv_to_rgb(float(index) / float(count), 0.72, 0.95)
        result[f"Segment_{index:02d}"] = [round(value, 6) for value in rgb]
    return result


def segment_index_from_path(path, count=SEGMENT_COUNT):
    """Extract an in-range CouplingNut segment index from a collider path."""

    match = _SEGMENT_RE.search(str(path))
    if match is None:
        return None
    index = int(match.group(1))
    return index if index < count else None


def summarize_segment00_schema(op_names, op_values=None):
    """Create the pure-data portion of the Segment_00 xform schema report."""

    names = [str(value) for value in op_names]
    rotate_names = [name for name in names if name.endswith("xformOp:rotateZ")]
    values = dict(op_values or {})
    rotate_value = values.get(rotate_names[0]) if rotate_names else None
    return {
        "explicit_rotate_z": bool(rotate_names),
        "explicit_rotate_z_degrees": (
            float(rotate_value) if rotate_value is not None else None
        ),
        "op_names": names,
        "schema_outlier": not bool(rotate_names),
    }


def _session_edit(stage, callback):
    previous = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        return callback()
    finally:
        stage.SetEditTarget(previous)


def normalize_segment00_rotate_z_session(stage, nut_root, UsdGeom):
    """Add an explicit zero rotateZ op in the session layer, before play.

    The op order is normalized to translate, rotateZ, scale when those three
    canonical ops are present.  No referenced or on-disk layer is edited.
    """

    path = f"{nut_root}/Segment_00"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing CouplingNut diagnostic prim: {path}")

    def author():
        xformable = UsdGeom.Xformable(prim)
        ordered = list(xformable.GetOrderedXformOps())
        rotate_ops = [
            op
            for op in ordered
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ
        ]
        changed = not bool(rotate_ops)
        rotate_op = (
            rotate_ops[0] if rotate_ops else xformable.AddRotateZOp()
        )
        rotate_op.Set(0.0)
        all_ops = list(xformable.GetOrderedXformOps())
        rank = {
            UsdGeom.XformOp.TypeTranslate: 0,
            UsdGeom.XformOp.TypeRotateZ: 1,
            UsdGeom.XformOp.TypeScale: 2,
        }
        # Stable sorting preserves any unknown ops while putting the canonical
        # three in the same order used by Segment_01..Segment_23.
        normalized = sorted(
            enumerate(all_ops),
            key=lambda item: (rank.get(item[1].GetOpType(), 3), item[0]),
        )
        xformable.SetXformOpOrder(
            [item[1] for item in normalized],
            xformable.GetResetXformStack(),
        )
        return {
            "authored_in_session_layer": True,
            "changed_missing_rotate_z": changed,
            "path": path,
            "rotate_z_degrees": 0.0,
        }

    return _session_edit(stage, author)


def colorize_segments_session(stage, nut_root, UsdGeom, Gf):
    """Author deterministic displayColor IDs in the session layer only."""

    colors = deterministic_segment_colors()

    def author():
        for name, rgb in colors.items():
            path = f"{nut_root}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"missing CouplingNut color prim: {path}")
            primvar = UsdGeom.Gprim(prim).CreateDisplayColorPrimvar(
                UsdGeom.Tokens.constant
            )
            primvar.Set([Gf.Vec3f(*rgb)])
        return {
            "authored_in_session_layer": True,
            "colors_rgb": colors,
            "note": "bound render materials may override displayColor",
        }

    return _session_edit(stage, author)


def _quat_error(Gf, first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


def _matrix_error(Gf, first, second):
    first_transform = Gf.Transform(first)
    second_transform = Gf.Transform(second)
    translation = float(
        (second_transform.GetTranslation() - first_transform.GetTranslation())
        .GetLength()
    )
    rotation = _quat_error(
        Gf,
        first_transform.GetRotation().GetQuat(),
        second_transform.GetRotation().GetQuat(),
    )
    return translation, rotation


def _json_dump_line(stream, value):
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


class NutToothJitterProbe:
    """Sample USD transforms, parent PhysX state and per-tooth contacts."""

    _CSV_FIELDS = (
        "global_step",
        "phase",
        "phase_step",
        "maximum_local_translation_error_m",
        "maximum_local_rotation_error_rad",
        "maximum_parent_relative_translation_error_m",
        "maximum_parent_relative_rotation_error_rad",
        "maximum_segment",
        "anomalous",
        "parent_px_m",
        "parent_py_m",
        "parent_pz_m",
        "parent_qw",
        "parent_qx",
        "parent_qy",
        "parent_qz",
        "parent_linear_speed_m_s",
        "parent_angular_speed_rad_s",
        "segment_contact_records",
    )

    def __init__(
        self,
        *,
        stage,
        nut_root,
        parent_rigid,
        output_directory,
        Gf,
        Usd,
        UsdGeom,
        PhysicsSchemaTools,
        normalization_report=None,
        color_report=None,
    ):
        self._stage = stage
        self._nut_root = str(nut_root)
        self._parent_rigid = parent_rigid
        self._Gf = Gf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._PhysicsSchemaTools = PhysicsSchemaTools
        self._finalized = False
        self._steps = 0
        self._anomaly_steps = 0
        self._phase_steps = {}
        self._normalization_report = normalization_report or {
            "authored_in_session_layer": False,
            "changed_missing_rotate_z": False,
        }
        self._color_report = color_report or {
            "authored_in_session_layer": False,
            "colors_rgb": deterministic_segment_colors(),
        }

        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        self._output = output
        self._summary_stream = (output / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._summary_writer = csv.DictWriter(
            self._summary_stream, fieldnames=self._CSV_FIELDS
        )
        self._summary_writer.writeheader()
        self._anomaly_stream = (output / "anomalies.jsonl").open(
            "w", encoding="utf-8"
        )

        parent_prim = stage.GetPrimAtPath(self._nut_root)
        if not parent_prim or not parent_prim.IsValid():
            raise RuntimeError(f"missing CouplingNut parent: {self._nut_root}")
        self._parent_prim = parent_prim
        self._segments = []
        self._aggregate = {}
        for index in range(SEGMENT_COUNT):
            name = f"Segment_{index:02d}"
            path = f"{self._nut_root}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"missing CouplingNut segment: {path}")
            xformable = UsdGeom.Xformable(prim)
            local = xformable.GetLocalTransformation(Usd.TimeCode.Default())
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            relative, _ = cache.ComputeRelativeTransform(prim, parent_prim)
            ops = list(xformable.GetOrderedXformOps())
            op_values = {}
            for op in ops:
                value = op.Get(Usd.TimeCode.Default())
                if isinstance(value, (float, int)):
                    op_values[str(op.GetOpName())] = float(value)
            schema = summarize_segment00_schema(
                [str(op.GetOpName()) for op in ops], op_values
            )
            self._segments.append(
                {
                    "index": index,
                    "name": name,
                    "path": path,
                    "prim": prim,
                    "local": local,
                    "relative": relative,
                    "schema": schema,
                }
            )
            self._aggregate[name] = {
                "contact_counterparts": set(),
                "contact_records": 0,
                "maximum_contact_impulse_norm": 0.0,
                "maximum_local_rotation_error_rad": 0.0,
                "maximum_local_translation_error_m": 0.0,
                "maximum_parent_relative_rotation_error_rad": 0.0,
                "maximum_parent_relative_translation_error_m": 0.0,
                "minimum_contact_separation_m": None,
            }
        self._segment00_schema = self._segments[0]["schema"]

    def _sample_contacts(self, contact_report):
        per_step = {index: 0 for index in range(SEGMENT_COUNT)}
        if contact_report is None:
            return per_step
        headers, contacts, _ = contact_report
        for header in headers:
            collider_paths = (
                str(
                    self._PhysicsSchemaTools.intToSdfPath(header.collider0)
                ),
                str(
                    self._PhysicsSchemaTools.intToSdfPath(header.collider1)
                ),
            )
            for side, path in enumerate(collider_paths):
                if not path.startswith(self._nut_root + "/"):
                    continue
                index = segment_index_from_path(path)
                if index is None:
                    continue
                name = f"Segment_{index:02d}"
                counterpart = collider_paths[1 - side]
                aggregate = self._aggregate[name]
                aggregate["contact_counterparts"].add(counterpart)
                count = int(header.num_contact_data)
                aggregate["contact_records"] += count
                per_step[index] += count
                for contact_index in range(
                    int(header.contact_data_offset),
                    int(header.contact_data_offset) + count,
                ):
                    contact = contacts[contact_index]
                    impulse = math.sqrt(
                        sum(float(value) ** 2 for value in contact.impulse)
                    )
                    aggregate["maximum_contact_impulse_norm"] = max(
                        aggregate["maximum_contact_impulse_norm"], impulse
                    )
                    separation = float(contact.separation)
                    previous = aggregate["minimum_contact_separation_m"]
                    if previous is None or separation < previous:
                        aggregate["minimum_contact_separation_m"] = separation
        return per_step

    def sample(self, *, global_step, phase, phase_step, contact_report=None):
        """Observe all 24 teeth once; call exactly after each physics step."""

        if self._finalized:
            raise RuntimeError("nut tooth jitter probe is already finalized")
        cache = self._UsdGeom.XformCache(self._Usd.TimeCode.Default())
        details = []
        maximum_score = -1.0
        maximum_name = None
        maxima = {
            "local_t": 0.0,
            "local_r": 0.0,
            "relative_t": 0.0,
            "relative_r": 0.0,
        }
        for segment in self._segments:
            xformable = self._UsdGeom.Xformable(segment["prim"])
            local = xformable.GetLocalTransformation(
                self._Usd.TimeCode.Default()
            )
            relative, _ = cache.ComputeRelativeTransform(
                segment["prim"], self._parent_prim
            )
            local_t, local_r = _matrix_error(
                self._Gf, segment["local"], local
            )
            relative_t, relative_r = _matrix_error(
                self._Gf, segment["relative"], relative
            )
            values = {
                "index": segment["index"],
                "name": segment["name"],
                "local_rotation_error_rad": local_r,
                "local_translation_error_m": local_t,
                "parent_relative_rotation_error_rad": relative_r,
                "parent_relative_translation_error_m": relative_t,
            }
            details.append(values)
            aggregate = self._aggregate[segment["name"]]
            for key, value in (
                ("maximum_local_translation_error_m", local_t),
                ("maximum_local_rotation_error_rad", local_r),
                (
                    "maximum_parent_relative_translation_error_m",
                    relative_t,
                ),
                ("maximum_parent_relative_rotation_error_rad", relative_r),
            ):
                aggregate[key] = max(aggregate[key], value)
            maxima["local_t"] = max(maxima["local_t"], local_t)
            maxima["local_r"] = max(maxima["local_r"], local_r)
            maxima["relative_t"] = max(maxima["relative_t"], relative_t)
            maxima["relative_r"] = max(maxima["relative_r"], relative_r)
            score = max(
                local_t / TRANSLATION_THRESHOLD_M,
                relative_t / TRANSLATION_THRESHOLD_M,
                local_r / ROTATION_THRESHOLD_RAD,
                relative_r / ROTATION_THRESHOLD_RAD,
            )
            if score > maximum_score:
                maximum_score = score
                maximum_name = segment["name"]

        per_step_contacts = self._sample_contacts(contact_report)
        position, orientation = self._parent_rigid.get_world_pose()
        linear = self._parent_rigid.get_linear_velocity()
        angular = self._parent_rigid.get_angular_velocity()
        anomalous = bool(
            maxima["local_t"] > TRANSLATION_THRESHOLD_M
            or maxima["relative_t"] > TRANSLATION_THRESHOLD_M
            or maxima["local_r"] > ROTATION_THRESHOLD_RAD
            or maxima["relative_r"] > ROTATION_THRESHOLD_RAD
        )
        self._steps += 1
        self._phase_steps[str(phase)] = (
            self._phase_steps.get(str(phase), 0) + 1
        )
        if anomalous:
            self._anomaly_steps += 1
            _json_dump_line(
                self._anomaly_stream,
                {
                    "global_step": int(global_step),
                    "phase": str(phase),
                    "phase_step": int(phase_step),
                    "segments": details,
                },
            )
        row = {
            "global_step": int(global_step),
            "phase": str(phase),
            "phase_step": int(phase_step),
            "maximum_local_translation_error_m": maxima["local_t"],
            "maximum_local_rotation_error_rad": maxima["local_r"],
            "maximum_parent_relative_translation_error_m": maxima[
                "relative_t"
            ],
            "maximum_parent_relative_rotation_error_rad": maxima[
                "relative_r"
            ],
            "maximum_segment": maximum_name,
            "anomalous": int(anomalous),
            "parent_px_m": float(position[0]),
            "parent_py_m": float(position[1]),
            "parent_pz_m": float(position[2]),
            "parent_qw": float(orientation[0]),
            "parent_qx": float(orientation[1]),
            "parent_qy": float(orientation[2]),
            "parent_qz": float(orientation[3]),
            "parent_linear_speed_m_s": math.sqrt(
                sum(float(value) ** 2 for value in linear)
            ),
            "parent_angular_speed_rad_s": math.sqrt(
                sum(float(value) ** 2 for value in angular)
            ),
            "segment_contact_records": sum(per_step_contacts.values()),
        }
        self._summary_writer.writerow(row)
        self._summary_stream.flush()

    def snapshot(self):
        aggregate = {}
        for name, item in self._aggregate.items():
            serializable = dict(item)
            serializable["contact_counterparts"] = sorted(
                serializable["contact_counterparts"]
            )
            aggregate[name] = serializable
        return {
            "anomaly_steps": self._anomaly_steps,
            "color_identification": self._color_report,
            "fabric": {
                "available": False,
                "reason": (
                    "v1 records USD transforms and parent PhysX pose; a "
                    "stable usdrt per-prim interface was not assumed"
                ),
            },
            "normalization_ab": self._normalization_report,
            "output_directory": str(self._output),
            "phase_steps": dict(sorted(self._phase_steps.items())),
            "schema_version": SCHEMA_VERSION,
            "segment00_schema": self._segment00_schema,
            "segment_aggregate": aggregate,
            "steps": self._steps,
            "thresholds": {
                "rotation_rad": ROTATION_THRESHOLD_RAD,
                "translation_m": TRANSLATION_THRESHOLD_M,
            },
        }

    def finalize(self):
        """Write report.json and close streams; repeated calls are harmless."""

        report = self.snapshot()
        if not self._finalized:
            (self._output / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._summary_stream.close()
            self._anomaly_stream.close()
            self._finalized = True
        return report
