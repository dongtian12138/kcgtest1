#!/usr/bin/env python3

"""Frozen simulation-only RGB-D yaw dataset for keyed D38999 public-spec v2.

The module is safe to import without Isaac Sim. Its CPU side freezes the sample
schedule and enforces the observation/truth boundary. Isaac is loaded only by
the explicit ``--run`` path and requests RGB plus planar depth only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_keyed_v2_yaw_dataset_v1"
TRUTH_SCHEMA_VERSION = "kcg_d38999_keyed_v2_yaw_truth_record_v1"
FAILURE_SCHEMA_VERSION = "kcg_d38999_keyed_v2_yaw_generation_failure_v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "config/d38999_keyed_v2_yaw_dataset_v1.yaml"
)
ALLOWED_OBSERVATION_FIELDS = frozenset(
    {"rgb", "depth_m", "connector_face_mask", "occlusion_mask", "intrinsics"}
)
INTRINSIC_FIELDS = frozenset(
    {"width_px", "height_px", "fx_px", "fy_px", "cx_px", "cy_px"}
)
INFERENCE_SAMPLE_FILES = frozenset(
    {
        "rgb.png",
        "depth_m.npy",
        "connector_face_mask.npy",
        "occlusion_mask.npy",
        "intrinsics.json",
    }
)
DIAGNOSTIC_FILES = frozenset(
    {
        "rgb.png",
        "depth_m.npy",
        "connector_face_mask.npy",
        "occlusion_mask.npy",
        "diagnostic.json",
    }
)
DIAGNOSTIC_JSON_FIELDS = frozenset(
    {
        "expected",
        "observed",
        "face_pixels",
        "face_missing_depth_pixels",
        "valid_depth_pixels",
        "occlusion_pixels",
        "rgb_range",
        "rgb_std",
        "dataset_sample",
        "control",
        "truth_fields_included",
    }
)
DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS = frozenset(
    {
        "authored_yaw_deg",
        "authored_pose",
        "authored_light",
        "reject_injection",
        "expected_reject",
        "expected_reject_reason",
    }
)
REJECTION_REASONS = (
    "KEY_REGION_OCCLUDED",
    "CONNECTOR_FACE_OUT_OF_FRAME",
    "KEY_REGION_DEPTH_MISSING",
    "KEY_REGION_LOW_CONFIDENCE",
)
VISIBLE_VALID = "VISIBLE_VALID"
POSTCONDITION_CLASSES = (VISIBLE_VALID,) + REJECTION_REASONS
POSTCONDITION_PRECEDENCE = REJECTION_REASONS + (VISIBLE_VALID,)
POSTCONDITION_BORDER_MARGIN_PX = 2
POSTCONDITION_MINIMUM_FACE_PIXELS = 200
POSTCONDITION_MINIMUM_RGB_CONTRAST = 18.0
MISSING_DEPTH_COMPONENT_CONNECTIVITY = 8
MISSING_DEPTH_MINIMUM_COMPONENT_PIXELS = 16
FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
        "semantic_segmentation_truth",
        "semantic_label_truth",
        "object_pose_truth",
        "authored_yaw_deg",
        "authored_pose",
        "authored_light",
        "expected_reject",
        "expected_reject_reason",
        "contact_report",
        "contact_point_truth",
        "collider_identity",
        "penetration_depth_truth",
        "physx_manifold_truth",
    }
)
CONTROL_FIELDS = (
    "selected_for_control_allowed",
    "simulation_insertion_control_authorized",
    "control_authorized",
    "robot_control_authorized",
    "hardware_control_authorized",
)
FAILURE_REPORT_FIELDS = (
    "schema_version",
    "status",
    "exception_type",
    "exception_message",
    "generated_samples",
    "selected_mode",
    "selected_split",
    "control_authorized",
    "selected_for_control_allowed",
    "simulation_insertion_control_authorized",
    "robot_control_authorized",
    "hardware_control_authorized",
)
EXPECTED_SPLITS = {
    "dev": (1024, 512, 8, 512, 128),
    "heldout": (3072, 2048, 32, 1024, 256),
}
FROZEN_RANGES = {
    "camera_axial_distance_m": (0.09, 0.14),
    "lateral_x_m": (-0.006, 0.006),
    "lateral_y_m": (-0.006, 0.006),
    "tilt_x_deg": (-8.0, 8.0),
    "tilt_y_deg": (-8.0, 8.0),
}
_UINT64_MASK = (1 << 64) - 1


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _range(
    value: Any,
    label: str,
    expected: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be [minimum, maximum]")
    result = (_finite(value[0], label), _finite(value[1], label))
    if result[0] >= result[1]:
        raise ValueError(f"{label} must be increasing")
    if expected is not None and result != expected:
        raise ValueError(f"{label} is frozen at {list(expected)}")
    return result


def load_contract(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the contract and fail closed if its critical shape drifted."""

    document = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "contract"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected dataset schema")
    identity = _mapping(document.get("identity"), "identity")
    if identity.get("simulation_only") is not True:
        raise ValueError("dataset must remain simulation-only")
    if identity.get("real_hardware_data_claimed") is not False:
        raise ValueError("dataset cannot claim real-hardware data")

    schedule = _mapping(document.get("frozen_schedule"), "frozen_schedule")
    if schedule.get("generator") != "splitmix64_v1":
        raise ValueError("schedule generator changed")
    if schedule.get("order") != "stable_fisher_yates_v1":
        raise ValueError("schedule order changed")
    if schedule.get("sample_id_format") != "{split}_{index:06d}":
        raise ValueError("sample ID format changed")
    _range(schedule.get("yaw_range_deg"), "yaw range", (-180.0, 180.0))
    if _integer(schedule.get("yaw_strata"), "yaw_strata") != 64:
        raise ValueError("yaw_strata must remain 64")
    if schedule.get("yaw_upper_bound_exclusive") is not True:
        raise ValueError("yaw range must stay half-open")

    for name, expected in EXPECTED_SPLITS.items():
        total, visible_count, nuisance_count, reject_count, per_reject = expected
        split = _mapping(schedule.get(name), name)
        for seed_name in ("sampling_seed_u64", "order_seed_u64"):
            if _integer(split.get(seed_name), f"{name}.{seed_name}") < 0:
                raise ValueError("schedule seeds must be nonnegative")
        visible = _mapping(split.get("visible_valid"), f"{name}.visible_valid")
        reject = _mapping(split.get("must_reject"), f"{name}.must_reject")
        categories = _mapping(reject.get("categories"), f"{name}.categories")
        counts_match = all(
            (
                _integer(split.get("total"), f"{name}.total") == total,
                _integer(visible.get("count"), f"{name}.visible")
                == visible_count,
                _integer(visible.get("yaw_strata"), f"{name}.yaw_strata")
                == 64,
                _integer(
                    visible.get("nuisance_per_stratum"), f"{name}.nuisance"
                )
                == nuisance_count,
                _integer(reject.get("count"), f"{name}.reject")
                == reject_count,
                set(categories) == set(REJECTION_REASONS),
                all(
                    _integer(categories[key], key) == per_reject
                    for key in REJECTION_REASONS
                ),
            )
        )
        if not counts_match or visible_count + reject_count != total:
            raise ValueError(f"{name} frozen counts changed")
    if schedule["heldout"].get("used_for_tuning") is not False:
        raise ValueError("heldout split cannot be used for tuning")

    nuisance = _mapping(document.get("nuisance_ranges"), "nuisance_ranges")
    for name, expected in FROZEN_RANGES.items():
        _range(nuisance.get(name), name, expected)
    for name in (
        "key_light_intensity",
        "light_color_temperature_k",
        "light_azimuth_deg",
        "exposure_ev",
    ):
        _range(nuisance.get(name), name)

    directory = _mapping(document.get("directory_contract"), "directory_contract")
    if directory.get("existing_output_policy") != "REFUSE_OVERWRITE":
        raise ValueError("output policy must refuse overwrite")
    if set(directory.get("inference_sample_files_exactly", ())) != INFERENCE_SAMPLE_FILES:
        raise ValueError("inference file contract changed")
    if directory.get("connector_face_mask_source") != "RGB_DEPTH_IMAGE_DERIVED_ONLY":
        raise ValueError("mask must remain image-derived")
    if directory.get("occlusion_mask_source") != "RGB_DEPTH_IMAGE_DERIVED_ONLY":
        raise ValueError("occlusion mask must remain image-derived")
    if directory.get("occlusion_unknown_encoding") != (
        "SCALAR_UNICODE_OCCLUSION_UNKNOWN"
    ):
        raise ValueError("occlusion unknown encoding changed")
    if directory.get("write_requires_pixel_postcondition_match") is not True:
        raise ValueError("pixel postcondition must gate every sample write")

    firewall = _mapping(document.get("prediction_api_firewall"), "firewall")
    if set(firewall.get("allowed_fields_exactly", ())) != ALLOWED_OBSERVATION_FIELDS:
        raise ValueError("prediction input fields changed")
    if set(firewall.get("forbidden_fields", ())) != FORBIDDEN_PREDICTION_FIELDS:
        raise ValueError("prediction truth firewall changed")
    if firewall.get("truth_directory_visible_to_prediction_api") is not False:
        raise ValueError("prediction API cannot see truth")
    if firewall.get("mask_may_use_semantics") is not False:
        raise ValueError("mask cannot use semantic truth")
    if firewall.get("occlusion_mask_may_use_reject_plan_or_truth") is not False:
        raise ValueError("occlusion mask cannot use reject plan or truth")

    postcondition = _mapping(
        document.get("pixel_postcondition"), "pixel_postcondition"
    )
    if tuple(postcondition.get("classes_exactly", ())) != POSTCONDITION_CLASSES:
        raise ValueError("pixel postcondition classes changed")
    if tuple(postcondition.get("precedence", ())) != POSTCONDITION_PRECEDENCE:
        raise ValueError("pixel postcondition precedence changed")
    if set(postcondition.get("classifier_inputs_exactly", ())) != (
        ALLOWED_OBSERVATION_FIELDS
    ):
        raise ValueError("postcondition classifier inputs changed")
    if postcondition.get("connector_face_mask_semantics") != (
        "FINAL_RGBD_VISIBLE_CONNECTOR_FACE_EXCLUDES_OCCLUDER"
    ):
        raise ValueError("connector face mask semantics changed")
    if postcondition.get("occlusion_mask_semantics") != (
        "FINAL_RGBD_FOREGROUND_OCCLUDER_ONLY_DISJOINT_FROM_FACE"
    ):
        raise ValueError("occlusion mask semantics changed")
    if postcondition.get("missing_depth_support_semantics") != (
        "RGB_FOREGROUND_INVALID_DEPTH_NO_VALID_DEPTH_IN_3X3"
    ):
        raise ValueError("dataset missing-depth support semantics changed")
    if postcondition.get("missing_depth_3x3_boundary_policy") != (
        "OUT_OF_IMAGE_TREATED_AS_NO_VALID_DEPTH"
    ):
        raise ValueError("dataset missing-depth boundary policy changed")
    if postcondition.get("missing_depth_component_connectivity") != (
        MISSING_DEPTH_COMPONENT_CONNECTIVITY
    ):
        raise ValueError("dataset missing-depth connectivity changed")
    if postcondition.get("missing_depth_minimum_component_pixels") != (
        MISSING_DEPTH_MINIMUM_COMPONENT_PIXELS
    ):
        raise ValueError("dataset missing-depth component threshold changed")
    if postcondition.get("depth_missing_scope") != (
        "SUBSTANTIAL_SIMULATED_DROPOUT_DATASET_POSTCONDITION_ONLY"
    ):
        raise ValueError("dataset missing-depth scope changed")
    if postcondition.get(
        "refiner_key_centroid_3x3_strict_missing_depth_gate_replaced"
    ) is not False:
        raise ValueError("dataset mask cannot replace the refiner depth gate")
    for policy in (
        "empty_face_policy",
        "occlusion_unknown_policy",
        "expected_class_mismatch_policy",
    ):
        if postcondition.get(policy) != "ERROR_NO_WRITE":
            raise ValueError(f"pixel_postcondition.{policy} must fail before write")
    if postcondition.get("image_border_margin_px") != (
        POSTCONDITION_BORDER_MARGIN_PX
    ):
        raise ValueError("postcondition border margin changed")
    if postcondition.get("minimum_face_pixels") != (
        POSTCONDITION_MINIMUM_FACE_PIXELS
    ):
        raise ValueError("postcondition minimum face support changed")
    if postcondition.get("minimum_rgb_face_background_contrast") != (
        POSTCONDITION_MINIMUM_RGB_CONTRAST
    ):
        raise ValueError("postcondition RGB contrast changed")

    preflight = _mapping(document.get("preflight"), "preflight")
    if preflight.get("explicit_cli_flag") != "--preflight":
        raise ValueError("preflight must remain explicit")
    if preflight.get("source_split") != "dev":
        raise ValueError("preflight source split must remain dev")
    if tuple(preflight.get("required_classes", ())) != POSTCONDITION_CLASSES:
        raise ValueError("preflight must cover all postconditions")
    if preflight.get("existing_output_policy") != "REFUSE_OVERWRITE":
        raise ValueError("preflight must refuse overwrite")
    if preflight.get("success_requires_all_postconditions") is not True:
        raise ValueError("preflight must require every postcondition")

    camera = _mapping(document.get("camera"), "camera")
    if camera.get("prim_path") != "/World/KeyedV2YawDatasetCamera":
        raise ValueError("camera prim path changed")
    if camera.get("render_product_name") != (
        "D38999KeyedV2YawDatasetProduct"
    ):
        raise ValueError("camera render product name changed")
    if camera.get("rgb_channel") != "rgb":
        raise ValueError("only the RGB annotator is allowed")
    if camera.get("depth_channel") != "distance_to_image_plane":
        raise ValueError("only planar depth is allowed")
    if camera.get("semantic_annotator_allowed") is not False:
        raise ValueError("semantic annotator must remain disabled")

    runtime = _mapping(document.get("runtime_capture"), "runtime_capture")
    expected_runtime = {
        "app_updates_after_annotator_attach": 2,
        "replicator_warmup_frames": 4,
        "per_sample_render_frames": 1,
        "rt_subframes": 2,
    }
    for field, expected in expected_runtime.items():
        if _integer(runtime.get(field), f"runtime_capture.{field}") != expected:
            raise ValueError(f"runtime_capture.{field} changed")
    for field in (
        "require_exact_resolution",
        "require_nonempty_rgb",
        "require_positive_finite_depth",
    ):
        if runtime.get(field) is not True:
            raise ValueError(f"runtime_capture.{field} must remain true")

    reports = _mapping(document.get("runtime_reports"), "runtime_reports")
    if reports.get("success_filename") != "generation_report.json":
        raise ValueError("success report filename changed")
    if reports.get("failure_filename") != "generation_failure.json":
        raise ValueError("failure report filename changed")
    if reports.get("exclusive_create") is not True:
        raise ValueError("runtime reports must use exclusive create")
    if reports.get("report_write_before_app_close") is not True:
        raise ValueError("runtime report must be written before app close")
    if reports.get("traceback_on_failure") is not True:
        raise ValueError("runtime failures must record a traceback")
    if reports.get("app_close_exit_code_success") != 0:
        raise ValueError("successful app close exit code must remain zero")
    if reports.get("app_close_exit_code_failure") != 1:
        raise ValueError("failed app close exit code must remain one")
    if reports.get("preflight_success_generated_samples") != 5:
        raise ValueError("preflight generated count must remain five")
    if reports.get("preflight_success_each_postcondition_count") != 1:
        raise ValueError("preflight class count must remain one each")
    if reports.get("empty_output_directories_are_success") is not False:
        raise ValueError("empty output directories can never pass")
    if tuple(reports.get("failure_fields_exactly", ())) != FAILURE_REPORT_FIELDS:
        raise ValueError("failure report fields changed")

    diagnostics = _mapping(
        document.get("failure_diagnostics"), "failure_diagnostics"
    )
    if diagnostics.get("enabled") is not True:
        raise ValueError("preflight failure diagnostics must remain enabled")
    if diagnostics.get("allowed_mode") != "PREFLIGHT_FAILURE_ONLY":
        raise ValueError("failure diagnostics are allowed only for preflight")
    if diagnostics.get("root") != "diagnostics":
        raise ValueError("failure diagnostics root changed")
    if diagnostics.get("sample_directory_format") != "{sample_id}":
        raise ValueError("failure diagnostic sample directory changed")
    if diagnostics.get("existing_sample_policy") != "REFUSE_OVERWRITE":
        raise ValueError("failure diagnostics must refuse overwrite")
    if set(diagnostics.get("files_exactly", ())) != DIAGNOSTIC_FILES:
        raise ValueError("failure diagnostic file contract changed")
    if diagnostics.get("occlusion_unknown_encoding") != (
        "SCALAR_UNICODE_OCCLUSION_UNKNOWN"
    ):
        raise ValueError("failure diagnostic occlusion encoding changed")
    if set(diagnostics.get("json_fields_exactly", ())) != (
        DIAGNOSTIC_JSON_FIELDS
    ):
        raise ValueError("failure diagnostic JSON fields changed")
    if set(diagnostics.get("forbidden_truth_fields", ())) != (
        DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS
    ):
        raise ValueError("failure diagnostic truth firewall changed")

    authorization = _mapping(document.get("authorization"), "authorization")
    for field in (
        "selected_for_control_allowed",
        "simulation_insertion_control_authorized",
        "robot_control_authorized",
        "hardware_control_authorized",
        "real_hardware_fidelity_claimed",
        "real_assembly_success_claimed",
    ):
        if authorization.get(field) is not False:
            raise ValueError(f"authorization.{field} must remain false")
    return dict(document)


class SplitMix64:
    """Versioned PRNG independent of Python and NumPy RNG implementations."""

    def __init__(self, seed: int):
        self._state = _integer(seed, "seed") & _UINT64_MASK

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return (value ^ (value >> 31)) & _UINT64_MASK

    def uniform(self, minimum: float, maximum: float) -> float:
        unit = (self.next_u64() >> 11) * (1.0 / (1 << 53))
        return minimum + (maximum - minimum) * unit

    def randbelow(self, upper: int) -> int:
        bound = _integer(upper, "upper")
        if bound <= 0:
            raise ValueError("upper must be positive")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    split: str
    authored_yaw_deg: float
    yaw_stratum: int
    authored_pose: Mapping[str, Any]
    authored_light: Mapping[str, Any]
    expected_reject: bool
    expected_reject_reason: str | None
    reject_injection: Mapping[str, Any]

    def truth_record(self) -> dict[str, Any]:
        return {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "split": self.split,
            "authored_yaw_deg": self.authored_yaw_deg,
            "yaw_stratum": self.yaw_stratum,
            "authored_pose": dict(self.authored_pose),
            "authored_light": dict(self.authored_light),
            "expected_reject": self.expected_reject,
            "expected_reject_reason": self.expected_reject_reason,
            "reject_injection": dict(self.reject_injection),
        }


def _draw(rng: SplitMix64, ranges: Mapping[str, Any], name: str) -> float:
    minimum, maximum = _range(ranges[name], name)
    return rng.uniform(minimum, maximum)


def _reject_injection(
    rng: SplitMix64,
    documents: Mapping[str, Any],
    reason: str | None,
) -> dict[str, Any]:
    if reason is None:
        return {"kind": "NONE"}
    source = _mapping(documents[reason], reason)
    result: dict[str, Any] = {
        "kind": reason,
        "implementation": source["implementation"],
    }
    if reason == "KEY_REGION_OCCLUDED":
        result["occluded_face_fraction"] = _draw(
            rng, source, "occluded_face_fraction"
        )
        result["angle_deg"] = rng.uniform(-180.0, 180.0)
    elif reason == "CONNECTOR_FACE_OUT_OF_FRAME":
        result["camera_framing_shift_fraction"] = _draw(
            rng, source, "camera_framing_shift_fraction"
        )
        result["angle_deg"] = rng.uniform(-180.0, 180.0)
    elif reason == "KEY_REGION_DEPTH_MISSING":
        result["key_region_depth_dropout_fraction"] = _draw(
            rng, source, "key_region_depth_dropout_fraction"
        )
        result["angle_deg"] = rng.uniform(-180.0, 180.0)
    elif reason == "KEY_REGION_LOW_CONFIDENCE":
        result["contrast_scale"] = _draw(rng, source, "contrast_scale")
        result["gaussian_blur_radius_px"] = _draw(
            rng, source, "gaussian_blur_radius_px"
        )
    else:
        raise ValueError(f"unknown rejection reason: {reason}")
    return result


def _sample_payload(
    rng: SplitMix64,
    contract: Mapping[str, Any],
    stratum: int,
    reason: str | None,
) -> dict[str, Any]:
    schedule = contract["frozen_schedule"]
    ranges = contract["nuisance_ranges"]
    width = 360.0 / 64.0
    center = -180.0 + (stratum + 0.5) * width
    jitter = 0.5 * width * float(schedule["yaw_within_stratum_fraction"])
    yaw = rng.uniform(center - jitter, center + jitter)
    axial = _draw(rng, ranges, "camera_axial_distance_m")
    lateral_x = _draw(rng, ranges, "lateral_x_m")
    lateral_y = _draw(rng, ranges, "lateral_y_m")
    tilt_x = _draw(rng, ranges, "tilt_x_deg")
    tilt_y = _draw(rng, ranges, "tilt_y_deg")
    return {
        "authored_yaw_deg": yaw,
        "yaw_stratum": stratum,
        "authored_pose": {
            "frame": "camera_optical",
            "translation_m": [lateral_x, lateral_y, axial],
            "rotation_xyz_deg": [tilt_x, tilt_y, yaw],
        },
        "authored_light": {
            "key_light_intensity": _draw(rng, ranges, "key_light_intensity"),
            "color_temperature_k": _draw(
                rng, ranges, "light_color_temperature_k"
            ),
            "azimuth_deg": _draw(rng, ranges, "light_azimuth_deg"),
            "exposure_ev": _draw(rng, ranges, "exposure_ev"),
        },
        "expected_reject": reason is not None,
        "expected_reject_reason": reason,
        "reject_injection": _reject_injection(
            rng, contract["reject_injections"], reason
        ),
    }


def build_split_schedule(
    contract: Mapping[str, Any], split: str
) -> tuple[SamplePlan, ...]:
    """Build one deterministic split without importing or querying Isaac."""

    if split not in EXPECTED_SPLITS:
        raise ValueError("split must be dev or heldout")
    schedule = contract["frozen_schedule"]
    split_config = schedule[split]
    rng = SplitMix64(split_config["sampling_seed_u64"])
    payloads: list[dict[str, Any]] = []
    nuisance_per_stratum = int(
        split_config["visible_valid"]["nuisance_per_stratum"]
    )
    for stratum in range(64):
        for _ in range(nuisance_per_stratum):
            payloads.append(_sample_payload(rng, contract, stratum, None))
    for reason in REJECTION_REASONS:
        count = int(split_config["must_reject"]["categories"][reason])
        if count % 64:
            raise ValueError("reject category count must divide 64 strata")
        for index in range(count):
            payloads.append(_sample_payload(rng, contract, index % 64, reason))

    order_rng = SplitMix64(split_config["order_seed_u64"])
    for index in range(len(payloads) - 1, 0, -1):
        other = order_rng.randbelow(index + 1)
        payloads[index], payloads[other] = payloads[other], payloads[index]
    if len(payloads) != split_config["total"]:
        raise RuntimeError("built schedule size differs from contract")

    sample_format = schedule["sample_id_format"]
    return tuple(
        SamplePlan(
            sample_id=sample_format.format(split=split, index=index),
            split=split,
            **payload,
        )
        for index, payload in enumerate(payloads)
    )


def build_dataset_schedule(
    contract: Mapping[str, Any],
) -> dict[str, tuple[SamplePlan, ...]]:
    return {
        split: build_split_schedule(contract, split) for split in EXPECTED_SPLITS
    }


def expected_postcondition(plan: SamplePlan) -> str:
    return (
        plan.expected_reject_reason
        if plan.expected_reject_reason is not None
        else VISIBLE_VALID
    )


def build_preflight_schedule(
    contract: Mapping[str, Any],
) -> dict[str, tuple[SamplePlan, ...]]:
    """Select one deterministic frozen dev sample for every postcondition."""

    dev = build_split_schedule(contract, "dev")
    groups = {
        category: [plan for plan in dev if expected_postcondition(plan) == category]
        for category in POSTCONDITION_CLASSES
    }
    if any(not plans for plans in groups.values()):
        raise RuntimeError("frozen dev schedule cannot cover every preflight class")

    selected = {
        VISIBLE_VALID: groups[VISIBLE_VALID][0],
        "KEY_REGION_OCCLUDED": max(
            groups["KEY_REGION_OCCLUDED"],
            key=lambda plan: plan.reject_injection["occluded_face_fraction"],
        ),
        "CONNECTOR_FACE_OUT_OF_FRAME": min(
            groups["CONNECTOR_FACE_OUT_OF_FRAME"],
            key=lambda plan: plan.reject_injection[
                "camera_framing_shift_fraction"
            ],
        ),
        "KEY_REGION_DEPTH_MISSING": max(
            groups["KEY_REGION_DEPTH_MISSING"],
            key=lambda plan: plan.reject_injection[
                "key_region_depth_dropout_fraction"
            ],
        ),
        "KEY_REGION_LOW_CONFIDENCE": min(
            groups["KEY_REGION_LOW_CONFIDENCE"],
            key=lambda plan: plan.reject_injection["contrast_scale"],
        ),
    }
    return {"dev": tuple(selected[name] for name in POSTCONDITION_CLASSES)}


def selected_splits(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return tuple(EXPECTED_SPLITS)
    if selection in EXPECTED_SPLITS:
        return (selection,)
    raise ValueError("split selection must be dev, heldout, or all")


def default_output_root(
    repository: Path, selection: str, *, preflight: bool = False
) -> Path:
    base = repository / "artifacts/kcg_connector/d38999_keyed_v2_yaw_dataset_v1"
    if preflight:
        return base.with_name(f"{base.name}_preflight")
    return base if selection == "all" else base.with_name(f"{base.name}_{selection}")


def summarize_schedule(
    schedules: Mapping[str, Sequence[SamplePlan]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, plans in schedules.items():
        result[split] = {
            "total": len(plans),
            "visible_valid": sum(not plan.expected_reject for plan in plans),
            "must_reject": sum(plan.expected_reject for plan in plans),
            "reject_categories": {
                reason: sum(plan.expected_reject_reason == reason for plan in plans)
                for reason in REJECTION_REASONS
            },
        }
    return result


def require_new_output_root(output_root: Path | str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {root}")
    return root


def prepare_output_layout(
    output_root: Path | str,
    contract: Mapping[str, Any],
    splits: Sequence[str] = tuple(EXPECTED_SPLITS),
) -> dict[str, Path]:
    """Create disjoint inference/truth trees, refusing any existing root."""

    root = require_new_output_root(output_root)
    directory = contract["directory_contract"]
    layout = {
        "root": root,
        "inference_root": root / directory["inference_root"],
        "truth_root": root / directory["truth_root"],
        "diagnostics_root": root / contract["failure_diagnostics"]["root"],
    }
    root.mkdir(parents=True, exist_ok=False)
    layout["inference_root"].mkdir()
    layout["truth_root"].mkdir()
    split_names = tuple(splits)
    if not split_names or len(set(split_names)) != len(split_names):
        raise ValueError("output split list must be nonempty and unique")
    if any(split not in EXPECTED_SPLITS for split in split_names):
        raise ValueError("output split must be dev or heldout")
    for split in split_names:
        layout[f"inference_{split}"] = layout["inference_root"] / split
        layout[f"truth_{split}"] = layout["truth_root"] / split
        layout[f"inference_{split}"].mkdir()
        layout[f"truth_{split}"].mkdir()
    return layout


def _validate_intrinsics(value: Any) -> dict[str, float | int]:
    raw = _mapping(value, "intrinsics")
    if set(raw) != INTRINSIC_FIELDS:
        raise ValueError("intrinsics must contain exactly six camera fields")
    result: dict[str, float | int] = {}
    for name in INTRINSIC_FIELDS:
        number = _finite(raw[name], f"intrinsics.{name}")
        if name in {"width_px", "height_px"}:
            if number <= 0 or not number.is_integer():
                raise ValueError(f"intrinsics.{name} must be a positive integer")
            result[name] = int(number)
        else:
            if name in {"fx_px", "fy_px"} and number <= 0:
                raise ValueError(f"intrinsics.{name} must be positive")
            result[name] = number
    return result


def validate_prediction_observation(observation: Any) -> dict[str, Any]:
    """Accept exactly RGB, depth, image-derived mask and intrinsics."""

    raw = _mapping(observation, "prediction observation")
    keys = set(raw)
    forbidden = keys & FORBIDDEN_PREDICTION_FIELDS
    if forbidden:
        raise ValueError("truth fields reached prediction: " + ", ".join(sorted(forbidden)))
    if keys != ALLOWED_OBSERVATION_FIELDS:
        raise ValueError("prediction observation must contain exactly five image fields")
    rgb = np.asarray(raw["rgb"])
    depth = np.asarray(raw["depth_m"])
    mask = np.asarray(raw["connector_face_mask"])
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError("RGB, depth and mask dimensions must match")
    if mask.dtype != np.bool_:
        raise ValueError("connector_face_mask must be boolean")
    occlusion_value = raw["occlusion_mask"]
    occlusion = None if occlusion_value is None else np.asarray(occlusion_value)
    if occlusion is not None:
        if occlusion.shape != depth.shape or occlusion.dtype != np.bool_:
            raise ValueError("occlusion_mask must be boolean and match RGB-D")
    intrinsics = _validate_intrinsics(raw["intrinsics"])
    if (intrinsics["width_px"], intrinsics["height_px"]) != (
        rgb.shape[1],
        rgb.shape[0],
    ):
        raise ValueError("intrinsics dimensions do not match RGB-D")
    return {
        "rgb": rgb,
        "depth_m": depth,
        "connector_face_mask": mask,
        "occlusion_mask": occlusion,
        "intrinsics": intrinsics,
    }


def classify_observation_postcondition(observation: Mapping[str, Any]) -> str:
    """Classify final pixels only; this API accepts no plan or truth."""

    safe = validate_prediction_observation(observation)
    face = safe["connector_face_mask"]
    occlusion = safe["occlusion_mask"]
    face_pixels = int(np.count_nonzero(face))
    if face_pixels == 0:
        raise RuntimeError("final connector_face_mask is empty; refusing write")
    if occlusion is None:
        raise RuntimeError("final occlusion is unknown; refusing write")
    if np.any(face & occlusion):
        raise RuntimeError("face and occlusion masks must be disjoint")
    if np.any(occlusion):
        return "KEY_REGION_OCCLUDED"

    margin = POSTCONDITION_BORDER_MARGIN_PX
    touches_border = bool(
        np.any(face[:margin])
        or np.any(face[-margin:])
        or np.any(face[:, :margin])
        or np.any(face[:, -margin:])
    )
    if touches_border:
        return "CONNECTOR_FACE_OUT_OF_FRAME"

    depth = np.asarray(safe["depth_m"], dtype=np.float64)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    if np.any(face & ~valid_depth):
        return "KEY_REGION_DEPTH_MISSING"
    if face_pixels < POSTCONDITION_MINIMUM_FACE_PIXELS:
        return "KEY_REGION_LOW_CONFIDENCE"

    rgb = np.asarray(safe["rgb"], dtype=np.float64)
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    background = ~face
    if not np.any(background):
        raise RuntimeError("final observation has no background support")
    contrast = abs(float(np.median(luminance[face])) - float(
        np.median(luminance[background])
    ))
    if contrast < POSTCONDITION_MINIMUM_RGB_CONTRAST:
        return "KEY_REGION_LOW_CONFIDENCE"
    return VISIBLE_VALID


def call_prediction_api(
    predictor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Call an offline predictor and reject any attempted control promotion."""

    safe_observation = validate_prediction_observation(observation)
    if safe_observation["occlusion_mask"] is None:
        result = {
            "status": "OCCLUSION_UNKNOWN",
            "rejection_code": "KEY_REGION_OCCLUSION_UNKNOWN",
        }
    else:
        result = dict(_mapping(predictor(safe_observation), "prediction result"))
    for field in CONTROL_FIELDS:
        if result.get(field, False) is not False:
            raise RuntimeError(f"prediction attempted to promote {field}")
        result[field] = False
    result["authorization_scope"] = "OFFLINE_RGBD_YAW_EVALUATION_ONLY_NO_CONTROL"
    return result


def _any_valid_depth_in_3x3(valid_depth: Any) -> np.ndarray:
    """Return whether each 3x3 neighborhood contains valid in-image depth.

    Padding is False: pixels outside the image are explicitly treated as not
    having valid depth. This keeps a genuinely missing foreground region at an
    image edge available to the separate out-of-frame postcondition.
    """

    valid = np.asarray(valid_depth, dtype=np.bool_)
    if valid.ndim != 2:
        raise ValueError("valid_depth must be a two-dimensional mask")
    height, width = valid.shape
    padded = np.pad(valid, ((1, 1), (1, 1)), constant_values=False)
    neighborhood = np.zeros(valid.shape, dtype=np.bool_)
    for row_offset in range(3):
        for column_offset in range(3):
            neighborhood |= padded[
                row_offset:row_offset + height,
                column_offset:column_offset + width,
            ]
    return neighborhood


def _filter_small_8_connected_components(
    mask: Any,
    minimum_pixels: int = MISSING_DEPTH_MINIMUM_COMPONENT_PIXELS,
) -> np.ndarray:
    """Keep only 8-connected components meeting the frozen pixel minimum."""

    candidate = np.asarray(mask, dtype=np.bool_)
    if candidate.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if (
        isinstance(minimum_pixels, bool)
        or not isinstance(minimum_pixels, Integral)
        or int(minimum_pixels) <= 0
    ):
        raise ValueError("minimum component pixels must be a positive integer")
    minimum = int(minimum_pixels)
    kept = np.zeros(candidate.shape, dtype=np.bool_)
    visited = np.zeros(candidate.shape, dtype=np.bool_)
    height, width = candidate.shape
    for seed_row, seed_column in np.argwhere(candidate):
        seed_row, seed_column = int(seed_row), int(seed_column)
        if visited[seed_row, seed_column]:
            continue
        visited[seed_row, seed_column] = True
        stack = [(seed_row, seed_column)]
        component: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            component.append((row, column))
            for row_neighbor in range(max(0, row - 1), min(height, row + 2)):
                for column_neighbor in range(
                    max(0, column - 1), min(width, column + 2)
                ):
                    if (
                        candidate[row_neighbor, column_neighbor]
                        and not visited[row_neighbor, column_neighbor]
                    ):
                        visited[row_neighbor, column_neighbor] = True
                        stack.append((row_neighbor, column_neighbor))
        if len(component) >= minimum:
            rows, columns = zip(*component)
            kept[np.asarray(rows), np.asarray(columns)] = True
    return kept


def derive_image_masks(
    rgb: Any, depth_m: Any
) -> tuple[np.ndarray, np.ndarray | None]:
    """Derive face/occlusion masks from final RGB-D pixels only.

    The isolated simulation stage has a dark background.  Front-face support
    comes from a generous depth band; RGB support preserves only substantial
    8-connected regions whose depth was deliberately dropped. A foreground
    patch is called an occluder only when both its observed color and its nearer
    depth support that conclusion. If no non-occluder front-depth reference
    exists, occlusion stays unknown.
    """

    rgb_array = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if depth.shape != rgb_array.shape[:2]:
        raise ValueError("depth dimensions must match RGB")
    valid = np.isfinite(depth) & (depth > 0.0)
    empty = np.zeros(depth.shape, dtype=np.bool_)
    if not np.any(valid):
        return empty, None

    border = np.concatenate(
        (rgb_array[0], rgb_array[-1], rgb_array[:, 0], rgb_array[:, -1]),
        axis=0,
    ).astype(np.float64)
    background_rgb = np.median(border, axis=0)
    rgb_foreground = np.max(
        np.abs(rgb_array.astype(np.float64) - background_rgb), axis=2
    ) >= 6.0

    # The simulated occluder is rendered into the final observation as a dark
    # foreground surface.  Color alone is insufficient: it must also be at
    # least 2 mm closer than the independently estimated connector face.
    dark_patch = np.max(
        np.abs(rgb_array.astype(np.int16) - np.asarray((25, 25, 25))), axis=2
    ) <= 1
    reference_candidates = valid & ~dark_patch
    if not np.any(reference_candidates):
        return np.asarray(valid | rgb_foreground, dtype=np.bool_), None
    front = float(np.quantile(depth[reference_candidates], 0.03))
    occlusion = np.asarray(
        dark_patch & valid & (depth <= front - 0.002), dtype=np.bool_
    )
    front_visible = valid & ~occlusion & (depth <= front + 0.008)
    valid_depth_in_3x3 = _any_valid_depth_in_3x3(valid)
    missing_depth_candidates = (
        ~valid & rgb_foreground & ~valid_depth_in_3x3
    )
    missing_depth_support = _filter_small_8_connected_components(
        missing_depth_candidates
    )
    face = np.asarray(
        (front_visible | missing_depth_support) & ~occlusion,
        dtype=np.bool_,
    )
    return face, occlusion


def camera_intrinsics(contract: Mapping[str, Any]) -> dict[str, float | int]:
    camera = contract["camera"]
    width, height = (int(value) for value in camera["resolution_px"])
    focal = float(camera["focal_length_mm"])
    horizontal = float(camera["horizontal_aperture_mm"])
    vertical = float(camera["vertical_aperture_mm"])
    return {
        "width_px": width,
        "height_px": height,
        "fx_px": width * focal / horizontal,
        "fy_px": height * focal / vertical,
        "cx_px": 0.5 * (width - 1),
        "cy_px": 0.5 * (height - 1),
    }


def _write_preflight_mismatch_diagnostic(
    layout: Mapping[str, Path],
    plan: SamplePlan,
    observation: Mapping[str, Any],
    observed_class: str,
    *,
    preflight: bool,
) -> Path:
    """Persist final pixels for one preflight mismatch, never as a sample."""

    if preflight is not True:
        raise RuntimeError("failure diagnostics are allowed only for preflight")
    expected_class = expected_postcondition(plan)
    if observed_class == expected_class:
        raise RuntimeError("failure diagnostics require a pixel class mismatch")
    safe = validate_prediction_observation(observation)
    rgb = np.asarray(safe["rgb"], dtype=np.uint8)
    depth = np.asarray(safe["depth_m"], dtype=np.float64)
    face = np.asarray(safe["connector_face_mask"], dtype=np.bool_)
    occlusion = safe["occlusion_mask"]
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    diagnostic = {
        "expected": expected_class,
        "observed": str(observed_class),
        "face_pixels": int(np.count_nonzero(face)),
        "face_missing_depth_pixels": int(
            np.count_nonzero(face & ~valid_depth)
        ),
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
        "occlusion_pixels": (
            None if occlusion is None else int(np.count_nonzero(occlusion))
        ),
        "rgb_range": [int(rgb.min()), int(rgb.max())],
        "rgb_std": float(np.std(rgb.astype(np.float64))),
        "dataset_sample": False,
        "control": False,
        "truth_fields_included": False,
    }
    if set(diagnostic) != DIAGNOSTIC_JSON_FIELDS:
        raise RuntimeError("internal failure diagnostic fields changed")
    if set(diagnostic) & DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS:
        raise RuntimeError("truth fields reached failure diagnostic")

    diagnostics_root = layout["diagnostics_root"]
    diagnostics_root.mkdir(exist_ok=True)
    diagnostic_dir = diagnostics_root / plan.sample_id
    diagnostic_dir.mkdir(exist_ok=False)

    from PIL import Image

    with (diagnostic_dir / "rgb.png").open("xb") as stream:
        Image.fromarray(rgb).save(stream, format="PNG")
    with (diagnostic_dir / "depth_m.npy").open("xb") as stream:
        np.save(stream, depth.astype(np.float32), allow_pickle=False)
    with (diagnostic_dir / "connector_face_mask.npy").open("xb") as stream:
        np.save(stream, face, allow_pickle=False)
    encoded_occlusion = (
        np.asarray("OCCLUSION_UNKNOWN")
        if occlusion is None
        else np.asarray(occlusion, dtype=np.bool_)
    )
    with (diagnostic_dir / "occlusion_mask.npy").open("xb") as stream:
        np.save(stream, encoded_occlusion, allow_pickle=False)
    with (diagnostic_dir / "diagnostic.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(diagnostic, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    if {item.name for item in diagnostic_dir.iterdir()} != DIAGNOSTIC_FILES:
        raise RuntimeError("failure diagnostic file set is incomplete")
    return diagnostic_dir


def _raise_pixel_postcondition_mismatch(
    layout: Mapping[str, Path],
    plan: SamplePlan,
    observation: Mapping[str, Any],
    observed_class: str,
    *,
    preflight: bool,
) -> None:
    expected_class = expected_postcondition(plan)
    if preflight:
        _write_preflight_mismatch_diagnostic(
            layout,
            plan,
            observation,
            observed_class,
            preflight=True,
        )
    raise RuntimeError(
        "pixel postcondition mismatch before write: "
        f"expected {expected_class}, observed {observed_class}"
    )


def write_separated_sample(
    layout: Mapping[str, Path],
    plan: SamplePlan,
    observation: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write observation files and authored truth into separate new paths."""

    safe = validate_prediction_observation(observation)
    observed_class = classify_observation_postcondition(safe)
    expected_class = expected_postcondition(plan)
    if observed_class != expected_class:
        raise RuntimeError(
            "pixel postcondition mismatch before write: "
            f"expected {expected_class}, observed {observed_class}"
        )
    inference_dir = layout[f"inference_{plan.split}"] / plan.sample_id
    truth_path = layout[f"truth_{plan.split}"] / f"{plan.sample_id}.json"
    if inference_dir.exists() or truth_path.exists():
        raise FileExistsError(f"refusing to overwrite sample {plan.sample_id}")

    from PIL import Image

    inference_dir.mkdir(exist_ok=False)
    Image.fromarray(np.asarray(safe["rgb"], dtype=np.uint8)).save(
        inference_dir / "rgb.png"
    )
    np.save(inference_dir / "depth_m.npy", safe["depth_m"].astype(np.float32))
    np.save(inference_dir / "connector_face_mask.npy", safe["connector_face_mask"])
    occlusion = safe["occlusion_mask"]
    encoded_occlusion = (
        np.asarray("OCCLUSION_UNKNOWN") if occlusion is None else occlusion
    )
    np.save(inference_dir / "occlusion_mask.npy", encoded_occlusion)
    with (inference_dir / "intrinsics.json").open("x", encoding="utf-8") as stream:
        json.dump(safe["intrinsics"], stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
    with truth_path.open("x", encoding="utf-8") as stream:
        json.dump(plan.truth_record(), stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
    if {item.name for item in inference_dir.iterdir()} != INFERENCE_SAMPLE_FILES:
        raise RuntimeError("inference directory contains non-observation files")
    return inference_dir, truth_path


def _apply_reject_injection(
    plan: SamplePlan, rgb: np.ndarray, depth_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inject rejection evidence using rendered image coordinates only."""

    rgb_result = np.asarray(rgb, dtype=np.uint8).copy()
    depth_result = np.asarray(depth_m, dtype=np.float64).copy()
    reason = plan.expected_reject_reason
    if reason is None:
        return rgb_result, depth_result
    mask, initial_occlusion = derive_image_masks(rgb_result, depth_result)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise RuntimeError("initial connector face mask is empty; injection refused")
    if initial_occlusion is None:
        raise RuntimeError("initial occlusion state is unknown; injection refused")
    if reason == "CONNECTOR_FACE_OUT_OF_FRAME":
        return rgb_result, depth_result
    center_v, center_u = float(rows.mean()), float(columns.mean())
    grid_v, grid_u = np.indices(mask.shape, dtype=np.float64)
    radial = np.hypot(grid_v - center_v, grid_u - center_u)
    radius = max(1.0, float(radial[mask].max()))

    if reason == "KEY_REGION_OCCLUDED":
        fraction = float(plan.reject_injection["occluded_face_fraction"])
        angle = math.radians(float(plan.reject_injection["angle_deg"]))
        half_size = max(2, round(radius * math.sqrt(fraction)))
        target_u = round(center_u + 0.35 * radius * math.cos(angle))
        target_v = round(center_v + 0.35 * radius * math.sin(angle))
        v0, v1 = max(0, target_v - half_size), min(mask.shape[0], target_v + half_size)
        u0, u1 = max(0, target_u - half_size), min(mask.shape[1], target_u + half_size)
        rgb_result[v0:v1, u0:u1] = (25, 25, 25)
        valid_depth = depth_result[np.isfinite(depth_result) & (depth_result > 0)]
        if valid_depth.size:
            depth_result[v0:v1, u0:u1] = max(0.001, float(valid_depth.min()) - 0.005)
    elif reason == "KEY_REGION_DEPTH_MISSING":
        fraction = float(
            plan.reject_injection["key_region_depth_dropout_fraction"]
        )
        start = math.radians(float(plan.reject_injection["angle_deg"]))
        angle = np.mod(
            np.arctan2(grid_v - center_v, grid_u - center_u) - start,
            2.0 * math.pi,
        )
        depth_result[
            mask & (radial >= 0.65 * radius) & (angle <= 2.0 * math.pi * fraction)
        ] = np.nan
    elif reason == "KEY_REGION_LOW_CONFIDENCE":
        from PIL import Image, ImageEnhance, ImageFilter

        image = ImageEnhance.Contrast(Image.fromarray(rgb_result)).enhance(
            float(plan.reject_injection["contrast_scale"])
        )
        rgb_result = np.asarray(
            image.filter(
                ImageFilter.GaussianBlur(
                    float(plan.reject_injection["gaussian_blur_radius_px"])
                )
            )
        )
    else:
        raise ValueError(f"unsupported rejection reason: {reason}")
    return rgb_result, depth_result


def _light_rgb(kelvin: float) -> tuple[float, float, float]:
    """Small black-body approximation used only for simulation lighting."""

    value = min(40000.0, max(1000.0, kelvin)) / 100.0
    if value <= 66:
        red = 255.0
        green = 99.4708 * math.log(value) - 161.1196
        blue = 0.0 if value <= 19 else 138.5177 * math.log(value - 10) - 305.0448
    else:
        red = 329.6987 * (value - 60) ** -0.1332
        green = 288.1222 * (value - 60) ** -0.0755
        blue = 255.0
    return tuple(float(np.clip(channel, 0, 255) / 255.0) for channel in (red, green, blue))


def _author_scene(stage, plan: SamplePlan, camera, light) -> None:
    """Apply authored values without reading pose, semantic, or physics truth."""

    from pxr import Gf, UsdGeom

    plug = stage.GetPrimAtPath("/World/D38999Shell25JKeyedPublicSpecV2/LoosePlug")
    if not plug.IsValid():
        raise RuntimeError("keyed-v2 loose plug prim is missing")
    translation = plan.authored_pose["translation_m"]
    rotation = plan.authored_pose["rotation_xyz_deg"]
    plug_xform = UsdGeom.Xformable(plug)
    plug_xform.ClearXformOpOrder()
    plug_xform.AddTranslateOp().Set(Gf.Vec3d(translation[0], translation[1], 0.0))
    plug_xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

    axial = float(translation[2])
    camera_x = camera_y = 0.0
    if plan.expected_reject_reason == "CONNECTOR_FACE_OUT_OF_FRAME":
        angle = math.radians(float(plan.reject_injection["angle_deg"]))
        fraction = float(plan.reject_injection["camera_framing_shift_fraction"])
        half_view = axial * math.tan(math.atan(20.955 / (2.0 * 24.0)))
        camera_x = 2.0 * half_view * fraction * math.cos(angle)
        camera_y = 2.0 * half_view * fraction * math.sin(angle)
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(camera_x, camera_y, axial))

    light.CreateIntensityAttr().Set(plan.authored_light["key_light_intensity"])
    light.CreateColorAttr().Set(
        Gf.Vec3f(*_light_rgb(plan.authored_light["color_temperature_k"]))
    )
    light_xform = UsdGeom.Xformable(light)
    light_xform.ClearXformOpOrder()
    light_xform.AddRotateXYZOp().Set(
        Gf.Vec3f(55.0, 0.0, plan.authored_light["azimuth_deg"])
    )


def _generate_with_isaac(
    simulation_app,
    asset_path: Path,
    contract: Mapping[str, Any],
    schedules: Mapping[str, Sequence[SamplePlan]],
    layout: Mapping[str, Path],
    *,
    preflight: bool = False,
) -> dict[str, Any]:
    """Lazy renderer: the only requested annotators are RGB and planar depth."""

    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    context = omni.usd.get_context()
    if context.open_stage(str(asset_path)) is not True:
        raise RuntimeError(f"could not open asset: {asset_path}")
    for _ in range(3):
        simulation_app.update()
    stage = context.get_stage()
    fixed = stage.GetPrimAtPath(
        "/World/D38999Shell25JKeyedPublicSpecV2/FixedReceptacle"
    )
    if not fixed.IsValid():
        raise RuntimeError("keyed-v2 fixed receptacle prim is missing")
    fixed.SetActive(False)

    camera_config = contract["camera"]
    camera = UsdGeom.Camera.Define(stage, camera_config["prim_path"])
    camera_prim = camera.GetPrim()
    if not camera_prim.IsValid() or str(camera_prim.GetPath()) != (
        camera_config["prim_path"]
    ):
        raise RuntimeError("dataset camera prim is invalid")
    camera.CreateFocalLengthAttr(camera_config["focal_length_mm"])
    camera.CreateHorizontalApertureAttr(camera_config["horizontal_aperture_mm"])
    camera.CreateVerticalApertureAttr(camera_config["vertical_aperture_mm"])
    camera.CreateClippingRangeAttr(Gf.Vec2f(*camera_config["clipping_range_m"]))
    light = UsdLux.DistantLight.Define(stage, "/World/KeyedV2YawDatasetKeyLight")

    runtime = contract["runtime_capture"]
    expected_shape = (
        int(camera_config["resolution_px"][1]),
        int(camera_config["resolution_px"][0]),
    )
    render_product = None
    annotators = []
    generated = 0
    postcondition_counts = {name: 0 for name in POSTCONDITION_CLASSES}
    try:
        render_product = rep.create.render_product(
            camera_prim,
            tuple(camera_config["resolution_px"]),
            name=camera_config["render_product_name"],
        )
        render_product_path = render_product.path
        if render_product_path is None or not str(render_product_path):
            raise RuntimeError("dataset render product path is invalid")
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        annotators.extend((rgb_annotator, depth_annotator))
        for annotator in annotators:
            annotator.attach([render_product_path])
        for _ in range(runtime["app_updates_after_annotator_attach"]):
            simulation_app.update()
        for _ in range(runtime["replicator_warmup_frames"]):
            rep.orchestrator.step(
                rt_subframes=runtime["rt_subframes"],
                delta_time=0.0,
                pause_timeline=True,
            )

        for split, plans in schedules.items():
            for plan in plans:
                _author_scene(stage, plan, camera_prim, light)
                for _ in range(runtime["per_sample_render_frames"]):
                    rep.orchestrator.step(
                        rt_subframes=runtime["rt_subframes"],
                        delta_time=0.0,
                        pause_timeline=True,
                    )
                rgba_data = rgb_annotator.get_data()
                depth_data = depth_annotator.get_data()
                if rgba_data is None:
                    raise RuntimeError("RGB annotator returned no frame")
                if depth_data is None:
                    raise RuntimeError("depth annotator returned no frame")
                rgba = np.asarray(rgba_data)
                depth = np.asarray(depth_data, dtype=np.float64)
                if (
                    rgba.ndim != 3
                    or rgba.shape[:2] != expected_shape
                    or rgba.shape[2] < 3
                    or rgba.size == 0
                ):
                    raise RuntimeError("RGB annotator returned an invalid shape")
                if depth.shape != expected_shape or depth.size == 0:
                    raise RuntimeError("depth annotator returned an invalid shape")
                rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
                if not np.any(rgb):
                    raise RuntimeError("RGB annotator returned an empty image")
                valid_depth = np.isfinite(depth) & (depth > 0.0)
                if not np.any(valid_depth):
                    raise RuntimeError("depth annotator returned no positive finite depth")
                exposure = 2.0 ** float(plan.authored_light["exposure_ev"])
                rgb = np.clip(
                    np.rint(rgb.astype(np.float64) * exposure), 0, 255
                ).astype(np.uint8)
                rgb, depth = _apply_reject_injection(plan, rgb, depth)
                face_mask, occlusion_mask = derive_image_masks(rgb, depth)
                observation = {
                    "rgb": rgb,
                    "depth_m": depth,
                    "connector_face_mask": face_mask,
                    "occlusion_mask": occlusion_mask,
                    "intrinsics": camera_intrinsics(contract),
                }
                observed_class = classify_observation_postcondition(observation)
                if observed_class != expected_postcondition(plan):
                    _raise_pixel_postcondition_mismatch(
                        layout,
                        plan,
                        observation,
                        observed_class,
                        preflight=preflight,
                    )
                write_separated_sample(
                    layout,
                    plan,
                    observation,
                )
                postcondition_counts[observed_class] += 1
                generated += 1
    finally:
        if render_product is not None:
            render_product_path = render_product.path
            for annotator in annotators:
                try:
                    annotator.detach([render_product_path])
                except Exception:
                    pass
            try:
                render_product.destroy()
            except Exception:
                pass
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_samples": generated,
        "pixel_postcondition_counts": postcondition_counts,
        "annotators": ["rgb", "distance_to_image_plane"],
        "truth_and_inference_directories_separate": True,
        **{field: False for field in CONTROL_FIELDS},
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir")
    arguments = parser.parse_args(argv)
    if not arguments.run:
        parser.error("dataset generation requires explicit --run")
    if arguments.preflight and arguments.split != "all":
        parser.error("--preflight cannot be combined with --split")
    return arguments


def _sample_ids_on_disk(
    layout: Mapping[str, Path], split: str
) -> tuple[set[str], set[str]]:
    inference_ids = {
        path.name
        for path in layout[f"inference_{split}"].iterdir()
        if path.is_dir()
    }
    truth_ids = {
        path.stem
        for path in layout[f"truth_{split}"].iterdir()
        if path.is_file() and path.suffix == ".json"
    }
    return inference_ids, truth_ids


def _completed_sample_count(
    layout: Mapping[str, Path], splits: Sequence[str]
) -> int:
    total = 0
    for split in splits:
        inference_ids, truth_ids = _sample_ids_on_disk(layout, split)
        total += len(inference_ids & truth_ids)
    return total


def _validate_generation_success(
    report: Mapping[str, Any],
    schedules: Mapping[str, Sequence[SamplePlan]],
    layout: Mapping[str, Path],
    *,
    preflight: bool,
) -> None:
    expected_total = sum(len(plans) for plans in schedules.values())
    if report.get("generated_samples") != expected_total:
        raise RuntimeError("reported generated sample count is incomplete")
    for split, plans in schedules.items():
        expected_ids = {plan.sample_id for plan in plans}
        inference_ids, truth_ids = _sample_ids_on_disk(layout, split)
        if inference_ids != expected_ids or truth_ids != expected_ids:
            raise RuntimeError("on-disk inference/truth samples are incomplete")
        for sample_id in expected_ids:
            sample_dir = layout[f"inference_{split}"] / sample_id
            files = {path.name for path in sample_dir.iterdir() if path.is_file()}
            if files != INFERENCE_SAMPLE_FILES:
                raise RuntimeError("on-disk observation file set is incomplete")
            if any((sample_dir / name).stat().st_size <= 0 for name in files):
                raise RuntimeError("on-disk observation file is empty")
            truth_path = layout[f"truth_{split}"] / f"{sample_id}.json"
            if truth_path.stat().st_size <= 0:
                raise RuntimeError("on-disk truth record is empty")
    if _completed_sample_count(layout, tuple(schedules)) != expected_total:
        raise RuntimeError("empty or partial output directories cannot pass")
    for field in CONTROL_FIELDS:
        if report.get(field) is not False:
            raise RuntimeError(f"success report attempted to promote {field}")
    counts = _mapping(
        report.get("pixel_postcondition_counts"), "pixel_postcondition_counts"
    )
    if sum(_integer(value, name) for name, value in counts.items()) != expected_total:
        raise RuntimeError("pixel postcondition counts do not cover every sample")
    if preflight:
        if expected_total != 5:
            raise RuntimeError("preflight must generate exactly five samples")
        if set(counts) != set(POSTCONDITION_CLASSES) or any(
            counts.get(name) != 1 for name in POSTCONDITION_CLASSES
        ):
            raise RuntimeError("preflight must pass each pixel postcondition once")


def _write_generation_failure(
    root: Path,
    error: BaseException,
    *,
    generated_samples: int,
    selected_mode: str,
    selected_split: str,
) -> Path:
    report = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "GENERATION_FAILED",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "generated_samples": int(generated_samples),
        "selected_mode": selected_mode,
        "selected_split": selected_split,
        "control_authorized": False,
        "selected_for_control_allowed": False,
        "simulation_insertion_control_authorized": False,
        "robot_control_authorized": False,
        "hardware_control_authorized": False,
    }
    if tuple(report) != FAILURE_REPORT_FIELDS:
        raise RuntimeError("internal failure report fields differ from contract")
    path = root / "generation_failure.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    app = None
    layout = None
    split_names: tuple[str, ...] = ()
    mode = "PREFLIGHT" if arguments.preflight else "DATASET"
    selected_split = "preflight" if arguments.preflight else arguments.split
    completed = False
    try:
        repository = Path(__file__).resolve().parents[3]
        contract = load_contract(arguments.config)
        if arguments.preflight:
            schedules = build_preflight_schedule(contract)
            split_names = tuple(schedules)
        else:
            split_names = selected_splits(arguments.split)
            schedules = {
                split: build_split_schedule(contract, split)
                for split in split_names
            }
        output_root = (
            Path(arguments.output_dir)
            if arguments.output_dir is not None
            else default_output_root(
                repository, arguments.split, preflight=arguments.preflight
            )
        )
        output_root = require_new_output_root(output_root)
        asset_path = (repository / contract["identity"]["asset"]).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(f"keyed-v2 asset is missing: {asset_path}")

        layout = prepare_output_layout(output_root, contract, split_names)
        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": not arguments.gui,
                "multi_gpu": False,
                "active_gpu": 0,
                "physics_gpu": 0,
            }
        )
        report = _generate_with_isaac(
            app,
            asset_path,
            contract,
            schedules,
            layout,
            preflight=arguments.preflight,
        )
        _validate_generation_success(
            report, schedules, layout, preflight=arguments.preflight
        )
        if arguments.preflight:
            report["preflight_passed"] = True
        report["mode"] = mode
        report["selected_split"] = selected_split
        report["generated_splits"] = list(split_names)
        report["schedule"] = summarize_schedule(schedules)
        report_path = layout["root"] / "generation_report.json"
        with report_path.open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        if not report_path.is_file() or report_path.stat().st_size <= 0:
            raise RuntimeError("generation success report was not created")
        completed = True
        return 0
    except BaseException as error:
        traceback.print_exc()
        if layout is not None:
            generated = _completed_sample_count(layout, split_names)
            try:
                _write_generation_failure(
                    layout["root"],
                    error,
                    generated_samples=generated,
                    selected_mode=mode,
                    selected_split=selected_split,
                )
            except BaseException:
                traceback.print_exc()
        return 1
    finally:
        if app is not None:
            app.close(exit_code=0 if completed else 1)


if __name__ == "__main__":
    raise SystemExit(main())
