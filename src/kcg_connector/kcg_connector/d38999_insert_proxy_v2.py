"""Validated contract helpers for the D38999 insertion-contact proxy V2.

This module is simulator-free.  It deliberately keeps the V2 insertion proxy
separate from the frozen public-dimensional V1 regression asset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_d38999_insert_proxy_v2"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_insert_proxy_v2.yaml"
)
RECOMMENDED_ASSET_NAME = "d38999_insert_proxy_v2.usda"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _triple(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    result = tuple(_positive(item, label) for item in value)
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class PlugGeometry:
    body_radius: float
    body_length: float
    guide_outer_radius: float
    guide_inner_radius: float
    guide_length: float
    nose_radius: float
    nose_chamfer_length: float
    coupling_nut_inner_radius: float
    coupling_nut_outer_radius: float
    coupling_nut_length: float
    c2_key_outer_radius: float
    c2_key_tangential_width: float
    c2_key_length: float
    c2_key_start_from_tip: float


@dataclass(frozen=True)
class ReceptacleGeometry:
    shell_outer_radius: float
    guide_bore_radius: float
    mouth_bore_radius: float
    entrance_chamfer_length: float
    guide_length: float
    rear_body_radius: float
    rear_body_length: float
    flange_side: float
    flange_thickness: float
    collision_segment_count: int
    c2_channel_half_width_rad: float


@dataclass(frozen=True)
class Physics:
    plug_body_mass_kg: float
    coupling_nut_mass_kg: float
    plug_body_diagonal_inertia_kg_m2: tuple[float, float, float]
    coupling_nut_diagonal_inertia_kg_m2: tuple[float, float, float]
    static_friction: float
    dynamic_friction: float
    restitution: float
    compliant_contact_stiffness_n_m: float
    compliant_contact_damping_n_s_m: float
    contact_offset_m: float
    rest_offset_m: float
    solver_position_iterations: int
    solver_velocity_iterations: int


@dataclass(frozen=True)
class InsertProxyV2:
    path: Path
    sha256: str
    document: Mapping[str, Any]
    plug: PlugGeometry
    receptacle: ReceptacleGeometry
    physics: Physics
    radial_clearance: float
    key_side_clearance: float
    insertion_depth: float
    preinsert_gap: float
    c2_symmetry_order: int
    gate_fraction: float


def _dataclass_from_numbers(cls, value: Any, label: str):
    document = _mapping(value, label)
    expected = set(cls.__dataclass_fields__)
    if set(document) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected-set(document))}, "
            f"unexpected={sorted(set(document)-expected)}"
        )
    parsed = {}
    for name in cls.__dataclass_fields__:
        item = document[name]
        if name == "collision_segment_count":
            if isinstance(item, bool) or not isinstance(item, int) or item < 16:
                raise ValueError(f"{label}.{name} must be an integer >= 16")
            parsed[name] = int(item)
        else:
            parsed[name] = _positive(item, f"{label}.{name}")
    return cls(**parsed)


def load_insert_proxy_v2(path: Path | str = DEFAULT_CONFIG_PATH) -> InsertProxyV2:
    config_path = Path(path).expanduser().resolve()
    raw = config_path.read_bytes()
    document = _mapping(yaml.safe_load(raw), "document")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    identity = _mapping(document.get("identity"), "identity")
    if identity.get("proxy_id") != "d38999_insert_proxy_v2":
        raise ValueError("identity.proxy_id is invalid")
    if identity.get("source_proxy_id") != "d38999_shell25j_61_pair_proxy_v1":
        raise ValueError("V2 must identify the frozen V1 source proxy")
    if identity.get("certification_claim") != "none":
        raise ValueError("V2 cannot claim certification")
    labels = identity.get("required_labels")
    if labels != ["PROXY THREAD", "PROXY LOCK"]:
        raise ValueError("V2 must retain explicit proxy thread/lock labels")

    geometry = _mapping(document.get("geometry_m"), "geometry_m")
    plug = _dataclass_from_numbers(PlugGeometry, geometry.get("plug"), "plug")
    receptacle = _dataclass_from_numbers(
        ReceptacleGeometry, geometry.get("receptacle"), "receptacle"
    )
    radial_clearance = _positive(
        geometry.get("radial_clearance"), "geometry_m.radial_clearance"
    )
    measured_clearance = receptacle.guide_bore_radius - plug.guide_outer_radius
    if not math.isclose(measured_clearance, radial_clearance, abs_tol=1.0e-12):
        raise ValueError("declared radial clearance differs from radii")
    if not plug.nose_radius < plug.guide_outer_radius:
        raise ValueError("plug nose must form an inward lead chamfer")
    if not receptacle.mouth_bore_radius > receptacle.guide_bore_radius:
        raise ValueError("receptacle mouth must form an outward entry chamfer")
    if not plug.c2_key_outer_radius > receptacle.guide_bore_radius:
        raise ValueError("C2 key must interfere outside the guide channels")
    c2_order = geometry.get("c2_symmetry_order")
    if c2_order != 2:
        raise ValueError("V2 guide geometry must remain C2")

    physics_doc = _mapping(document.get("physics"), "physics")
    expected_physics = set(Physics.__dataclass_fields__)
    if set(physics_doc) != expected_physics:
        raise ValueError("physics keys differ from V2 schema")
    physics = Physics(
        plug_body_mass_kg=_positive(physics_doc["plug_body_mass_kg"], "mass"),
        coupling_nut_mass_kg=_positive(
            physics_doc["coupling_nut_mass_kg"], "nut mass"
        ),
        plug_body_diagonal_inertia_kg_m2=_triple(
            physics_doc["plug_body_diagonal_inertia_kg_m2"], "plug inertia"
        ),
        coupling_nut_diagonal_inertia_kg_m2=_triple(
            physics_doc["coupling_nut_diagonal_inertia_kg_m2"], "nut inertia"
        ),
        static_friction=_finite(physics_doc["static_friction"], "static friction"),
        dynamic_friction=_finite(
            physics_doc["dynamic_friction"], "dynamic friction"
        ),
        restitution=_finite(physics_doc["restitution"], "restitution"),
        compliant_contact_stiffness_n_m=_positive(
            physics_doc["compliant_contact_stiffness_n_m"], "contact stiffness"
        ),
        compliant_contact_damping_n_s_m=_positive(
            physics_doc["compliant_contact_damping_n_s_m"], "contact damping"
        ),
        contact_offset_m=_positive(physics_doc["contact_offset_m"], "contact offset"),
        rest_offset_m=_finite(physics_doc["rest_offset_m"], "rest offset"),
        solver_position_iterations=int(physics_doc["solver_position_iterations"]),
        solver_velocity_iterations=int(physics_doc["solver_velocity_iterations"]),
    )
    if not 0.0 <= physics.dynamic_friction <= physics.static_friction:
        raise ValueError("friction values are invalid")
    if not 0.0 <= physics.restitution <= 1.0:
        raise ValueError("restitution is invalid")
    if min(physics.solver_position_iterations, physics.solver_velocity_iterations) < 1:
        raise ValueError("solver iterations must be positive")

    mechanism = _mapping(document.get("proxy_mechanism"), "proxy_mechanism")
    if mechanism.get("fine_thread_teeth_modeled") is not False:
        raise ValueError("V2 must not claim modeled fine thread teeth")
    if mechanism.get("labels") != {
        "thread": "PROXY THREAD",
        "lock": "PROXY LOCK",
    }:
        raise ValueError("proxy mechanism labels are invalid")
    boundaries = _mapping(document.get("boundaries"), "boundaries")
    forbidden_true = (
        "old_proxy_modified",
        "real_thread_claimed",
        "real_lock_claimed",
        "real_d38999_certification_claimed",
        "object_truth_allowed_for_control",
        "physx_contact_normal_allowed_for_control",
        "collider_identity_allowed_for_control",
    )
    if any(boundaries.get(name) is not False for name in forbidden_true):
        raise ValueError("V2 boundary must keep all forbidden claims/inputs false")
    sweep = _mapping(document.get("tolerance_sweep"), "tolerance_sweep")
    if sweep.get("controller_truth_inputs") != []:
        raise ValueError("tolerance controller cannot receive truth inputs")
    gate_fraction = _positive(
        sweep.get("conservative_gate_fraction_of_measured_failure_boundary"),
        "gate fraction",
    )
    if not 0.5 <= gate_fraction <= 0.8:
        raise ValueError("gate fraction must remain in [0.5, 0.8]")
    return InsertProxyV2(
        path=config_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        document=document,
        plug=plug,
        receptacle=receptacle,
        physics=physics,
        radial_clearance=radial_clearance,
        key_side_clearance=_positive(
            geometry.get("key_side_clearance"), "key side clearance"
        ),
        insertion_depth=_positive(
            geometry.get("insertion_depth"), "insertion depth"
        ),
        preinsert_gap=_positive(geometry.get("preinsert_gap"), "preinsert gap"),
        c2_symmetry_order=int(c2_order),
        gate_fraction=gate_fraction,
    )


def safe_new_output(path: Path | str) -> Path:
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".usd", ".usda"}:
        raise ValueError("V2 asset output must be .usd or .usda")
    if output.name == "d38999_shell25j_61_pair_proxy_v1.usda":
        raise ValueError("the frozen V1 asset cannot be overwritten")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    return output
