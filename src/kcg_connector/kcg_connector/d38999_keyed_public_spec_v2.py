"""Pure-CPU contract for the public-specification D38999 keyed-v2 model.

The model is intentionally a simulation identity beside the frozen C2
proxies.  It transcribes public interface dimensions and the 25-61 contact
table, but it never claims manufacturer CAD, hardware calibration, or space
qualification.  Importing this module does not import Isaac Sim or Pixar USD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_d38999_keyed_public_spec_v2_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_public_spec_v2.yaml"
)
RECOMMENDED_ASSET_NAME = (
    "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
)
PAIR_MODEL_ID = "d38999_shell25j_25_61_n_keyed_public_spec_pair_v2"
PLUG_MODEL_ID = "d38999_26kj61sn_keyed_proxy_v2"
RECEPTACLE_MODEL_ID = "d38999_20kj61pn_keyed_proxy_v2"
ROOT_PRIM = "/World/D38999Shell25JKeyedPublicSpecV2"
INCH_TO_MM = 25.4
EXPECTED_KEY_ANGLES_DEG = (0.0, 80.0, 142.0, 196.0, 293.0)
EXPECTED_CONTACT_LABELS = (
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a",
    "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "AA",
    "BB", "CC", "DD", "EE", "FF", "GG", "HH", "JJ", "KK", "LL",
    "MM", "NN", "PP",
)
FORBIDDEN_OUTPUT_NAMES = {
    "connector_pair.usda",
    "d38999_shell25j_61_pair_proxy_v1.usda",
    "d38999_insert_proxy_v2.usda",
}
R4_MASS_PROPERTY_SOURCE_REVISION = "keyed_v2_public_b_rear_r4"
MASS_PROPERTY_SOURCE_KIND = (
    "frozen_r4_physx_autocompute_snapshot_simulation_assumption_not_public_spec"
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _vector(
    value: Any, label: str, length: int, *, positive: bool = False
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    result = tuple(
        _positive(item, f"{label}[{index}]")
        if positive
        else _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result


def keyed_yaw_peak_to_peak_clearance_deg(
    *, slot_width_mm: float, key_width_mm: float, radius_mm: float
) -> float:
    """Return the exact centred rectangular-key yaw window.

    The key centre lies on ``radius_mm``.  At yaw ``delta`` its tangential
    envelope is ``r*sin(delta) + key_width/2*cos(delta)``.  The returned
    value is the full negative-to-positive window, not a one-sided limit.
    """

    slot = _positive(slot_width_mm, "slot_width_mm")
    key = _positive(key_width_mm, "key_width_mm")
    radius = _positive(radius_mm, "radius_mm")
    if slot <= key:
        raise ValueError("slot_width_mm must be greater than key_width_mm")
    hypotenuse = math.hypot(radius, 0.5 * key)
    ratio = (0.5 * slot) / hypotenuse
    if not 0.0 < ratio < 1.0:
        raise ValueError("key/slot geometry cannot produce a finite yaw window")
    half_angle = math.asin(ratio) - math.atan2(0.5 * key, radius)
    if half_angle <= 0.0:
        raise ValueError("key/slot geometry has no positive yaw window")
    return math.degrees(2.0 * half_angle)


def rectangular_key_fits(
    *,
    yaw_deg: float,
    slot_width_mm: float,
    key_width_mm: float,
    radius_mm: float,
) -> bool:
    """Pure same-slot cross-section check used by tolerance sweeps.

    The tangential-envelope formula is local to one radial slot.  A key that
    has rotated beyond 90 degrees is on the opposite half of the connector
    and therefore cannot fit this slot even if its tangential projection is
    numerically small again.
    """

    yaw = math.radians(abs(_finite(yaw_deg, "yaw_deg")))
    slot = _positive(slot_width_mm, "slot_width_mm")
    key = _positive(key_width_mm, "key_width_mm")
    radius = _positive(radius_mm, "radius_mm")
    if yaw > 0.5 * math.pi:
        return False
    tangential_envelope = radius * math.sin(yaw) + 0.5 * key * math.cos(yaw)
    return bool(tangential_envelope <= 0.5 * slot + 1.0e-12)


def _wrapped_angle_error_deg(first: float, second: float) -> float:
    return (float(first) - float(second) + 180.0) % 360.0 - 180.0


def keyed_pattern_fits_at_yaw(
    yaw_deg: float,
    *,
    plug_key_angles_deg: tuple[float, ...] = EXPECTED_KEY_ANGLES_DEG,
    receptacle_keyway_angles_deg: tuple[float, ...] = EXPECTED_KEY_ANGLES_DEG,
    plug_key_widths_mm: tuple[float, ...] = (2.54, 1.32, 1.32, 1.32, 1.32),
    receptacle_keyway_widths_mm: tuple[float, ...] = (
        3.20,
        1.60,
        1.60,
        1.60,
        1.60,
    ),
    radius_mm: float = 18.855,
) -> bool:
    """Return whether all five keys have a unique compatible keyway.

    This is a two-dimensional preflight model, not a replacement for the USD
    collision sweep.  It is deliberately general enough to compare N against
    alternate A-E keyway patterns and to reject the old 180-degree C2 branch.
    """

    yaw = _finite(yaw_deg, "yaw_deg")
    radius = _positive(radius_mm, "radius_mm")
    sequences = (
        plug_key_angles_deg,
        receptacle_keyway_angles_deg,
        plug_key_widths_mm,
        receptacle_keyway_widths_mm,
    )
    if any(not isinstance(value, tuple) or len(value) != 5 for value in sequences):
        raise ValueError("five-key pattern inputs must be tuples of length five")
    plug_angles = tuple(_finite(value, "plug key angle") for value in plug_key_angles_deg)
    slot_angles = tuple(
        _finite(value, "receptacle keyway angle")
        for value in receptacle_keyway_angles_deg
    )
    key_widths = tuple(_positive(value, "plug key width") for value in plug_key_widths_mm)
    slot_widths = tuple(
        _positive(value, "receptacle keyway width")
        for value in receptacle_keyway_widths_mm
    )
    if len({round(value % 360.0, 12) for value in plug_angles}) != 5:
        raise ValueError("plug key angles must be unique")
    if len({round(value % 360.0, 12) for value in slot_angles}) != 5:
        raise ValueError("receptacle keyway angles must be unique")

    candidates: list[tuple[int, ...]] = []
    for key_angle, key_width in zip(plug_angles, key_widths):
        matching_slots = []
        rotated_key_angle = key_angle + yaw
        for slot_index, (slot_angle, slot_width) in enumerate(
            zip(slot_angles, slot_widths)
        ):
            error = _wrapped_angle_error_deg(rotated_key_angle, slot_angle)
            if rectangular_key_fits(
                yaw_deg=error,
                slot_width_mm=slot_width,
                key_width_mm=key_width,
                radius_mm=radius,
            ):
                matching_slots.append(slot_index)
        if not matching_slots:
            return False
        candidates.append(tuple(matching_slots))

    # Five features make exhaustive one-to-one matching tiny and easier to
    # audit than introducing a graph dependency.
    def assign(key_index: int, used_slots: frozenset[int]) -> bool:
        if key_index == 5:
            return True
        return any(
            slot_index not in used_slots
            and assign(key_index + 1, used_slots | {slot_index})
            for slot_index in candidates[key_index]
        )

    return assign(0, frozenset())


@dataclass(frozen=True)
class ContactPosition:
    label: str
    x_in: float
    y_in: float

    @property
    def pin_front_mm(self) -> tuple[float, float]:
        return (self.x_in * INCH_TO_MM, self.y_in * INCH_TO_MM)

    @property
    def socket_front_same_screen_axes_mm(self) -> tuple[float, float]:
        x_mm, y_mm = self.pin_front_mm
        return (-x_mm, y_mm)


@dataclass(frozen=True)
class ClearanceProfile:
    name: str
    slot_width_mm: float
    key_width_mm: float
    radius_mm: float
    peak_to_peak_deg: float
    required_p95_deg: float
    derivation_kind: str
    drawing_specified_clearance: bool


@dataclass(frozen=True)
class MassPropertiesAssumption:
    """Explicit simulation-only mass properties in one rigid-body frame."""

    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PublicSpecKeyedV2:
    path: Path
    document: Mapping[str, Any]
    contacts: tuple[ContactPosition, ...]
    clearance_profiles: tuple[ClearanceProfile, ...]
    simulation_acceptance_profile: str
    mass_property_source_kind: str
    mass_property_source_asset_revision: str
    body_mass_properties: MassPropertiesAssumption
    nut_mass_properties: MassPropertiesAssumption

    @property
    def key_angles_deg(self) -> tuple[float, ...]:
        keying = _mapping(self.document["keying"], "keying")
        return (
            float(keying["main_key_angle_deg_bsc"]),
            *(float(value) for value in keying["minor_key_angles_deg_bsc"]),
        )

    def clearance_profile(self, name: str) -> ClearanceProfile:
        for profile in self.clearance_profiles:
            if profile.name == name:
                return profile
        raise KeyError(name)


def _load_contacts(document: Mapping[str, Any]) -> tuple[ContactPosition, ...]:
    pattern = _mapping(document.get("contact_pattern"), "contact_pattern")
    if pattern.get("controlling_unit") != "inch":
        raise ValueError("contact positions must retain controlling inch values")
    if pattern.get("view") != "front_face_of_pin_insert":
        raise ValueError("contact table must use the pin-front view")
    raw_positions = pattern.get("positions_in")
    if not isinstance(raw_positions, list):
        raise ValueError("contact_pattern.positions_in must be a list")
    contacts: list[ContactPosition] = []
    for index, item in enumerate(raw_positions):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"contact position {index} must be [label, x, y]")
        label = item[0]
        if not isinstance(label, str) or not label:
            raise ValueError(f"contact position {index} label is invalid")
        contacts.append(
            ContactPosition(
                label=label,
                x_in=_finite(item[1], f"contact {label} x"),
                y_in=_finite(item[2], f"contact {label} y"),
            )
        )
    labels = tuple(item.label for item in contacts)
    if labels != EXPECTED_CONTACT_LABELS:
        raise ValueError("25-61 contact labels/order differ from MIL-STD-1560")
    coordinate_pairs = {(item.x_in, item.y_in) for item in contacts}
    if len(coordinate_pairs) != 61:
        raise ValueError("25-61 contact coordinates must contain 61 unique points")
    return tuple(contacts)


def _load_clearance_profiles(
    document: Mapping[str, Any],
) -> tuple[tuple[ClearanceProfile, ...], str]:
    profiles_doc = _mapping(
        document.get("yaw_clearance_profiles"), "yaw_clearance_profiles"
    )
    names = ("nominal_centered", "tight_size_centered", "adversarial_gdt_stress")
    profiles: list[ClearanceProfile] = []
    for name in names:
        value = _mapping(profiles_doc.get(name), f"yaw_clearance_profiles.{name}")
        slot = _positive(value.get("slot_width_mm"), f"{name}.slot_width_mm")
        key = _positive(value.get("key_width_mm"), f"{name}.key_width_mm")
        radius = _positive(value.get("radius_mm"), f"{name}.radius_mm")
        derived = keyed_yaw_peak_to_peak_clearance_deg(
            slot_width_mm=slot, key_width_mm=key, radius_mm=radius
        )
        declared = _positive(
            value.get("derived_peak_to_peak_deg"), f"{name}.derived_peak_to_peak_deg"
        )
        p95 = _positive(
            value.get("required_p95_strictly_below_deg"),
            f"{name}.required_p95_strictly_below_deg",
        )
        derivation_kind = value.get("derivation_kind")
        if not isinstance(derivation_kind, str) or not derivation_kind.strip():
            raise ValueError(f"{name}.derivation_kind must be non-empty text")
        drawing_specified = value.get(
            "drawing_specified_mechanical_yaw_clearance"
        )
        if drawing_specified is not False:
            raise ValueError(
                f"{name} must not claim a drawing-specified mechanical yaw clearance"
            )
        if not math.isclose(derived, declared, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} declared clearance differs from geometry")
        if not math.isclose(0.5 * declared, p95, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} p95 threshold must equal half the full yaw window")
        profiles.append(
            ClearanceProfile(
                name=name,
                slot_width_mm=slot,
                key_width_mm=key,
                radius_mm=radius,
                peak_to_peak_deg=declared,
                required_p95_deg=p95,
                derivation_kind=derivation_kind,
                drawing_specified_clearance=False,
            )
        )
    selected = profiles_doc.get("simulation_acceptance_profile")
    if selected not in names:
        raise ValueError("simulation acceptance profile must name a declared profile")
    if profiles_doc.get("real_measured_clearance_deg", "missing") is not None:
        raise ValueError("simulation-only model cannot invent measured hardware clearance")
    return tuple(profiles), selected


def _load_mass_body(
    value: Any, label: str
) -> MassPropertiesAssumption:
    document = _mapping(value, label)
    expected = {
        "mass_kg",
        "center_of_mass_m",
        "diagonal_inertia_kg_m2",
        "principal_axes_xyzw",
    }
    if set(document) != expected:
        raise ValueError(f"{label} mass-property keys are invalid")
    inertia = _vector(
        document["diagonal_inertia_kg_m2"],
        f"{label}.diagonal_inertia_kg_m2",
        3,
        positive=True,
    )
    if any(
        inertia[index] + inertia[(index + 1) % 3]
        < inertia[(index + 2) % 3] - 1.0e-12
        for index in range(3)
    ):
        raise ValueError(f"{label} inertia violates the triangle inequality")
    principal = _vector(
        document["principal_axes_xyzw"],
        f"{label}.principal_axes_xyzw",
        4,
    )
    if not math.isclose(
        math.sqrt(sum(item * item for item in principal)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(f"{label} principal axes must be a unit quaternion")
    return MassPropertiesAssumption(
        mass_kg=_positive(document["mass_kg"], f"{label}.mass_kg"),
        center_of_mass_m=_vector(
            document["center_of_mass_m"],
            f"{label}.center_of_mass_m",
            3,
        ),
        diagonal_inertia_kg_m2=inertia,
        principal_axes_xyzw=principal,
    )


def _load_simulation_mass_properties(document: Mapping[str, Any]):
    value = _mapping(
        document.get("simulation_mass_properties"),
        "simulation_mass_properties",
    )
    expected = {
        "source_kind",
        "source_asset_revision",
        "exact_hardware_mass_properties_available",
        "runtime_randomization_enabled_for_checkpoint_a",
        "body_assembly",
        "coupling_nut",
    }
    if set(value) != expected:
        raise ValueError("simulation_mass_properties keys are invalid")
    if value.get("source_kind") != MASS_PROPERTY_SOURCE_KIND:
        raise ValueError("mass properties must retain the frozen r4 source kind")
    revision = value.get("source_asset_revision")
    if revision != R4_MASS_PROPERTY_SOURCE_REVISION:
        raise ValueError("mass-property source asset revision differs from r4")
    if value.get("exact_hardware_mass_properties_available") is not False:
        raise ValueError("simulation mass properties cannot claim hardware truth")
    if value.get("runtime_randomization_enabled_for_checkpoint_a") is not False:
        raise ValueError("checkpoint A mass properties must remain deterministic")
    body = _load_mass_body(value.get("body_assembly"), "body_assembly")
    nut = _load_mass_body(value.get("coupling_nut"), "coupling_nut")
    if not math.isclose(body.mass_kg, 0.23, abs_tol=1.0e-12):
        raise ValueError("body simulation mass must remain 0.23 kg")
    if not math.isclose(nut.mass_kg, 0.08, abs_tol=1.0e-12):
        raise ValueError("nut simulation mass must remain 0.08 kg")
    return value["source_kind"], revision, body, nut


def load_keyed_public_spec_v2(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> PublicSpecKeyedV2:
    config_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("enabled_for_simulation") is not True:
        raise ValueError("public-spec keyed-v2 must be explicitly enabled for simulation")

    identity = _mapping(document.get("identity"), "identity")
    expected_ids = {
        "pair_model_id": PAIR_MODEL_ID,
        "loose_plug_model_id": PLUG_MODEL_ID,
        "fixed_receptacle_model_id": RECEPTACLE_MODEL_ID,
    }
    for field, expected in expected_ids.items():
        if identity.get(field) != expected:
            raise ValueError(f"identity.{field} must be {expected}")
    if identity.get("selected_class") != "K":
        raise ValueError("this vendor-catalog pair selects class K")
    if identity.get("space_grade_claimed") is not False:
        raise ValueError("class K must not be represented as space grade")
    if identity.get("real_hardware_identity_claimed") is not False:
        raise ValueError("simulation model must not claim a verified physical unit")

    dimensions = _mapping(
        document.get("interface_dimensions_mm"), "interface_dimensions_mm"
    )
    plug_dimensions = _mapping(dimensions.get("plug"), "interface plug")
    rear_basic = _positive(
        plug_dimensions.get("b_rear_shell_diameter_basic"),
        "plug B rear-shell diameter",
    )
    rear_plus = _finite(
        plug_dimensions.get("b_rear_shell_diameter_plus"),
        "plug B rear-shell plus tolerance",
    )
    rear_minus = _finite(
        plug_dimensions.get("b_rear_shell_diameter_minus"),
        "plug B rear-shell minus tolerance",
    )
    if (rear_basic, rear_plus, rear_minus) != (44.30, 0.20, 0.00):
        raise ValueError(
            "plug shell-25 B diameter must retain the /26G public values"
        )

    keying = _mapping(document.get("keying"), "keying")
    angles = (
        _finite(keying.get("main_key_angle_deg_bsc"), "main key angle"),
        *(
            _finite(value, "minor key angle")
            for value in keying.get("minor_key_angles_deg_bsc", [])
        ),
    )
    if angles != EXPECTED_KEY_ANGLES_DEG:
        raise ValueError("N polarization must contain the public five-key angles")
    if keying.get("polarization_must_precede_coupling_and_contact") is not True:
        raise ValueError("keying must engage before coupling and electrical contacts")
    if _positive(keying.get("plug_key_axial_length_min"), "plug key length") < 7.24:
        raise ValueError("plug collision key must retain the public 7.24 mm minimum length")

    authorization = _mapping(document.get("authorization"), "authorization")
    forbidden_true = (
        "selected_for_control_allowed",
        "simulation_insertion_control_authorized",
        "robot_control_authorized",
        "hardware_control_authorized",
        "real_hardware_fidelity_claimed",
        "space_qualification_claimed",
        "real_assembly_success_claimed",
    )
    if any(authorization.get(name) is not False for name in forbidden_true):
        raise ValueError("public-spec keyed-v2 authorization boundary was relaxed")

    contacts = _load_contacts(document)
    profiles, selected = _load_clearance_profiles(document)
    mass_source, mass_source_revision, body_mass, nut_mass = (
        _load_simulation_mass_properties(document)
    )
    return PublicSpecKeyedV2(
        path=config_path,
        document=document,
        contacts=contacts,
        clearance_profiles=profiles,
        simulation_acceptance_profile=selected,
        mass_property_source_kind=mass_source,
        mass_property_source_asset_revision=mass_source_revision,
        body_mass_properties=body_mass,
        nut_mass_properties=nut_mass,
    )


def safe_new_asset_output(path: Path | str) -> Path:
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".usd", ".usda"}:
        raise ValueError("keyed-v2 output must be a .usd or .usda file")
    if output.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError("keyed-v2 cannot overwrite a legacy or C2 proxy asset")
    if output.name != RECOMMENDED_ASSET_NAME:
        raise ValueError(f"keyed-v2 output basename must be {RECOMMENDED_ASSET_NAME}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing asset: {output}")
    return output


__all__ = [
    "ClearanceProfile",
    "ContactPosition",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_CONTACT_LABELS",
    "EXPECTED_KEY_ANGLES_DEG",
    "MASS_PROPERTY_SOURCE_KIND",
    "MassPropertiesAssumption",
    "PAIR_MODEL_ID",
    "PLUG_MODEL_ID",
    "PublicSpecKeyedV2",
    "RECEPTACLE_MODEL_ID",
    "RECOMMENDED_ASSET_NAME",
    "R4_MASS_PROPERTY_SOURCE_REVISION",
    "ROOT_PRIM",
    "keyed_yaw_peak_to_peak_clearance_deg",
    "keyed_pattern_fits_at_yaw",
    "load_keyed_public_spec_v2",
    "rectangular_key_fits",
    "safe_new_asset_output",
]
