"""Strict, simulator-free contract for the shell-25/J D38999 proxy.

The shipped YAML transcribes public DLA drawing rows and keeps every inferred
simulation value visibly separate.  Importing this module never imports Isaac
Sim, Omni, or Pixar USD.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


D38999_PROXY_SCHEMA_VERSION = "kcg_d38999_shell25j_proxy_v1"
DEFAULT_D38999_PROXY_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_shell25j_proxy_v1.yaml"
)
LEGACY_SYNTHETIC_ASSET_NAME = "connector_pair.usda"
RECOMMENDED_D38999_ASSET_NAME = (
    "d38999_shell25j_61_pair_proxy_v1.usda"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECEPTACLE_APPLICABILITY_NOTE = (
    "V, W, and Z apply to selected class K; primed values and the 91.310 "
    "N-m external bending moment apply only to classes J and M and are not "
    "applicable to selected class K."
)
_PLUG_APPLICABILITY_NOTE = (
    "B and Z apply to selected class K; B-prime, H, Z-prime, and the 91.310 "
    "N-m external bending moment apply only to classes J and M and are not "
    "applicable to selected class K."
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        unexpected = sorted(actual - wanted)
        raise ValueError(
            f"{label} keys are invalid; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _exact_text(value: Any, expected: str, label: str) -> str:
    result = _text(value, label)
    if result != expected:
        raise ValueError(f"{label} must be {expected!r}")
    return result


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


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


@dataclass(frozen=True)
class ProxyIdentity:
    proxy_id: str
    fidelity: str
    loose_part_number: str
    fixed_part_number: str
    shell_size: int
    shell_size_code: str
    insert_arrangement: int
    loose_contact_style: str
    fixed_contact_style: str
    polarization: str
    certification_claim: str


@dataclass(frozen=True)
class PublicSource:
    local_path: str
    sha256: str
    document_id: str
    amendment: int
    issue_date: str
    figure: str
    dimensional_table_page: int
    selected_shell_row: str
    selected_class: str
    applicability_note: str


@dataclass(frozen=True)
class ReceptaclePublicDimensions:
    m_panel_thickness_max: float
    p_nominal: float
    p_plus_minus: float
    pp_nominal: float
    pp_plus_minus: float
    r1: float
    r2: float
    s_nominal: float
    s_plus_minus: float
    v_nominal: float
    v_plus: float
    v_minus: float
    v_prime_nominal: float
    v_prime_plus: float
    v_prime_minus: float
    w_max: float
    w_min: float
    w_prime_max: float
    w_prime_min: float
    z_max: float
    z_prime_max: float
    class_j_m_external_bending_moment_minimum_nm: float


@dataclass(frozen=True)
class PlugPublicDimensions:
    b_diameter_nominal: float
    b_diameter_plus: float
    b_diameter_minus: float
    b_prime_diameter_max: float
    h_max_rib_count: int
    k_max: float
    s_diameter_max: float
    z_max: float
    z_prime_max: float
    class_j_m_external_bending_moment_minimum_nm: float


@dataclass(frozen=True)
class PlugProxyGeometry:
    overall_length: float
    coupling_nut_outer_radius: float
    coupling_nut_inner_radius: float
    coupling_nut_length: float
    rear_body_radius: float
    rear_body_length: float
    mating_shell_outer_radius: float
    mating_shell_inner_radius: float
    mating_shell_length: float
    contact_face_radius: float
    contact_visual_radius: float
    contact_visual_depth: float
    contact_count: int
    grip_segment_count: int
    initial_pair_separation: float


@dataclass(frozen=True)
class ReceptacleProxyGeometry:
    flange_side: float
    flange_thickness: float
    shell_outer_radius: float
    entry_radius: float
    front_shell_length: float
    rear_body_radius: float
    rear_body_length: float
    contact_face_radius: float
    pin_visual_radius: float
    pin_visual_length: float
    mounting_slot_width: float
    mounting_slot_length: float
    mounting_slot_center_offset: float
    contact_count: int


@dataclass(frozen=True)
class PhysicsAssumptions:
    plug_body_mass_kg: float
    coupling_nut_mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float


@dataclass(frozen=True)
class ProxyRules:
    thread_collision_mode: str
    contact_collision_mode: str
    mounting_slot_collision_mode: str
    exact_insert_geometry_available: bool
    exact_mass_and_inertia_available: bool
    certified_geometry: bool
    space_qualified_claim: bool
    intended_use: str


@dataclass(frozen=True)
class D38999Shell25JProxy:
    schema_version: str
    identity: ProxyIdentity
    receptacle_source: PublicSource
    plug_source: PublicSource
    receptacle_public_mm: ReceptaclePublicDimensions
    plug_public_mm: PlugPublicDimensions
    plug_geometry_m: PlugProxyGeometry
    receptacle_geometry_m: ReceptacleProxyGeometry
    physics: PhysicsAssumptions
    rules: ProxyRules
    unknowns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_identity(value: Any) -> ProxyIdentity:
    document = _mapping(value, "identity")
    keys = (
        "proxy_id",
        "fidelity",
        "loose_part_number",
        "fixed_part_number",
        "shell_size",
        "shell_size_code",
        "insert_arrangement",
        "loose_contact_style",
        "fixed_contact_style",
        "polarization",
        "certification_claim",
    )
    _exact_keys(document, keys, "identity")
    result = ProxyIdentity(
        proxy_id=_exact_text(
            document["proxy_id"],
            "d38999_shell25j_61_pair_proxy_v1",
            "identity.proxy_id",
        ),
        fidelity=_exact_text(
            document["fidelity"],
            "public_dimensional_visual_physics_proxy",
            "identity.fidelity",
        ),
        loose_part_number=_exact_text(
            document["loose_part_number"],
            "D38999/26KJ61SN",
            "identity.loose_part_number",
        ),
        fixed_part_number=_exact_text(
            document["fixed_part_number"],
            "D38999/20KJ61PN",
            "identity.fixed_part_number",
        ),
        shell_size=_positive_integer(
            document["shell_size"], "identity.shell_size"
        ),
        shell_size_code=_exact_text(
            document["shell_size_code"], "J", "identity.shell_size_code"
        ),
        insert_arrangement=_positive_integer(
            document["insert_arrangement"],
            "identity.insert_arrangement",
        ),
        loose_contact_style=_exact_text(
            document["loose_contact_style"],
            "socket",
            "identity.loose_contact_style",
        ),
        fixed_contact_style=_exact_text(
            document["fixed_contact_style"],
            "pin",
            "identity.fixed_contact_style",
        ),
        polarization=_exact_text(
            document["polarization"], "N", "identity.polarization"
        ),
        certification_claim=_exact_text(
            document["certification_claim"],
            "none",
            "identity.certification_claim",
        ),
    )
    if result.shell_size != 25 or result.insert_arrangement != 61:
        raise ValueError("identity must select shell 25/J arrangement 61")
    return result


def _load_source(value: Any, label: str) -> PublicSource:
    document = _mapping(value, label)
    keys = (
        "local_path",
        "sha256",
        "document_id",
        "amendment",
        "issue_date",
        "figure",
        "dimensional_table_page",
        "selected_shell_row",
        "selected_class",
        "applicability_note",
    )
    _exact_keys(document, keys, label)
    sha256 = _text(document["sha256"], f"{label}.sha256")
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    local_path = _text(document["local_path"], f"{label}.local_path")
    if Path(local_path).is_absolute() or ".." in Path(local_path).parts:
        raise ValueError(f"{label}.local_path must be package-relative")
    result = PublicSource(
        local_path=local_path,
        sha256=sha256,
        document_id=_text(document["document_id"], f"{label}.document_id"),
        amendment=_positive_integer(
            document["amendment"], f"{label}.amendment"
        ),
        issue_date=_text(document["issue_date"], f"{label}.issue_date"),
        figure=_text(document["figure"], f"{label}.figure"),
        dimensional_table_page=_positive_integer(
            document["dimensional_table_page"],
            f"{label}.dimensional_table_page",
        ),
        selected_shell_row=_exact_text(
            document["selected_shell_row"],
            "25/J",
            f"{label}.selected_shell_row",
        ),
        selected_class=_exact_text(
            document["selected_class"], "K", f"{label}.selected_class"
        ),
        applicability_note=_text(
            document["applicability_note"],
            f"{label}.applicability_note",
        ),
    )
    if result.dimensional_table_page != 2:
        raise ValueError(f"{label} dimensions must cite drawing page 2")
    return result


def _numeric_dataclass(cls, value: Any, label: str):
    document = _mapping(value, label)
    keys = tuple(cls.__dataclass_fields__)
    _exact_keys(document, keys, label)
    parsed = {}
    for name, field in cls.__dataclass_fields__.items():
        item_label = f"{label}.{name}"
        if field.type == "int" or name.endswith("_count"):
            parsed[name] = _positive_integer(document[name], item_label)
        else:
            parsed[name] = _finite(document[name], item_label)
    return cls(**parsed)


def _load_physics(value: Any) -> PhysicsAssumptions:
    result = _numeric_dataclass(
        PhysicsAssumptions, value, "physics_assumptions"
    )
    if result.plug_body_mass_kg <= 0.0 or result.coupling_nut_mass_kg <= 0.0:
        raise ValueError("proxy masses must be positive assumptions")
    if result.static_friction < result.dynamic_friction:
        raise ValueError("static friction must not be below dynamic friction")
    if result.dynamic_friction < 0.0:
        raise ValueError("friction assumptions must be nonnegative")
    if not 0.0 <= result.restitution <= 1.0:
        raise ValueError("restitution must be in [0, 1]")
    return result


def _load_rules(value: Any) -> ProxyRules:
    document = _mapping(value, "proxy_rules")
    keys = tuple(ProxyRules.__dataclass_fields__)
    _exact_keys(document, keys, "proxy_rules")
    result = ProxyRules(
        thread_collision_mode=_exact_text(
            document["thread_collision_mode"],
            "none",
            "proxy_rules.thread_collision_mode",
        ),
        contact_collision_mode=_exact_text(
            document["contact_collision_mode"],
            "simplified_face_stop",
            "proxy_rules.contact_collision_mode",
        ),
        mounting_slot_collision_mode=_exact_text(
            document["mounting_slot_collision_mode"],
            "visual_only_not_subtracted",
            "proxy_rules.mounting_slot_collision_mode",
        ),
        exact_insert_geometry_available=_boolean(
            document["exact_insert_geometry_available"],
            "proxy_rules.exact_insert_geometry_available",
        ),
        exact_mass_and_inertia_available=_boolean(
            document["exact_mass_and_inertia_available"],
            "proxy_rules.exact_mass_and_inertia_available",
        ),
        certified_geometry=_boolean(
            document["certified_geometry"],
            "proxy_rules.certified_geometry",
        ),
        space_qualified_claim=_boolean(
            document["space_qualified_claim"],
            "proxy_rules.space_qualified_claim",
        ),
        intended_use=_exact_text(
            document["intended_use"],
            "visualization_and_early_simulation_only",
            "proxy_rules.intended_use",
        ),
    )
    if any(
        (
            result.exact_insert_geometry_available,
            result.exact_mass_and_inertia_available,
            result.certified_geometry,
            result.space_qualified_claim,
        )
    ):
        raise ValueError("the public-dimensional proxy cannot claim fidelity")
    return result


def _validate_public_rows(config: D38999Shell25JProxy) -> None:
    receptacle_expected = {
        "m_panel_thickness_max": 5.00,
        "p_nominal": 3.91,
        "p_plus_minus": 0.20,
        "pp_nominal": 6.15,
        "pp_plus_minus": 0.20,
        "r1": 38.10,
        "r2": 34.93,
        "s_nominal": 46.0,
        "s_plus_minus": 0.3,
        "v_nominal": 20.07,
        "v_plus": 0.0,
        "v_minus": 1.25,
        "v_prime_nominal": 18.7,
        "v_prime_plus": 1.4,
        "v_prime_minus": 0.0,
        "w_max": 3.2,
        "w_min": 2.1,
        "w_prime_max": 4.35,
        "w_prime_min": 2.1,
        "z_max": 31.5,
        "z_prime_max": 32.0,
        "class_j_m_external_bending_moment_minimum_nm": 91.310,
    }
    plug_expected = {
        "b_diameter_nominal": 44.3,
        "b_diameter_plus": 0.2,
        "b_diameter_minus": 0.0,
        "b_prime_diameter_max": 46.8,
        "h_max_rib_count": 28,
        "k_max": 44.9,
        "s_diameter_max": 48.0,
        "z_max": 31.0,
        "z_prime_max": 31.5,
        "class_j_m_external_bending_moment_minimum_nm": 91.310,
    }
    if asdict(config.receptacle_public_mm) != receptacle_expected:
        raise ValueError("receptacle public row differs from /20H page 2")
    if asdict(config.plug_public_mm) != plug_expected:
        raise ValueError("plug public row differs from /26G page 2")


def _validate_proxy_geometry(config: D38999Shell25JProxy) -> None:
    plug = config.plug_geometry_m
    receptacle = config.receptacle_geometry_m
    source_plug = config.plug_public_mm
    source_receptacle = config.receptacle_public_mm
    positive_values = [
        value
        for values in (asdict(plug), asdict(receptacle))
        for value in values.values()
        if not isinstance(value, int)
    ]
    if any(value <= 0.0 for value in positive_values):
        raise ValueError("all proxy geometry dimensions must be positive")
    if plug.contact_count != 61 or receptacle.contact_count != 61:
        raise ValueError("the visual insert must retain exactly 61 contacts")
    if plug.coupling_nut_outer_radius * 2.0 > (
        source_plug.s_diameter_max * 1.0e-3 + 1.0e-12
    ):
        raise ValueError("coupling nut exceeds public S diameter envelope")
    if plug.rear_body_radius * 2.0 > (
        (source_plug.b_diameter_nominal + source_plug.b_diameter_plus)
        * 1.0e-3
        + 1.0e-12
    ):
        raise ValueError("plug rear body exceeds public B diameter envelope")
    if plug.overall_length > source_plug.z_max * 1.0e-3 + 1.0e-12:
        raise ValueError("plug length exceeds public Z envelope")
    if receptacle.flange_side > (
        (source_receptacle.s_nominal + source_receptacle.s_plus_minus)
        * 1.0e-3
        + 1.0e-12
    ):
        raise ValueError("receptacle flange exceeds public S envelope")
    if not (
        plug.mating_shell_inner_radius
        < plug.mating_shell_outer_radius
        < plug.coupling_nut_inner_radius
        < plug.coupling_nut_outer_radius
    ):
        raise ValueError("plug radial proxy dimensions are inconsistent")
    if not (
        receptacle.contact_face_radius
        < receptacle.entry_radius
        < receptacle.shell_outer_radius
    ):
        raise ValueError("receptacle radial proxy dimensions are inconsistent")
    expected_offset = 0.25 * (
        source_receptacle.r1 + source_receptacle.r2
    ) * 1.0e-3
    if abs(receptacle.mounting_slot_center_offset - expected_offset) > 1e-9:
        raise ValueError("mounting slot center must remain an explicit proxy")


def load_d38999_shell25j_proxy(
    config_path: Path | str = DEFAULT_D38999_PROXY_CONFIG_PATH,
) -> D38999Shell25JProxy:
    """Load the versioned proxy and fail closed on every schema drift."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "root")
    root_keys = (
        "schema_version",
        "identity",
        "source_documents",
        "public_dimensions_mm",
        "proxy_geometry_m",
        "physics_assumptions",
        "proxy_rules",
        "unknowns",
    )
    _exact_keys(root, root_keys, "root")
    schema = _exact_text(
        root["schema_version"],
        D38999_PROXY_SCHEMA_VERSION,
        "schema_version",
    )
    sources = _mapping(root["source_documents"], "source_documents")
    _exact_keys(sources, ("receptacle", "plug"), "source_documents")
    dimensions = _mapping(
        root["public_dimensions_mm"], "public_dimensions_mm"
    )
    _exact_keys(dimensions, ("receptacle", "plug"), "public_dimensions_mm")
    geometry = _mapping(root["proxy_geometry_m"], "proxy_geometry_m")
    _exact_keys(geometry, ("plug", "receptacle"), "proxy_geometry_m")
    unknowns_value = root["unknowns"]
    if not isinstance(unknowns_value, list) or not unknowns_value:
        raise ValueError("unknowns must be a non-empty list")
    unknowns = tuple(
        _text(value, f"unknowns[{index}]")
        for index, value in enumerate(unknowns_value)
    )
    if len(set(unknowns)) != len(unknowns):
        raise ValueError("unknowns must not contain duplicates")
    config = D38999Shell25JProxy(
        schema_version=schema,
        identity=_load_identity(root["identity"]),
        receptacle_source=_load_source(
            sources["receptacle"], "source_documents.receptacle"
        ),
        plug_source=_load_source(
            sources["plug"], "source_documents.plug"
        ),
        receptacle_public_mm=_numeric_dataclass(
            ReceptaclePublicDimensions,
            dimensions["receptacle"],
            "public_dimensions_mm.receptacle",
        ),
        plug_public_mm=_numeric_dataclass(
            PlugPublicDimensions,
            dimensions["plug"],
            "public_dimensions_mm.plug",
        ),
        plug_geometry_m=_numeric_dataclass(
            PlugProxyGeometry,
            geometry["plug"],
            "proxy_geometry_m.plug",
        ),
        receptacle_geometry_m=_numeric_dataclass(
            ReceptacleProxyGeometry,
            geometry["receptacle"],
            "proxy_geometry_m.receptacle",
        ),
        physics=_load_physics(root["physics_assumptions"]),
        rules=_load_rules(root["proxy_rules"]),
        unknowns=unknowns,
    )
    if config.receptacle_source.document_id != "MIL-DTL-38999/20H":
        raise ValueError("receptacle source must be MIL-DTL-38999/20H")
    if config.receptacle_source.amendment != 1:
        raise ValueError("receptacle source must use amendment 1")
    if config.plug_source.document_id != "MIL-DTL-38999/26G":
        raise ValueError("plug source must be MIL-DTL-38999/26G")
    if config.plug_source.amendment != 4:
        raise ValueError("plug source must use amendment 4")
    if (
        config.receptacle_source.applicability_note
        != _RECEPTACLE_APPLICABILITY_NOTE
    ):
        raise ValueError(
            "receptacle applicability must exclude the class J/M bending "
            "row from selected class K"
        )
    if config.plug_source.applicability_note != _PLUG_APPLICABILITY_NOTE:
        raise ValueError(
            "plug applicability must exclude the class J/M bending row "
            "from selected class K"
        )
    _validate_public_rows(config)
    _validate_proxy_geometry(config)
    return config


def verify_public_source_files(
    config: D38999Shell25JProxy,
    package_root: Path | str,
) -> tuple[Path, Path]:
    """Verify that both local public PDFs match the recorded SHA-256."""
    root = Path(package_root).expanduser().resolve()
    verified = []
    for label, source in (
        ("receptacle", config.receptacle_source),
        ("plug", config.plug_source),
    ):
        path = (root / source.local_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} source escapes package root") from error
        if not path.is_file():
            raise ValueError(f"{label} source PDF is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.sha256:
            raise ValueError(f"{label} source PDF SHA-256 mismatch")
        verified.append(path)
    return tuple(verified)


def require_safe_d38999_output(output_path: Path | str) -> Path:
    """Reject the legacy synthetic asset name before Isaac starts."""
    path = Path(output_path).expanduser().resolve()
    if path.name == LEGACY_SYNTHETIC_ASSET_NAME:
        raise ValueError(
            "refusing to overwrite the legacy connector_pair.usda asset"
        )
    if path.suffix.lower() not in {".usd", ".usda"}:
        raise ValueError("D38999 proxy output must use .usd or .usda")
    return path
