"""Object-independent URDF kinematics for the CARTS-Grasp planner.

The planner deliberately obtains joint topology, limits, mimic couplings and
PAD geometry from mechanical contracts.  Nothing in this module contains a
connector model name, a stored grasp pose, or a dependency on an earlier
candidate generator.

URDF defines the rigid-body tree but, in general, does not identify which
subset of a terminal collision mesh is allowed to transmit grasp forces.
``ThreeFingerHandModel.from_urdf`` therefore accepts an optional, explicit PAD
geometry contract.  When it is omitted, terminal collision geometry is
reported as an unlabelled geometric reference and no contact normal is
invented.  Force-bearing planning should always supply the explicit contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


_DEFAULT_FINGER_PATTERN = re.compile(r"^(f\d+)j\d+$")
_MOVABLE_JOINT_TYPES = frozenset(("revolute", "continuous", "prismatic"))


class HandModelError(ValueError):
    """Raised when the hand's mechanical contract is incomplete or invalid."""


def _vector(
    value: Sequence[float] | str | None,
    *,
    length: int,
    default: Sequence[float],
    label: str,
) -> tuple[float, ...]:
    if value is None:
        values = tuple(float(item) for item in default)
    elif isinstance(value, str):
        values = tuple(float(item) for item in value.split())
    else:
        values = tuple(float(item) for item in value)
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise HandModelError(f"{label} must contain {length} finite values")
    return values


def _normalised_vector(
    value: Sequence[float] | str,
    *,
    label: str,
) -> tuple[float, float, float]:
    vector = np.asarray(
        _vector(value, length=3, default=(0.0, 0.0, 0.0), label=label),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise HandModelError(f"{label} must be non-zero")
    vector /= norm
    return tuple(float(item) for item in vector)


def _scaled_float_tolerance(*values: float) -> float:
    """Roundoff bound derived only from finite values participating in a check."""

    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    scale = max((1.0, *finite))
    return 64.0 * np.finfo(np.float64).eps * scale


def rpy_rotation(rpy_rad: Sequence[float]) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""

    roll, pitch, yaw = _vector(
        rpy_rad,
        length=3,
        default=(0.0, 0.0, 0.0),
        label="rpy_rad",
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def rigid_transform(
    xyz_m: Sequence[float] = (0.0, 0.0, 0.0),
    rpy_rad: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Construct a homogeneous transform from URDF ``xyz`` and ``rpy``."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_rotation(rpy_rad)
    transform[:3, 3] = np.asarray(
        _vector(xyz_m, length=3, default=(0.0, 0.0, 0.0), label="xyz_m"),
        dtype=np.float64,
    )
    return transform


def _axis_rotation(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64)
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    identity = np.eye(3, dtype=np.float64)
    return (
        identity
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )


@dataclass(frozen=True)
class JointLimit:
    """URDF joint bounds in SI units."""

    lower: float
    upper: float
    effort: float | None = None
    velocity: float | None = None

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise HandModelError("joint lower limit must not exceed upper limit")
        for label, value in (("effort", self.effort), ("velocity", self.velocity)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise HandModelError(f"joint {label} limit must be positive")

    def contains(self, position: float, *, tolerance: float = 0.0) -> bool:
        return self.lower - tolerance <= position <= self.upper + tolerance


@dataclass(frozen=True)
class MimicSpec:
    source_joint: str
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_joint:
            raise HandModelError("mimic source joint must be named")
        if not math.isfinite(self.multiplier) or not math.isfinite(self.offset):
            raise HandModelError("mimic multiplier and offset must be finite")


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    axis: tuple[float, float, float]
    limit: JointLimit | None
    mimic: MimicSpec | None = None

    @property
    def movable(self) -> bool:
        return self.joint_type in _MOVABLE_JOINT_TYPES

    def origin_transform(self) -> np.ndarray:
        return rigid_transform(self.origin_xyz_m, self.origin_rpy_rad)

    def motion_transform(self, position: float) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        if self.joint_type in ("revolute", "continuous"):
            transform[:3, :3] = _axis_rotation(self.axis, float(position))
        elif self.joint_type == "prismatic":
            transform[:3, 3] = np.asarray(self.axis, dtype=np.float64) * float(position)
        elif self.joint_type != "fixed":
            raise HandModelError(f"unsupported URDF joint type: {self.joint_type}")
        return transform


@dataclass(frozen=True)
class GeometrySpec:
    """Finite PAD footprint or a traceable URDF collision reference."""

    kind: str
    dimensions_m: tuple[float, ...] = ()
    mesh_uri: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not self.kind:
            raise HandModelError("geometry kind must be named")
        if any(not math.isfinite(item) or item <= 0.0 for item in self.dimensions_m):
            raise HandModelError("geometry dimensions must be finite and positive")
        if any(not math.isfinite(item) or item <= 0.0 for item in self.mesh_scale):
            raise HandModelError("mesh scale must be finite and positive")
        if self.kind == "mesh" and not self.mesh_uri:
            raise HandModelError("mesh geometry requires mesh_uri")


@dataclass(frozen=True)
class PadGeometry:
    """PAD frame and finite footprint expressed in its terminal link."""

    name: str
    finger_name: str
    link_name: str
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    geometry: GeometrySpec
    contact_normal_pad: tuple[float, float, float] | None = None
    normal_force_capacity_n: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.finger_name or not self.link_name:
            raise HandModelError("PAD name, finger and link must be named")
        if self.normal_force_capacity_n is not None and (
            not math.isfinite(self.normal_force_capacity_n)
            or self.normal_force_capacity_n <= 0.0
        ):
            raise HandModelError("PAD normal-force capacity must be positive")

    def link_from_pad_transform(self) -> np.ndarray:
        return rigid_transform(self.origin_xyz_m, self.origin_rpy_rad)


@dataclass(frozen=True)
class FingerChain:
    name: str
    joint_names: tuple[str, ...]
    terminal_link: str
    pad_name: str


@dataclass(frozen=True)
class KinematicNormalDomain:
    """Surface-normal half-space induced by a PAD's closing velocity.

    For an object outward normal ``n``, contact is approached when
    ``-n.dot(v_pad)`` is positive.  The only tolerance stored here is a bound
    on floating-point dot-product error scaled by the realised velocity; it is
    not an angular or object-specific acceptance parameter.
    """

    pad_name: str
    pad_origin_base_m: tuple[float, float, float]
    closing_velocity_base_m_s: tuple[float, float, float]
    numerical_tolerance_m_s: float

    def __post_init__(self) -> None:
        if not self.pad_name:
            raise HandModelError("kinematic normal domain must name a PAD")
        _vector(
            self.pad_origin_base_m,
            length=3,
            default=(0.0, 0.0, 0.0),
            label="PAD origin",
        )
        _vector(
            self.closing_velocity_base_m_s,
            length=3,
            default=(0.0, 0.0, 0.0),
            label="PAD closing velocity",
        )
        if (
            not math.isfinite(self.numerical_tolerance_m_s)
            or self.numerical_tolerance_m_s < 0.0
        ):
            raise HandModelError("normal-domain numerical tolerance cannot be negative")

    def approach_speed_m_s(self, outward_normal_base: Sequence[float]) -> float:
        normal = np.asarray(
            _normalised_vector(
                outward_normal_base, label="object outward contact normal"
            ),
            dtype=np.float64,
        )
        velocity = np.asarray(self.closing_velocity_base_m_s, dtype=np.float64)
        return -float(normal @ velocity)

    def contains(self, outward_normal_base: Sequence[float]) -> bool:
        return self.approach_speed_m_s(outward_normal_base) > self.numerical_tolerance_m_s


def _parse_origin(element: ET.Element | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = _vector(
        element.attrib.get("xyz"),
        length=3,
        default=(0.0, 0.0, 0.0),
        label="origin xyz",
    )
    rpy = _vector(
        element.attrib.get("rpy"),
        length=3,
        default=(0.0, 0.0, 0.0),
        label="origin rpy",
    )
    return xyz, rpy


def _parse_geometry(element: ET.Element) -> GeometrySpec:
    box = element.find("box")
    if box is not None:
        size = _vector(
            box.attrib.get("size"),
            length=3,
            default=(),
            label="box size",
        )
        return GeometrySpec("box", size)
    cylinder = element.find("cylinder")
    if cylinder is not None:
        try:
            radius = float(cylinder.attrib["radius"])
            length = float(cylinder.attrib["length"])
        except (KeyError, ValueError) as exc:
            raise HandModelError("cylinder requires numeric radius and length") from exc
        return GeometrySpec("cylinder", (radius, length))
    sphere = element.find("sphere")
    if sphere is not None:
        try:
            radius = float(sphere.attrib["radius"])
        except (KeyError, ValueError) as exc:
            raise HandModelError("sphere requires a numeric radius") from exc
        return GeometrySpec("sphere", (radius,))
    mesh = element.find("mesh")
    if mesh is not None:
        uri = mesh.attrib.get("filename")
        scale = _vector(
            mesh.attrib.get("scale"),
            length=3,
            default=(1.0, 1.0, 1.0),
            label="mesh scale",
        )
        return GeometrySpec("mesh", mesh_uri=uri, mesh_scale=scale)
    raise HandModelError("collision geometry has no supported primitive or mesh")


def _load_xml(source: str | bytes | Path) -> ET.Element:
    if isinstance(source, bytes):
        root = ET.fromstring(source)
    elif isinstance(source, Path):
        root = ET.parse(source).getroot()
    elif "<robot" in source:
        root = ET.fromstring(source)
    else:
        root = ET.parse(Path(source)).getroot()
    if root.tag != "robot":
        raise HandModelError("URDF root element must be <robot>")
    return root


def _load_contract(source: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - package dependency in workspace
            raise HandModelError("YAML PAD contracts require PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise HandModelError("PAD contract root must be a mapping")
    return value


def _geometry_from_contract(value: Mapping[str, Any]) -> GeometrySpec:
    kind = str(value.get("kind", value.get("type", ""))).lower()
    if not kind:
        raise HandModelError("PAD footprint requires kind")
    dimensions_value = value.get("dimensions_m")
    if dimensions_value is not None:
        dimensions = tuple(float(item) for item in dimensions_value)
    elif kind == "box":
        dimensions = _vector(
            value.get("size_m", value.get("size")),
            length=3,
            default=(),
            label="PAD box size",
        )
    elif kind in ("capsule", "cylinder"):
        dimensions = (
            float(value["radius_m"]),
            float(value.get("length_m", value.get("cylinder_length_m"))),
        )
    elif kind in ("disk", "sphere"):
        dimensions = (float(value["radius_m"]),)
    elif kind == "rectangle":
        dimensions = _vector(
            value.get("size_m"),
            length=2,
            default=(),
            label="PAD rectangle size",
        )
    elif kind == "mesh":
        dimensions = ()
    else:
        dimensions = tuple(float(item) for item in value.get("parameters_m", ()))
    mesh_scale = _vector(
        value.get("mesh_scale"),
        length=3,
        default=(1.0, 1.0, 1.0),
        label="PAD mesh scale",
    )
    return GeometrySpec(
        kind=kind,
        dimensions_m=dimensions,
        mesh_uri=value.get("mesh_uri", value.get("filename")),
        mesh_scale=mesh_scale,
    )


class ThreeFingerHandModel:
    """Parsed three-finger hand tree with deterministic FK and PAD frames."""

    def __init__(
        self,
        *,
        base_link: str,
        joints: Mapping[str, JointSpec],
        joint_order: Sequence[str],
        finger_joint_names: Mapping[str, Sequence[str]],
        pads: Mapping[str, PadGeometry],
    ) -> None:
        self.base_link = base_link
        self.joints = MappingProxyType(dict(joints))
        self.joint_order = tuple(joint_order)
        self.pads = MappingProxyType(dict(pads))

        self._children_by_parent: dict[str, list[str]] = {}
        self._joint_by_child: dict[str, str] = {}
        for name in self.joint_order:
            joint = self.joints[name]
            self._children_by_parent.setdefault(joint.parent_link, []).append(name)
            self._joint_by_child[joint.child_link] = name

        independent = tuple(
            name
            for name in self.joint_order
            if self.joints[name].movable and self.joints[name].mimic is None
        )
        if not independent:
            raise HandModelError("hand has no independently actuated joints")
        self.independent_joint_names = independent

        pad_by_finger = {pad.finger_name: pad for pad in self.pads.values()}
        chains: dict[str, FingerChain] = {}
        for finger_name, names_value in finger_joint_names.items():
            names = tuple(names_value)
            if not names:
                raise HandModelError(f"finger {finger_name} has no joints")
            pad = pad_by_finger.get(finger_name)
            if pad is None:
                raise HandModelError(f"finger {finger_name} has no PAD geometry")
            chains[finger_name] = FingerChain(
                name=finger_name,
                joint_names=names,
                terminal_link=self.joints[names[-1]].child_link,
                pad_name=pad.name,
            )
        if len(chains) != 3:
            raise HandModelError(f"expected exactly three fingers, found {len(chains)}")
        self.fingers = MappingProxyType(chains)

        self._independent_affine_limits = MappingProxyType(
            self._compute_independent_affine_limits()
        )

    @classmethod
    def from_urdf(
        cls,
        source: str | bytes | Path,
        *,
        pad_geometry_contract: Mapping[str, Any] | str | Path | None = None,
        base_link: str = "handbase_link",
        finger_joint_pattern: re.Pattern[str] = _DEFAULT_FINGER_PATTERN,
    ) -> "ThreeFingerHandModel":
        """Parse a URDF/xacro file and an optional explicit PAD contract."""

        root = _load_xml(source)
        links = {element.attrib["name"]: element for element in root.findall("link")}
        if base_link not in links:
            raise HandModelError(f"base link not present in URDF: {base_link}")

        parsed_joints: dict[str, JointSpec] = {}
        children_by_parent: dict[str, list[str]] = {}
        document_order: list[str] = []
        for element in root.findall("joint"):
            name = element.attrib.get("name", "")
            joint_type = element.attrib.get("type", "")
            parent_element = element.find("parent")
            child_element = element.find("child")
            if not name or parent_element is None or child_element is None:
                raise HandModelError("every joint requires name, parent and child")
            parent = parent_element.attrib.get("link", "")
            child = child_element.attrib.get("link", "")
            if not parent or not child:
                raise HandModelError(f"joint {name} has an unnamed parent or child")
            xyz, rpy = _parse_origin(element.find("origin"))
            axis_element = element.find("axis")
            axis_value = (
                "1 0 0" if axis_element is None else axis_element.attrib.get("xyz", "1 0 0")
            )
            axis = _normalised_vector(axis_value, label=f"joint {name} axis")

            limit_element = element.find("limit")
            limit: JointLimit | None
            if joint_type == "continuous":
                effort = None
                velocity = None
                if limit_element is not None:
                    if "effort" in limit_element.attrib:
                        effort = float(limit_element.attrib["effort"])
                    if "velocity" in limit_element.attrib:
                        velocity = float(limit_element.attrib["velocity"])
                limit = JointLimit(-math.inf, math.inf, effort, velocity)
            elif joint_type in ("revolute", "prismatic"):
                if limit_element is None:
                    raise HandModelError(f"joint {name} requires limits")
                try:
                    lower = float(limit_element.attrib["lower"])
                    upper = float(limit_element.attrib["upper"])
                except (KeyError, ValueError) as exc:
                    raise HandModelError(f"joint {name} has invalid limits") from exc
                effort = (
                    float(limit_element.attrib["effort"])
                    if "effort" in limit_element.attrib
                    else None
                )
                velocity = (
                    float(limit_element.attrib["velocity"])
                    if "velocity" in limit_element.attrib
                    else None
                )
                limit = JointLimit(lower, upper, effort, velocity)
            elif joint_type == "fixed":
                limit = None
            else:
                raise HandModelError(f"unsupported URDF joint type: {joint_type}")

            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                mimic = MimicSpec(
                    source_joint=mimic_element.attrib.get("joint", ""),
                    multiplier=float(mimic_element.attrib.get("multiplier", "1")),
                    offset=float(mimic_element.attrib.get("offset", "0")),
                )
            if name in parsed_joints:
                raise HandModelError(f"duplicate joint name: {name}")
            parsed_joints[name] = JointSpec(
                name=name,
                joint_type=joint_type,
                parent_link=parent,
                child_link=child,
                origin_xyz_m=xyz,
                origin_rpy_rad=rpy,
                axis=axis,
                limit=limit,
                mimic=mimic,
            )
            document_order.append(name)
            children_by_parent.setdefault(parent, []).append(name)

        reachable_order: list[str] = []
        seen_links = {base_link}
        pending = set(document_order)
        while pending:
            progress = False
            for joint_name in document_order:
                if joint_name not in pending:
                    continue
                joint = parsed_joints[joint_name]
                if joint.parent_link not in seen_links:
                    continue
                if joint.child_link in seen_links:
                    raise HandModelError("URDF hand subtree is not a tree")
                seen_links.add(joint.child_link)
                reachable_order.append(joint_name)
                pending.remove(joint_name)
                progress = True
            if not progress:
                break

        reachable = {name: parsed_joints[name] for name in reachable_order}
        for joint in reachable.values():
            if joint.mimic is not None and joint.mimic.source_joint not in reachable:
                raise HandModelError(
                    f"mimic source {joint.mimic.source_joint} for {joint.name} is outside hand subtree"
                )

        finger_joint_names: dict[str, list[str]] = {}
        for name in reachable_order:
            match = finger_joint_pattern.match(name)
            if match is not None:
                finger_joint_names.setdefault(match.group(1), []).append(name)
        if len(finger_joint_names) != 3:
            raise HandModelError(
                f"finger joint pattern must identify three fingers, found {len(finger_joint_names)}"
            )

        terminal_by_finger = {
            finger: reachable[names[-1]].child_link
            for finger, names in finger_joint_names.items()
        }
        collision_by_link: dict[
            str, list[tuple[tuple[float, ...], tuple[float, ...], GeometrySpec]]
        ] = {}
        for link_name in seen_links:
            link_element = links.get(link_name)
            if link_element is None:
                continue
            rows = []
            for collision in link_element.findall("collision"):
                geometry_element = collision.find("geometry")
                if geometry_element is None:
                    continue
                xyz, rpy = _parse_origin(collision.find("origin"))
                rows.append((xyz, rpy, _parse_geometry(geometry_element)))
            collision_by_link[link_name] = rows

        pads = cls._parse_pads(
            pad_geometry_contract=pad_geometry_contract,
            finger_joint_names=finger_joint_names,
            terminal_by_finger=terminal_by_finger,
            reachable_links=seen_links,
            collision_by_link=collision_by_link,
        )
        return cls(
            base_link=base_link,
            joints=reachable,
            joint_order=reachable_order,
            finger_joint_names=finger_joint_names,
            pads=pads,
        )

    @staticmethod
    def _parse_pads(
        *,
        pad_geometry_contract: Mapping[str, Any] | str | Path | None,
        finger_joint_names: Mapping[str, Sequence[str]],
        terminal_by_finger: Mapping[str, str],
        reachable_links: set[str],
        collision_by_link: Mapping[
            str, Sequence[tuple[tuple[float, ...], tuple[float, ...], GeometrySpec]]
        ],
    ) -> dict[str, PadGeometry]:
        if pad_geometry_contract is None:
            inferred: dict[str, PadGeometry] = {}
            for finger_name in sorted(finger_joint_names):
                link_name = terminal_by_finger[finger_name]
                collisions = collision_by_link.get(link_name, ())
                if collisions:
                    xyz, rpy, geometry = collisions[0]
                else:
                    xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                    geometry = GeometrySpec("link_reference")
                pad_name = f"{finger_name}_pad"
                inferred[pad_name] = PadGeometry(
                    name=pad_name,
                    finger_name=finger_name,
                    link_name=link_name,
                    origin_xyz_m=xyz,
                    origin_rpy_rad=rpy,
                    geometry=geometry,
                    contact_normal_pad=None,
                )
            return inferred

        contract = _load_contract(pad_geometry_contract)
        pads_value = contract.get("pads")
        if isinstance(pads_value, Mapping):
            rows = []
            for name, row_value in pads_value.items():
                if not isinstance(row_value, Mapping):
                    raise HandModelError("each PAD contract entry must be a mapping")
                row = dict(row_value)
                row.setdefault("name", name)
                rows.append(row)
        elif isinstance(pads_value, Sequence) and not isinstance(pads_value, (str, bytes)):
            rows = list(pads_value)
        else:
            raise HandModelError("PAD contract requires a pads mapping or list")

        pads: dict[str, PadGeometry] = {}
        claimed_fingers: set[str] = set()
        for index, row_value in enumerate(rows):
            if not isinstance(row_value, Mapping):
                raise HandModelError("each PAD contract entry must be a mapping")
            row = row_value
            name = str(row.get("name", f"pad_{index + 1}"))
            finger_name = str(row.get("finger_name", row.get("finger", "")))
            if finger_name not in finger_joint_names:
                raise HandModelError(f"PAD {name} references unknown finger {finger_name}")
            link_name = str(row.get("link_name", row.get("link", "")))
            if not link_name:
                link_name = terminal_by_finger[finger_name]
            if link_name not in reachable_links:
                raise HandModelError(f"PAD {name} references link outside hand subtree")
            if name in pads or finger_name in claimed_fingers:
                raise HandModelError("PAD names and finger assignments must be unique")

            footprint_value = row.get("footprint")
            use_urdf_collision = footprint_value is None or str(
                row.get("geometry_source", "")
            ).upper() == "URDF_COLLISION"
            collision_index = int(row.get("collision_index", 0))
            collision_rows = collision_by_link.get(link_name, ())
            if use_urdf_collision:
                if not 0 <= collision_index < len(collision_rows):
                    raise HandModelError(
                        f"PAD {name} has no URDF collision at index {collision_index}"
                    )
                default_xyz, default_rpy, geometry = collision_rows[collision_index]
            else:
                if not isinstance(footprint_value, Mapping):
                    raise HandModelError(f"PAD {name} footprint must be a mapping")
                default_xyz, default_rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                geometry = _geometry_from_contract(footprint_value)

            origin_value = row.get("origin", {})
            if not isinstance(origin_value, Mapping):
                raise HandModelError(f"PAD {name} origin must be a mapping")
            xyz = _vector(
                origin_value.get("xyz_m", origin_value.get("xyz", row.get("origin_xyz_m"))),
                length=3,
                default=default_xyz,
                label=f"PAD {name} origin xyz",
            )
            rpy = _vector(
                origin_value.get(
                    "rpy_rad", origin_value.get("rpy", row.get("origin_rpy_rad"))
                ),
                length=3,
                default=default_rpy,
                label=f"PAD {name} origin rpy",
            )
            normal_value = row.get("contact_normal_pad", row.get("contact_normal_local"))
            normal = (
                None
                if normal_value is None
                else _normalised_vector(normal_value, label=f"PAD {name} contact normal")
            )
            force_capacity = row.get("normal_force_capacity_n")
            pads[name] = PadGeometry(
                name=name,
                finger_name=finger_name,
                link_name=link_name,
                origin_xyz_m=xyz,
                origin_rpy_rad=rpy,
                geometry=geometry,
                contact_normal_pad=normal,
                normal_force_capacity_n=(
                    None if force_capacity is None else float(force_capacity)
                ),
            )
            claimed_fingers.add(finger_name)

        if claimed_fingers != set(finger_joint_names):
            missing = sorted(set(finger_joint_names) - claimed_fingers)
            raise HandModelError(f"PAD contract does not cover all fingers: {missing}")
        return pads

    def _joint_affine_map(
        self,
        joint_name: str,
        cache: dict[str, tuple[str, float, float]],
        active: set[str],
    ) -> tuple[str, float, float]:
        if joint_name in cache:
            return cache[joint_name]
        if joint_name in active:
            raise HandModelError("cyclic mimic relation in URDF")
        active.add(joint_name)
        joint = self.joints[joint_name]
        if joint.mimic is None:
            result = (joint_name, 1.0, 0.0)
        else:
            source, multiplier, offset = self._joint_affine_map(
                joint.mimic.source_joint, cache, active
            )
            result = (
                source,
                joint.mimic.multiplier * multiplier,
                joint.mimic.multiplier * offset + joint.mimic.offset,
            )
        active.remove(joint_name)
        cache[joint_name] = result
        return result

    def _compute_independent_affine_limits(self) -> dict[str, JointLimit]:
        bounds = {
            name: [-math.inf, math.inf]
            for name in self.independent_joint_names
        }
        cache: dict[str, tuple[str, float, float]] = {}
        for name in self.joint_order:
            joint = self.joints[name]
            if not joint.movable or joint.limit is None:
                continue
            source, multiplier, offset = self._joint_affine_map(name, cache, set())
            if source not in bounds:
                raise HandModelError(f"joint {name} resolves outside independent hand DOFs")
            lower, upper = joint.limit.lower, joint.limit.upper
            if multiplier == 0.0:
                if not joint.limit.contains(offset):
                    raise HandModelError(f"constant mimic position violates limit for {name}")
                continue
            transformed = ((lower - offset) / multiplier, (upper - offset) / multiplier)
            allowed_lower, allowed_upper = min(transformed), max(transformed)
            bounds[source][0] = max(bounds[source][0], allowed_lower)
            bounds[source][1] = min(bounds[source][1], allowed_upper)

        result: dict[str, JointLimit] = {}
        for name, (lower, upper) in bounds.items():
            if lower > upper:
                raise HandModelError(f"mimic limits leave no feasible interval for {name}")
            source_limit = self.joints[name].limit
            result[name] = JointLimit(
                lower,
                upper,
                None if source_limit is None else source_limit.effort,
                None if source_limit is None else source_limit.velocity,
            )
        return result

    @property
    def independent_joint_limits(self) -> Mapping[str, JointLimit]:
        return self._independent_affine_limits

    def joint_limit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(
            [self.independent_joint_limits[name].lower for name in self.independent_joint_names],
            dtype=np.float64,
        )
        upper = np.asarray(
            [self.independent_joint_limits[name].upper for name in self.independent_joint_names],
            dtype=np.float64,
        )
        return lower, upper

    def resolve_joint_positions(
        self,
        positions: Mapping[str, float] | Sequence[float],
        *,
        enforce_limits: bool = True,
        tolerance: float | None = None,
    ) -> Mapping[str, float]:
        """Resolve independent positions and all mimic joints deterministically."""

        supplied: dict[str, float]
        if isinstance(positions, Mapping):
            supplied = {str(name): float(value) for name, value in positions.items()}
            movable_names = {
                name for name in self.joint_order if self.joints[name].movable
            }
            unknown = set(supplied) - movable_names
            if unknown:
                raise HandModelError(f"unknown hand joint positions: {sorted(unknown)}")
            missing = set(self.independent_joint_names) - set(supplied)
            if missing:
                raise HandModelError(f"missing independent joint positions: {sorted(missing)}")
        else:
            values = tuple(float(value) for value in positions)
            if len(values) != len(self.independent_joint_names):
                raise HandModelError(
                    f"expected {len(self.independent_joint_names)} joint values, got {len(values)}"
                )
            supplied = dict(zip(self.independent_joint_names, values))
        if any(not math.isfinite(value) for value in supplied.values()):
            raise HandModelError("joint positions must be finite")
        if tolerance is not None and (
            not math.isfinite(float(tolerance)) or float(tolerance) < 0.0
        ):
            raise HandModelError("explicit joint tolerance must be finite and non-negative")

        cache: dict[str, tuple[str, float, float]] = {}
        resolved: dict[str, float] = {}
        for name in self.joint_order:
            joint = self.joints[name]
            if not joint.movable:
                resolved[name] = 0.0
                continue
            source, multiplier, offset = self._joint_affine_map(name, cache, set())
            value = multiplier * supplied[source] + offset
            limit_values: tuple[float, ...] = ()
            if joint.limit is not None:
                limit_values = (joint.limit.lower, joint.limit.upper)
            joint_tolerance = (
                float(tolerance)
                if tolerance is not None
                else _scaled_float_tolerance(value, supplied[source], *limit_values)
            )
            if name in supplied and name not in self.independent_joint_names:
                if not math.isclose(
                    supplied[name], value, rel_tol=0.0, abs_tol=joint_tolerance
                ):
                    raise HandModelError(f"supplied mimic value is inconsistent for {name}")
            if enforce_limits and joint.limit is not None and not joint.limit.contains(
                value, tolerance=joint_tolerance
            ):
                raise HandModelError(
                    f"joint {name} position {value:.12g} violates "
                    f"[{joint.limit.lower:.12g}, {joint.limit.upper:.12g}]"
                )
            resolved[name] = value
        return MappingProxyType(resolved)

    def within_joint_limits(
        self,
        positions: Mapping[str, float] | Sequence[float],
        *,
        tolerance: float | None = None,
    ) -> bool:
        try:
            self.resolve_joint_positions(
                positions, enforce_limits=True, tolerance=tolerance
            )
        except HandModelError:
            return False
        return True

    def resolve_joint_velocities(
        self,
        velocities: Mapping[str, float] | Sequence[float],
        *,
        enforce_limits: bool = True,
    ) -> Mapping[str, float]:
        """Resolve independent and mimic velocities using URDF couplings."""

        if isinstance(velocities, Mapping):
            supplied = {str(name): float(value) for name, value in velocities.items()}
            movable_names = {
                name for name in self.joint_order if self.joints[name].movable
            }
            unknown = set(supplied) - movable_names
            if unknown:
                raise HandModelError(f"unknown hand joint velocities: {sorted(unknown)}")
            missing = set(self.independent_joint_names) - set(supplied)
            if missing:
                raise HandModelError(f"missing independent joint velocities: {sorted(missing)}")
        else:
            values = tuple(float(value) for value in velocities)
            if len(values) != len(self.independent_joint_names):
                raise HandModelError(
                    f"expected {len(self.independent_joint_names)} joint velocities, "
                    f"got {len(values)}"
                )
            supplied = dict(zip(self.independent_joint_names, values))
        if any(not math.isfinite(value) for value in supplied.values()):
            raise HandModelError("joint velocities must be finite")

        cache: dict[str, tuple[str, float, float]] = {}
        resolved: dict[str, float] = {}
        for name in self.joint_order:
            joint = self.joints[name]
            if not joint.movable:
                resolved[name] = 0.0
                continue
            source, multiplier, _offset = self._joint_affine_map(name, cache, set())
            value = multiplier * supplied[source]
            if name in supplied and name not in self.independent_joint_names:
                scale = max(abs(value), abs(supplied[name]), np.finfo(np.float64).tiny)
                tolerance = 64.0 * np.finfo(np.float64).eps * scale
                if not math.isclose(
                    supplied[name], value, rel_tol=0.0, abs_tol=tolerance
                ):
                    raise HandModelError(f"supplied mimic velocity is inconsistent for {name}")
            velocity_limit = None if joint.limit is None else joint.limit.velocity
            if enforce_limits and velocity_limit is not None:
                tolerance = 64.0 * np.finfo(np.float64).eps * velocity_limit
                if abs(value) > velocity_limit + tolerance:
                    raise HandModelError(
                        f"joint {name} velocity {value:.12g} exceeds {velocity_limit:.12g}"
                    )
            resolved[name] = value
        return MappingProxyType(resolved)

    def forward_kinematics(
        self,
        positions: Mapping[str, float] | Sequence[float],
        *,
        base_transform: np.ndarray | None = None,
        enforce_limits: bool = True,
    ) -> Mapping[str, np.ndarray]:
        """Return base-to-link transforms for the complete hand subtree."""

        resolved = self.resolve_joint_positions(
            positions, enforce_limits=enforce_limits
        )
        base = np.eye(4, dtype=np.float64) if base_transform is None else np.asarray(
            base_transform, dtype=np.float64
        )
        if base.shape != (4, 4) or not np.all(np.isfinite(base)):
            raise HandModelError("base_transform must be a finite 4x4 matrix")
        transforms: dict[str, np.ndarray] = {self.base_link: base.copy()}
        for name in self.joint_order:
            joint = self.joints[name]
            parent_transform = transforms.get(joint.parent_link)
            if parent_transform is None:
                raise HandModelError(f"missing parent transform for joint {name}")
            transforms[joint.child_link] = (
                parent_transform
                @ joint.origin_transform()
                @ joint.motion_transform(resolved[name])
            )
        return MappingProxyType(transforms)

    def pad_transforms(
        self,
        positions: Mapping[str, float] | Sequence[float],
        *,
        base_transform: np.ndarray | None = None,
    ) -> Mapping[str, np.ndarray]:
        links = self.forward_kinematics(positions, base_transform=base_transform)
        return MappingProxyType(
            {
                name: links[pad.link_name] @ pad.link_from_pad_transform()
                for name, pad in self.pads.items()
            }
        )

    def pad_contact_normals(
        self,
        positions: Mapping[str, float] | Sequence[float],
        *,
        base_transform: np.ndarray | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Return contact normals; reject PADs whose contract omits semantics."""

        transforms = self.pad_transforms(positions, base_transform=base_transform)
        normals: dict[str, np.ndarray] = {}
        for name, pad in self.pads.items():
            if pad.contact_normal_pad is None:
                raise HandModelError(
                    f"PAD {name} has no explicit contact normal in its geometry contract"
                )
            normals[name] = transforms[name][:3, :3] @ np.asarray(
                pad.contact_normal_pad, dtype=np.float64
            )
        return MappingProxyType(normals)

    def pad_kinematic_normal_domains(
        self,
        positions: Mapping[str, float] | Sequence[float],
        closing_joint_velocities: Mapping[str, float] | Sequence[float],
        *,
        base_transform: np.ndarray | None = None,
    ) -> Mapping[str, KinematicNormalDomain]:
        """Derive feasible object-normal half-spaces from closing kinematics."""

        velocities = self.resolve_joint_velocities(closing_joint_velocities)
        independent_velocity = np.asarray(
            [velocities[name] for name in self.independent_joint_names],
            dtype=np.float64,
        )
        pad_transforms = self.pad_transforms(
            positions, base_transform=base_transform
        )
        domains: dict[str, KinematicNormalDomain] = {}
        for name, pad in self.pads.items():
            jacobian = self.geometric_jacobian(
                pad.link_name,
                positions,
                point_local_m=pad.origin_xyz_m,
                base_transform=base_transform,
            )
            velocity = jacobian[:3] @ independent_velocity
            velocity_norm = float(np.linalg.norm(velocity))
            numerical_tolerance = (
                64.0 * np.finfo(np.float64).eps * velocity_norm
            )
            domains[name] = KinematicNormalDomain(
                pad_name=name,
                pad_origin_base_m=tuple(
                    float(item) for item in pad_transforms[name][:3, 3]
                ),
                closing_velocity_base_m_s=tuple(float(item) for item in velocity),
                numerical_tolerance_m_s=numerical_tolerance,
            )
        return MappingProxyType(domains)

    def geometric_jacobian(
        self,
        link_name: str,
        positions: Mapping[str, float] | Sequence[float],
        *,
        point_local_m: Sequence[float] = (0.0, 0.0, 0.0),
        base_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a 6-by-N geometric Jacobian in independent-joint order."""

        transforms = self.forward_kinematics(positions, base_transform=base_transform)
        if link_name not in transforms:
            raise HandModelError(f"link is outside hand subtree: {link_name}")
        point_local = np.asarray(
            _vector(
                point_local_m,
                length=3,
                default=(0.0, 0.0, 0.0),
                label="point_local_m",
            ),
            dtype=np.float64,
        )
        point_base = (
            transforms[link_name][:3, :3] @ point_local
            + transforms[link_name][:3, 3]
        )

        ancestor_names: list[str] = []
        cursor = link_name
        while cursor != self.base_link:
            joint_name = self._joint_by_child.get(cursor)
            if joint_name is None:
                raise HandModelError(f"link {link_name} is disconnected from hand base")
            ancestor_names.append(joint_name)
            cursor = self.joints[joint_name].parent_link
        ancestor_names.reverse()

        jacobian = np.zeros((6, len(self.independent_joint_names)), dtype=np.float64)
        column_by_name = {
            name: index for index, name in enumerate(self.independent_joint_names)
        }
        affine_cache: dict[str, tuple[str, float, float]] = {}
        for name in ancestor_names:
            joint = self.joints[name]
            if not joint.movable:
                continue
            joint_frame = transforms[joint.parent_link] @ joint.origin_transform()
            axis_base = joint_frame[:3, :3] @ np.asarray(joint.axis, dtype=np.float64)
            source, multiplier, _offset = self._joint_affine_map(
                name, affine_cache, set()
            )
            column = column_by_name[source]
            if joint.joint_type in ("revolute", "continuous"):
                linear = np.cross(axis_base, point_base - joint_frame[:3, 3])
                angular = axis_base
            else:
                linear = axis_base
                angular = np.zeros(3, dtype=np.float64)
            jacobian[:3, column] += multiplier * linear
            jacobian[3:, column] += multiplier * angular
        return jacobian


__all__ = [
    "FingerChain",
    "GeometrySpec",
    "HandModelError",
    "JointLimit",
    "JointSpec",
    "KinematicNormalDomain",
    "MimicSpec",
    "PadGeometry",
    "ThreeFingerHandModel",
    "rigid_transform",
    "rpy_rotation",
]
