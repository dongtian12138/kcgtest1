"""Fail-closed audit of the KUKA/hand self-collision configuration.

This module deliberately has no ROS, MoveIt, Isaac Sim, or USD Python
dependency.  It inventories the collision-bearing URDF links, the SRDF
allowed-collision matrix entries, and the two persisted Isaac self-collision
switches without changing any of those inputs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable
import xml.etree.ElementTree as ET


KNOWN_DISABLED_REASONS = frozenset(("Adjacent", "Never"))
_FINGER_LINK = re.compile(r"^f([123])Link[123]$")
_ISAAC_SELF_COLLISION = re.compile(
    r"\b(?P<attribute>newton:selfCollisionEnabled|"
    r"physxArticulation:enabledSelfCollisions)\s*=\s*"
    r"(?P<value>0|1|false|true)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditInputs:
    """Paths needed to reproduce one audit."""

    urdf: Path
    srdf: Path
    isaac_importer: Path
    isaac_physics_usd: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_inputs() -> AuditInputs:
    """Return repository-local authoritative inputs."""
    root = _project_root()
    return AuditInputs(
        urdf=root / "artifacts/kcg_connector/urdf/handarm.urdf",
        srdf=root / "src/kcg_moveit1/config/handarm.srdf",
        isaac_importer=(
            root / "src/kcg_connector/isaac/import_robot.py"
        ),
        isaac_physics_usd=(
            root
            / "artifacts/kcg_connector/isaac/robot/handarm/payloads/"
            "Physics/physics.usda"
        ),
    )


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(link1: str, link2: str) -> tuple[str, str]:
    if link1 == link2:
        raise ValueError(f"self-pair is invalid: {link1}")
    return tuple(sorted((link1, link2)))


def _all_pairs(links: Iterable[str]) -> set[tuple[str, str]]:
    ordered = sorted(set(links))
    return {
        (ordered[first], ordered[second])
        for first in range(len(ordered))
        for second in range(first + 1, len(ordered))
    }


def _finger_number(link: str) -> str | None:
    match = _FINGER_LINK.fullmatch(link)
    return match.group(1) if match else None


def _never_category(link1: str, link2: str) -> str:
    arm1 = link1.startswith("iiwa_link_")
    arm2 = link2.startswith("iiwa_link_")
    finger1 = _finger_number(link1)
    finger2 = _finger_number(link2)
    base1 = link1 == "handbase_link"
    base2 = link2 == "handbase_link"

    if arm1 and arm2:
        return "arm_arm"
    if (arm1 and (finger2 or base2)) or (arm2 and (finger1 or base1)):
        return "arm_hand"
    if finger1 and finger2:
        if finger1 == finger2:
            return "intra_finger"
        return "inter_finger"
    if (finger1 and base2) or (finger2 and base1):
        return "finger_handbase"
    return "other"


def _parse_urdf(path: Path) -> tuple[list[str], list[str]]:
    robot = ET.parse(path).getroot()
    if robot.tag != "robot":
        raise ValueError(f"URDF root must be <robot>, got <{robot.tag}>")
    links = [node.attrib["name"] for node in robot.findall("link")]
    if len(links) != len(set(links)):
        raise ValueError("URDF contains duplicate link names")
    collision_links = [
        node.attrib["name"]
        for node in robot.findall("link")
        if node.findall("collision")
    ]
    if len(collision_links) < 2:
        raise ValueError("URDF must contain at least two collision links")
    return links, collision_links


def _parse_srdf(path: Path) -> list[dict[str, str]]:
    robot = ET.parse(path).getroot()
    if robot.tag != "robot":
        raise ValueError(f"SRDF root must be <robot>, got <{robot.tag}>")
    entries = []
    for node in robot.findall("disable_collisions"):
        entries.append(
            {
                "link1": node.attrib["link1"],
                "link2": node.attrib["link2"],
                "reason": node.attrib.get("reason", ""),
            }
        )
    return entries


def _importer_allow_self_collision(path: Path) -> bool | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else ""
        if isinstance(function, ast.Attribute):
            name = function.attr
        if name != "URDFImporterConfig":
            continue
        for keyword in node.keywords:
            if keyword.arg != "allow_self_collision":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, bool
            ):
                values.append(keyword.value.value)
            else:
                values.append(None)
    unique = set(values)
    if len(values) != 1 or len(unique) != 1:
        return None
    return values[0]


def _usd_self_collision_attributes(path: Path) -> dict[str, bool]:
    attributes = {}
    text = path.read_text(encoding="utf-8")
    for match in _ISAAC_SELF_COLLISION.finditer(text):
        value = match.group("value").lower() in ("1", "true")
        attributes[match.group("attribute")] = value
    return dict(sorted(attributes.items()))


def audit_self_collision(inputs: AuditInputs | None = None) -> dict:
    """Return a JSON-serializable, fail-closed self-collision report."""
    selected = inputs or default_inputs()
    paths = AuditInputs(
        urdf=_require_file(selected.urdf, "URDF"),
        srdf=_require_file(selected.srdf, "SRDF"),
        isaac_importer=_require_file(
            selected.isaac_importer, "Isaac importer"
        ),
        isaac_physics_usd=_require_file(
            selected.isaac_physics_usd, "Isaac physics USD"
        ),
    )

    urdf_links, collision_links = _parse_urdf(paths.urdf)
    candidates = _all_pairs(collision_links)
    srdf_entries = _parse_srdf(paths.srdf)
    urdf_link_set = set(urdf_links)
    collision_link_set = set(collision_links)

    unknown_links = sorted(
        {
            link
            for entry in srdf_entries
            for link in (entry["link1"], entry["link2"])
            if link not in urdf_link_set
        }
    )
    noncollision_references = sorted(
        {
            link
            for entry in srdf_entries
            for link in (entry["link1"], entry["link2"])
            if link in urdf_link_set and link not in collision_link_set
        }
    )

    normalized_entries = []
    invalid_self_pairs = []
    for entry in srdf_entries:
        if entry["link1"] == entry["link2"]:
            invalid_self_pairs.append(entry["link1"])
            continue
        normalized_entries.append(
            (_pair(entry["link1"], entry["link2"]), entry["reason"])
        )
    pair_frequency = Counter(pair for pair, _ in normalized_entries)
    duplicate_pairs = sorted(
        pair for pair, count in pair_frequency.items() if count > 1
    )
    unsupported_reasons = sorted(
        {
            reason
            for _, reason in normalized_entries
            if reason not in KNOWN_DISABLED_REASONS
        }
    )

    candidate_entries = [
        (pair, reason)
        for pair, reason in normalized_entries
        if pair in candidates
    ]
    disabled_candidates = {pair for pair, _ in candidate_entries}
    default_pairs = candidates - disabled_candidates
    reason_counts = Counter(reason for _, reason in candidate_entries)

    never_groups: dict[str, list[str]] = defaultdict(list)
    for pair, reason in candidate_entries:
        if reason != "Never":
            continue
        category = _never_category(*pair)
        never_groups[category].append(f"{pair[0]} <-> {pair[1]}")
    never_groups = {
        category: sorted(pairs)
        for category, pairs in sorted(never_groups.items())
    }

    importer_switch = _importer_allow_self_collision(
        paths.isaac_importer
    )
    usd_switches = _usd_self_collision_attributes(
        paths.isaac_physics_usd
    )

    blockers = []
    if reason_counts["Never"]:
        blockers.append(
            "SRDF disables collision checking for reason=Never pairs"
        )
    if unknown_links:
        blockers.append("SRDF references links absent from the URDF")
    if noncollision_references:
        blockers.append(
            "SRDF references URDF links without collision geometry"
        )
    if invalid_self_pairs:
        blockers.append("SRDF contains invalid same-link pairs")
    if duplicate_pairs:
        blockers.append("SRDF contains duplicate normalized pairs")
    if unsupported_reasons:
        blockers.append("SRDF contains unsupported disable reasons")
    if importer_switch is not True:
        blockers.append("Isaac URDF importer does not enable self collision")
    if not usd_switches or not all(usd_switches.values()):
        blockers.append("Isaac robot USD does not enable self collision")

    pair_counts = {
        "Never": reason_counts["Never"],
        "Adjacent": reason_counts["Adjacent"],
        "Default": len(default_pairs),
    }
    report = {
        "schema": "kcg.self_collision_audit.v1",
        "status": (
            "PASS_SELF_COLLISION_CONFIGURATION"
            if not blockers
            else "FAIL_CLOSED_UNVERIFIED"
        ),
        "self_collision_verified": not blockers,
        "full_path_self_collision_claim_allowed": not blockers,
        "blockers": blockers,
        "urdf": {
            "link_count": len(urdf_links),
            "collision_link_count": len(collision_links),
            "collision_links": sorted(collision_links),
            "noncollision_links": sorted(
                set(urdf_links) - collision_link_set
            ),
        },
        "pair_inventory": {
            "candidate_collision_pair_count": len(candidates),
            "srdf_disabled_entry_count": len(srdf_entries),
            "srdf_disabled_collision_pair_count": len(
                disabled_candidates
            ),
            "classification_counts": pair_counts,
            "classification_sum": sum(pair_counts.values()),
        },
        "never_pair_categories": {
            category: len(pairs)
            for category, pairs in never_groups.items()
        },
        "never_pairs_by_category": never_groups,
        "srdf_integrity": {
            "unknown_links": unknown_links,
            "noncollision_link_references": noncollision_references,
            "invalid_self_pairs": sorted(invalid_self_pairs),
            "duplicate_pairs": [list(pair) for pair in duplicate_pairs],
            "unsupported_reasons": unsupported_reasons,
        },
        "isaac": {
            "importer_allow_self_collision": importer_switch,
            "persisted_self_collision_attributes": usd_switches,
        },
        "inputs": {
            label: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for label, path in (
                ("urdf", paths.urdf),
                ("srdf", paths.srdf),
                ("isaac_importer", paths.isaac_importer),
                ("isaac_physics_usd", paths.isaac_physics_usd),
            )
        },
    }
    return report


def _human_report(report: dict) -> str:
    inventory = report["pair_inventory"]
    urdf = report["urdf"]
    isaac = report["isaac"]
    persisted_switches = json.dumps(
        isaac["persisted_self_collision_attributes"], sort_keys=True
    )
    lines = [
        "KCG KUKA + THREE-FINGER SELF-COLLISION AUDIT",
        f"status: {report['status']}",
        f"self_collision_verified: {report['self_collision_verified']}",
        (
            "URDF links: "
            f"{urdf['link_count']} total, "
            f"{urdf['collision_link_count']} collision-bearing"
        ),
        (
            "collision pairs: "
            f"{inventory['candidate_collision_pair_count']} total; "
            f"Never={inventory['classification_counts']['Never']}, "
            f"Adjacent={inventory['classification_counts']['Adjacent']}, "
            f"Default={inventory['classification_counts']['Default']}"
        ),
        (
            "Isaac importer allow_self_collision: "
            f"{isaac['importer_allow_self_collision']}"
        ),
        (
            "Isaac persisted switches: "
            f"{persisted_switches}"
        ),
        "Never categories:",
    ]
    for category, count in report["never_pair_categories"].items():
        lines.append(f"  {category}: {count}")
    lines.append("Never pairs (disabled from checking):")
    for category, pairs in report["never_pairs_by_category"].items():
        lines.append(f"  [{category}]")
        lines.extend(f"    {pair}" for pair in pairs)
    lines.append("Blockers:")
    lines.extend(f"  - {blocker}" for blocker in report["blockers"])
    lines.append(
        "decision: full-path self-collision safety MUST NOT be claimed"
        if not report["full_path_self_collision_claim_allowed"]
        else "decision: configuration gate passed"
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    defaults = default_inputs()
    parser = argparse.ArgumentParser(
        description=(
            "Audit SRDF and Isaac self-collision configuration without "
            "modifying project files."
        )
    )
    parser.add_argument("--urdf", type=Path, default=defaults.urdf)
    parser.add_argument("--srdf", type=Path, default=defaults.srdf)
    parser.add_argument(
        "--isaac-importer", type=Path, default=defaults.isaac_importer
    )
    parser.add_argument(
        "--isaac-physics-usd",
        type=Path,
        default=defaults.isaac_physics_usd,
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="return zero even when the fail-closed gate is not verified",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    inputs = AuditInputs(
        urdf=arguments.urdf,
        srdf=arguments.srdf,
        isaac_importer=arguments.isaac_importer,
        isaac_physics_usd=arguments.isaac_physics_usd,
    )
    try:
        report = audit_self_collision(inputs)
    except (OSError, ValueError, ET.ParseError, SyntaxError) as error:
        print(f"SELF-COLLISION AUDIT INPUT ERROR: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_human_report(report))
    if report["self_collision_verified"] or arguments.report_only:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
