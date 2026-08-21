#!/usr/bin/env python3
"""Independently validate generated J599 OpenUSD topology and truth fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = MODEL_ROOT / "config" / "model_contract.json"
DEFAULT_CONTACTS = MODEL_ROOT / "data" / "contact_positions_25_35.csv"
DEFAULT_ASSEMBLY = MODEL_ROOT / "generated" / "j599_25_35_pair_assembly.usda"
DEFAULT_BINARY = MODEL_ROOT / "generated" / "j599_25_35_pair_assembly.usdc"
DEFAULT_REPORT = MODEL_ROOT / "evidence" / "static_validation.json"
PAIR_ROOT = "/World/J599_25_35_N_Pair"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    with args.contacts.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_ids = list(range(1, 129))
    table_ids = [int(row["contact_id"]) for row in rows]

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(args.assembly.resolve()))
    binary_stage = Usd.Stage.Open(str(args.binary.resolve()))
    if stage is None or binary_stage is None:
        raise RuntimeError("OpenUSD could not open one or both assembly assets")
    root = stage.GetPrimAtPath(PAIR_ROOT)
    if not root:
        raise RuntimeError("pair root is missing")

    pin_ids: list[int] = []
    socket_ids: list[int] = []
    contact_collision_paths: list[str] = []
    key_paths: list[str] = []
    keyway_paths: set[str] = set()
    collision_roles: dict[str, int] = {}
    robot_or_hand_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lowered = path.lower()
        if any(
            token in lowered
            for token in ("robot", "finger", "hand", "iiwa", "gripper")
        ):
            robot_or_hand_paths.append(path)
        contact_id = prim.GetCustomDataByKey("j599:contactId")
        if contact_id is not None:
            if "/Contacts/Pins/" in path:
                pin_ids.append(int(contact_id))
            if "/Contacts/Sockets/" in path:
                socket_ids.append(int(contact_id))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                contact_collision_paths.append(path)
        if (
            "/Keys/Key_" in path
            and prim.GetCustomDataByKey("j599:keyIndex") is not None
        ):
            key_paths.append(path)
        if "/Physics/Keyways/Keyway_" in path:
            parts = path.split("/")
            keyway_paths.add("/".join(parts[: parts.index("Keyways") + 2]))
        role = prim.GetCustomDataByKey("j599:collisionRole")
        if role:
            collision_roles[str(role)] = collision_roles.get(str(role), 0) + 1

    metadata = {
        "hardware_authorized": root.GetCustomDataByKey(
            "j599:hardwareAuthorized"
        ),
        "hardware_exact_fidelity": root.GetCustomDataByKey(
            "j599:hardwareExactFidelity"
        ),
        "contact_count": root.GetCustomDataByKey("j599:contactCount"),
        "polarization": root.GetCustomDataByKey("j599:polarization"),
        "representation": root.GetCustomDataByKey("j599:representation"),
    }
    checks: dict[str, bool] = {
        "contact_table_ids_exact": table_ids == expected_ids,
        "pin_ids_exact": sorted(pin_ids) == expected_ids,
        "socket_ids_exact": sorted(socket_ids) == expected_ids,
        "contact_collision_api_count_zero": not contact_collision_paths,
        "five_keys_present": len(key_paths) == 5,
        "five_keyways_present": len(keyway_paths) == 5,
        "fixed_keyway_collision_present": collision_roles.get(
            "fixed_keyway_blocking_shell", 0
        )
        > 0,
        "plug_guide_collision_present": collision_roles.get(
            "plug_continuous_guide_shell", 0
        )
        > 0,
        "both_metal_stops_present": collision_roles.get(
            "fixed_metal_stop", 0
        )
        > 0
        and collision_roles.get("plug_metal_stop", 0) > 0,
        "plug_keys_have_collision": collision_roles.get(
            "plug_polarizing_key", 0
        )
        == 5,
        "coupling_nut_joint_present": bool(
            stage.GetPrimAtPath(
                PAIR_ROOT
                + "/LoosePlug_J599_26FJ35PN/CouplingNutRevolute"
            )
        ),
        "robot_or_hand_prim_count_zero": not robot_or_hand_paths,
        "hardware_authorized_false": metadata["hardware_authorized"] is False,
        "hardware_exact_fidelity_false": metadata[
            "hardware_exact_fidelity"
        ]
        is False,
        "root_contact_count_128": int(metadata["contact_count"]) == 128,
        "root_polarization_n": metadata["polarization"] == "N",
        "root_representation_assembly": metadata["representation"]
        == "assembly",
        "binary_root_present": bool(binary_stage.GetPrimAtPath(PAIR_ROOT)),
        "contract_hardware_authorized_false": contract["authorization"][
            "hardware_authorized"
        ]
        is False,
    }
    report: dict[str, Any] = {
        "schema_version": "j599_static_validation_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_PASS" if all(checks.values()) else "STATIC_FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "contact_table": len(rows),
            "plug_pins": len(pin_ids),
            "receptacle_sockets": len(socket_ids),
            "contact_collision_api": len(contact_collision_paths),
            "keys": len(key_paths),
            "keyways": len(keyway_paths),
            "robot_or_hand_prims": len(robot_or_hand_paths),
        },
        "collision_role_counts": dict(sorted(collision_roles.items())),
        "metadata": metadata,
        "unexpected_paths": {
            "contact_collision_paths": contact_collision_paths,
            "robot_or_hand_paths": robot_or_hand_paths,
        },
        "inputs": {
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": _sha256(args.contract),
            },
            "contacts": {
                "path": str(args.contacts.resolve()),
                "sha256": _sha256(args.contacts),
            },
            "assembly": {
                "path": str(args.assembly.resolve()),
                "sha256": _sha256(args.assembly),
            },
            "binary": {
                "path": str(args.binary.resolve()),
                "sha256": _sha256(args.binary),
            },
        },
        "truth_boundary": {
            "static_pass_is_dynamic_pass": False,
            "simulation_is_hardware_acceptance": False,
            "hardware_authorized": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
