"""Load immutable D38999 contract snapshots from compact package data.

The values remain independent from the editable YAML contracts and are still
compared as complete nested Python mappings by the model and acceptance
validators.  Keeping the large literal snapshot in a compressed data file
prevents 470+ KiB of static data from being indexed as executable Python.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any


_SNAPSHOT_FILENAME = "d38999_keyed_v2_frozen_contract_snapshot_v1.json.gz"
_SOURCE_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / _SNAPSHOT_FILENAME
)
_EXPECTED_NAMES = frozenset(
    {
        "FROZEN_A0_RESOLVED_DECISIONS",
        "FROZEN_A0_RESOLVED_SOURCE_MAPPINGS",
        "FROZEN_A2_COLLISION_AUTHORING_BLUEPRINT",
        "FROZEN_ACCEPTANCE_BENCHES",
        "FROZEN_ACCEPTANCE_IDENTITY_AND_EVIDENCE",
        "FROZEN_ACCEPTANCE_PHASE_RELEASE",
        "FROZEN_ACCEPTANCE_SHARED_NUMERIC_PROFILE",
        "FROZEN_CONVEX_COOKING_REPRESENTATION",
        "FROZEN_MODEL_IMMUTABLE_SECTIONS",
        "FROZEN_REALIZED_ROBOT_HAND_FIXTURE_BLUEPRINT",
    }
)


def _snapshot_candidates() -> tuple[Path, ...]:
    """Return source-tree and installed ROS share candidates in priority order."""

    prefixes: list[Path] = []
    for raw_prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if raw_prefix:
            prefixes.append(Path(raw_prefix))
    prefixes.append(Path(sys.prefix))

    module_path = Path(__file__).resolve()
    if len(module_path.parents) > 4:
        prefixes.append(module_path.parents[4])

    candidates = [_SOURCE_SNAPSHOT_PATH]
    candidates.extend(
        prefix / "share" / "kcg_connector" / "config" / _SNAPSHOT_FILENAME
        for prefix in prefixes
    )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _resolve_snapshot_path() -> Path:
    candidates = _snapshot_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"frozen contract snapshot is missing; searched: {searched}")


def _load_snapshot() -> dict[str, Any]:
    snapshot_path = _resolve_snapshot_path()
    try:
        with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen contract snapshot cannot be decoded") from error
    if not isinstance(document, dict) or set(document) != _EXPECTED_NAMES:
        raise RuntimeError("frozen contract snapshot has an invalid top-level schema")
    return document


_SNAPSHOT = _load_snapshot()

FROZEN_A0_RESOLVED_DECISIONS = _SNAPSHOT["FROZEN_A0_RESOLVED_DECISIONS"]
FROZEN_A0_RESOLVED_SOURCE_MAPPINGS = _SNAPSHOT[
    "FROZEN_A0_RESOLVED_SOURCE_MAPPINGS"
]
FROZEN_A2_COLLISION_AUTHORING_BLUEPRINT = _SNAPSHOT[
    "FROZEN_A2_COLLISION_AUTHORING_BLUEPRINT"
]
FROZEN_ACCEPTANCE_BENCHES = _SNAPSHOT["FROZEN_ACCEPTANCE_BENCHES"]
FROZEN_ACCEPTANCE_IDENTITY_AND_EVIDENCE = _SNAPSHOT[
    "FROZEN_ACCEPTANCE_IDENTITY_AND_EVIDENCE"
]
FROZEN_ACCEPTANCE_PHASE_RELEASE = _SNAPSHOT[
    "FROZEN_ACCEPTANCE_PHASE_RELEASE"
]
FROZEN_ACCEPTANCE_SHARED_NUMERIC_PROFILE = _SNAPSHOT[
    "FROZEN_ACCEPTANCE_SHARED_NUMERIC_PROFILE"
]
FROZEN_CONVEX_COOKING_REPRESENTATION = _SNAPSHOT[
    "FROZEN_CONVEX_COOKING_REPRESENTATION"
]
FROZEN_MODEL_IMMUTABLE_SECTIONS = _SNAPSHOT[
    "FROZEN_MODEL_IMMUTABLE_SECTIONS"
]
FROZEN_REALIZED_ROBOT_HAND_FIXTURE_BLUEPRINT = _SNAPSHOT[
    "FROZEN_REALIZED_ROBOT_HAND_FIXTURE_BLUEPRINT"
]

__all__ = sorted(_EXPECTED_NAMES)
