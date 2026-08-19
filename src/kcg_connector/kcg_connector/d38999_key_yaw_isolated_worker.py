#!/usr/bin/env python3

"""Minimal child-side protocol for the D38999 key-yaw bwrap evaluator.

This file intentionally has no dependency on the repository package.  The
parent mounts only this file, one predictor file, and anonymous observation
directories into the sandbox.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


PREDICTOR_RESULT_FIELDS = frozenset(
    {
        "passed",
        "estimated_axial_yaw_rad",
        "selected_hypothesis_id",
        "shadow_only",
        "control_authorized",
    }
)
EXPECTED_C2_IDS = frozenset({"YAW_0", "YAW_PI"})


def _load_predictor(path: Path):
    spec = importlib.util.spec_from_file_location("isolated_predictor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("predictor module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = getattr(module, "predict", None)
    if not callable(predictor):
        raise ValueError("predictor must define callable predict(sample_directory)")
    return predictor


def _validated_result(value: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PREDICTOR_RESULT_FIELDS:
        raise ValueError("predictor result must contain exactly the sealed fields")
    if type(value["passed"]) is not bool:
        raise ValueError("predictor result passed must be boolean")
    if value["shadow_only"] is not True or value["control_authorized"] is not False:
        raise ValueError("predictor attempted to relax the authorization boundary")
    estimate = value["estimated_axial_yaw_rad"]
    hypothesis = value["selected_hypothesis_id"]
    if value["passed"]:
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            raise ValueError("passed prediction estimate must be numeric")
        estimate = float(estimate)
        if not math.isfinite(estimate):
            raise ValueError("passed prediction estimate must be finite")
        if hypothesis not in EXPECTED_C2_IDS:
            raise ValueError("passed prediction must select one C2 hypothesis")
    elif estimate is not None or hypothesis is not None:
        raise ValueError("rejected prediction must not retain a yaw or branch")
    return {
        "sample_id": sample_id,
        "passed": value["passed"],
        "estimated_axial_yaw_rad": estimate,
        "selected_hypothesis_id": hypothesis,
        "shadow_only": True,
        "control_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictor", required=True, type=Path)
    arguments = parser.parse_args()
    predictor = _load_predictor(arguments.predictor)
    sample_directories = sorted(
        item for item in arguments.input.iterdir() if item.is_dir()
    )
    if not sample_directories:
        raise ValueError("anonymous input is empty")
    with arguments.output.open("x", encoding="utf-8") as stream:
        for sample_directory in sample_directories:
            record = _validated_result(
                predictor(sample_directory), sample_directory.name
            )
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
