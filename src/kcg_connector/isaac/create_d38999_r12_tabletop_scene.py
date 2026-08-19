#!/usr/bin/env python3

"""Create the formal or one bounded candidate r12 tabletop scene config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import yaml

from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
    R12_ASSET_DIRECTORY,
    R12_ASSET_NAME,
    R12_CONTRACT_PATH,
    R12_PAIR_MODEL_ID,
    R12_SCENE_CONFIG_PATH,
    candidate_asset_relative_path,
    candidate_scene_relative_path,
)


R11_SCENE = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v2_tabletop_scene_v1.yaml"
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--official", action="store_true")
    mode.add_argument("--candidate-index", type=int)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def build_scene_document(candidate_index: int | None = None) -> dict:
    document = deepcopy(yaml.safe_load(R11_SCENE.read_text(encoding="utf-8")))
    document["schema_version"] = "kcg_d38999_keyed_v3_tabletop_scene_r12_v1"
    document["asset_profile"].update(
        {
            "profile_id": R12_PAIR_MODEL_ID,
            "source_config": str(R12_CONTRACT_PATH.relative_to(Path(__file__).resolve().parents[3])),
            "expected_body_collider_count": 7438,
            "expected_nut_collider_count": 204,
        }
    )
    if candidate_index is None:
        local_path = Path(R12_ASSET_DIRECTORY) / R12_ASSET_NAME
    else:
        local_path = candidate_asset_relative_path(candidate_index)
    document["asset"]["local_path"] = str(local_path)
    document["asset"]["proxy_id"] = R12_PAIR_MODEL_ID
    document["physics"]["maximum_fixed_translation_drift_m"] = 0.000005
    return document


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    candidate_index = None if arguments.official else arguments.candidate_index
    default = (
        R12_SCENE_CONFIG_PATH
        if candidate_index is None
        else Path(__file__).resolve().parents[3]
        / candidate_scene_relative_path(candidate_index)
    )
    output = Path(arguments.output).expanduser().resolve() if arguments.output else default.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite r12 scene config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            build_scene_document(candidate_index),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"created={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
