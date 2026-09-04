#!/usr/bin/env python3

"""Run the TE-specific ordinary-RGB-D pose provider without Isaac/Kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser(
        description="Run the truth-firewalled TE RGB-D pose provider offline"
    )
    parser.add_argument("--provider-input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[3]
    provider_input = Path(arguments.provider_input).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    try:
        output.relative_to(repository)
    except ValueError as error:
        raise ValueError("provider output must remain inside the repository") from error
    if output.exists():
        raise ValueError("provider output already exists; overwrite is forbidden")

    from kcg_connector.te_rgbd_pose_provider import run_te_rgbd_pose_provider

    result = run_te_rgbd_pose_provider(provider_input, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "control_allowed": result["control_allowed"],
                "robot_command_count": result["robot_command_count"],
                "output": str(output.relative_to(repository)),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
