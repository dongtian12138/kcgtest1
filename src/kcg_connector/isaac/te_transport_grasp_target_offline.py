#!/usr/bin/env python3

"""Build or compare truth-free TE visual transport targets offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation")
    parser.add_argument("--provider-result")
    parser.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if (arguments.compare is None) == (
        arguments.relation is None or arguments.provider_result is None
    ):
        parser.error(
            "use either --relation plus --provider-result, or --compare"
        )
    repository = Path(__file__).resolve().parents[3]
    output = Path(arguments.output).expanduser().resolve()
    try:
        output.relative_to(repository)
    except ValueError as error:
        raise ValueError("output must remain inside the repository") from error
    if output.exists():
        raise ValueError("output already exists; overwrite is forbidden")

    from kcg_connector.te_transport_grasp_target import (
        build_visual_transport_target,
        compare_visual_transport_targets,
        load_visual_transport_target,
    )

    if arguments.compare is not None:
        first, _ = load_visual_transport_target(arguments.compare[0], repository)
        second, _ = load_visual_transport_target(arguments.compare[1], repository)
        result = compare_visual_transport_targets(first, second)
    else:
        result = build_visual_transport_target(
            provider_result_path=arguments.provider_result,
            relation_path=arguments.relation,
            repository_root=repository,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.relative_to(repository))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
