"""Fail-closed entry point for future D38999 full-skill training.

This module intentionally contains no training backend.  The readiness guard
is the first operational call after argument parsing; only a passing guard may
reach the backend loader.  Output directories are never created here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .full_skill_readiness import require_training_ready


EXIT_BLOCKED = 1
EXIT_INVALID = 2
EXIT_BACKEND_NOT_IMPLEMENTED = 3


class FullSkillBackendNotImplementedError(RuntimeError):
    """A passing readiness gate has no implemented training backend yet."""


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / (
        "d38999_full_skill_rl_readiness_v1.yaml"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate-first full-skill D38999 training entry point"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="strict full-skill readiness manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used by the readiness validator",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="reserved future training output; never created by this stub",
    )
    return parser


def _load_training_backend() -> None:
    """Future post-gate import point; currently fails closed explicitly."""
    raise FullSkillBackendNotImplementedError(
        "full-skill readiness passed, but the training backend is not "
        "implemented; no training was started and no output was created"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    # Do not move any backend import, output creation or training setup above
    # this call.  A failing guard must have no runtime side effects.
    try:
        require_training_ready(arguments.config, arguments.repo_root)
    except RuntimeError as error:
        print(f"FULL SKILL RL TRAINING: BLOCKED\n{error}", file=sys.stderr)
        return EXIT_BLOCKED
    except (OSError, ValueError) as error:
        print(f"FULL SKILL RL TRAINING: INVALID\n{error}", file=sys.stderr)
        return EXIT_INVALID

    try:
        _load_training_backend()
    except FullSkillBackendNotImplementedError as error:
        print(
            "FULL SKILL RL TRAINING: "
            "READY_GATE_PASSED_BACKEND_NOT_IMPLEMENTED\n"
            f"{error}",
            file=sys.stderr,
        )
        return EXIT_BACKEND_NOT_IMPLEMENTED
    raise AssertionError(
        "unreachable backend loader returned without training"
    )


if __name__ == "__main__":  # pragma: no cover - subprocess tested
    raise SystemExit(main())


__all__ = [
    "EXIT_BACKEND_NOT_IMPLEMENTED",
    "EXIT_BLOCKED",
    "EXIT_INVALID",
    "FullSkillBackendNotImplementedError",
    "build_argument_parser",
    "main",
]
