#!/usr/bin/env python3

"""Opt-in launcher for the two-view D38999 axial tooth supplement.

The prepared physics runner is imported and executed unchanged.  Only this
process's module table maps its existing capture-class import to the adapter
in ``d38999_tooth_axial_capture``.  All original CLI arguments retain their
meaning; this launcher consumes exactly one additional output-directory flag.
"""

from __future__ import annotations

from pathlib import Path
import sys


AXIAL_OUTPUT_FLAG = "--nut-tooth-axial-capture-output"


def split_axial_arguments(argv):
    """Remove and return the one required axial-output option."""

    values = list(argv)
    indices = [
        index
        for index, value in enumerate(values)
        if value == AXIAL_OUTPUT_FLAG
    ]
    if len(indices) != 1:
        raise ValueError(f"exactly one {AXIAL_OUTPUT_FLAG} is required")
    index = indices[0]
    if index + 1 >= len(values) or values[index + 1].startswith("--"):
        raise ValueError(f"{AXIAL_OUTPUT_FLAG} requires a directory")
    output = values[index + 1]
    del values[index : index + 2]  # noqa: E203
    required = {
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        "--nut-tooth-sync-capture-output",
        "--nut-tooth-ghost-fingers-output",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(
            "axial supplement requires the baseline ghost capture flags: "
            + ",".join(missing)
        )
    forbidden = {
        "--full-rotation-probe",
        "--nut-tooth-jitter-disable-fabric-scene-delegate",
        "--nut-tooth-jitter-normalize-segment00-op",
        "--nut-tooth-jitter-rtx-history",
        "--rewind-probe",
    }
    present = sorted(forbidden & set(values))
    if present:
        raise ValueError(
            "axial v1 permits only the baseline prepared twist treatment: "
            + ",".join(present)
        )
    return output, values


def _option_value(arguments, option):
    """Return one required value from the already-validated base CLI."""

    indices = [
        index for index, value in enumerate(arguments) if value == option
    ]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise ValueError(f"exactly one {option} value is required")
    return arguments[indices[0] + 1]


def main(argv=None):
    """Install the process-local adapter, then call the original main."""

    actual = sys.argv[1:] if argv is None else list(argv)
    try:
        output, base_arguments = split_axial_arguments(actual)
    except ValueError as exception:
        print(f"AXIAL CAPTURE ARGUMENT ERROR: {exception}", file=sys.stderr)
        return 2

    import d38999_tooth_axial_capture as axial

    source_directory = Path(__file__).resolve().parent
    runner_path = source_directory / "d38999_nut_regrasp_smoke.py"
    axial.configure_axial_extension(
        output_directory=output,
        wrapper_source_path=Path(__file__).resolve(),
        runner_source_path=runner_path,
    )
    # The axial module imported the original base helper before this swap and
    # retains its class reference.  Only the runner's subsequent local import
    # sees the adapter, so existing files and package state remain untouched.
    previous_argv = sys.argv
    previous_capture_module = sys.modules.get(
        "d38999_tooth_sync_capture"
    )
    try:
        sys.argv = [str(runner_path), *base_arguments]
        sys.modules["d38999_tooth_sync_capture"] = axial
        import d38999_nut_regrasp_smoke as runner

        # The prepared runner intentionally owns argparse and accepts no
        # Python argument.  Its exact CLI is supplied through sys.argv only
        # for this call and restored even if Isaac raises during startup.
        result = int(runner.main())
    finally:
        sys.argv = previous_argv
        if previous_capture_module is None:
            sys.modules.pop("d38999_tooth_sync_capture", None)
        else:
            sys.modules[
                "d38999_tooth_sync_capture"
            ] = previous_capture_module
    if result != 0:
        return result
    try:
        axial.finalize_axial_ghost_bundle(
            axial_output=output,
            ghost_output=_option_value(
                base_arguments, "--nut-tooth-ghost-fingers-output"
            ),
        )
    except Exception as exception:
        print(
            "AXIAL GHOST BUNDLE FAILED: "
            f"{type(exception).__name__}: {exception}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
