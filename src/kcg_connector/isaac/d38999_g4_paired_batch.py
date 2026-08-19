#!/usr/bin/env python3
"""Outer paired-batch orchestrator for G4 synchronous/sequential pairs.

Pure stdlib, no Isaac import.  For every seed it launches two brand-new
Isaac processes (synchronous first, then sequential-compliant) via the
repository run_isaac_python.sh wrapper, captures each child's
stdout/stderr directly into a per-side side_console.log, and finalizes a
per-pair manifest atomically.  The first side physically failing never
prevents the second side from running.  Resume semantics: a finalized
pair is skipped only after both sides and every recorded hash re-verify;
an unfinished .inprogress pair may continue only while the current
source/config hashes are unchanged and every existing side re-verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "kcg_g4_paired_batch_v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_WRAPPER = (
    REPOSITORY_ROOT / "src/kcg_connector/isaac/run_isaac_python.sh"
)
DEFAULT_RUNNER = (
    REPOSITORY_ROOT
    / "src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py"
)

SOURCE_FILES = {
    "runner_sha256": "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py",
    "wrapper_sha256": (
        "src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py"
    ),
    "three_finger_sequential_grasp_sha256": (
        "src/kcg_connector/kcg_connector/grasp/"
        "three_finger_sequential_grasp.py"
    ),
    "finger_contact_detector_sha256": (
        "src/kcg_connector/kcg_connector/grasp/finger_contact_detector.py"
    ),
    "grasp_stability_monitor_sha256": (
        "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
    ),
    "physical_grasp_config_loader_sha256": (
        "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
    ),
    "physical_grasp_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    ),
    "pick_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    ),
    "tabletop_scene_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
    ),
}

GPU_CONSOLE_REQUIRED_MARKERS = (
    "NVIDIA GeForce RTX 5070 Ti",
    "Yes: 0",
    '"cuda:0"',
    "CUDA Toolkit",
)

GPU_CONSOLE_FORBIDDEN_MARKERS = (
    "Failed to create any GPU devices",
    "CPU fallback",
    "cpu_fallback",
    "Warp initialized on cpu",
    "warp CPU backend",
)

SIDE_METHODS = ("synchronous", "sequential-compliant")
EXECUTION_RECORD = "execution_record.json"


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token}")


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_source_hashes() -> dict[str, str]:
    return {
        key: _sha256_file(REPOSITORY_ROOT / relative)
        for key, relative in SOURCE_FILES.items()
    }


def _journal(path: Path, event: str, **fields: Any) -> None:
    record = {"ts_utc": _utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _status(event: str, *, seed: int, side: str | None = None) -> None:
    suffix = f" side={side}" if side is not None else ""
    print(f"[G4_PAIR] event={event} seed={seed}{suffix}", flush=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_argv(
    *,
    wrapper: Path,
    runner: Path,
    method: str,
    seed: int,
    formal_lift_mode: str,
    gui: bool,
    output_dir: Path,
) -> list[str]:
    argv = [
        str(wrapper),
        str(runner),
        "--physical-grasp-method",
        method,
        "--formal-lift-mode",
        formal_lift_mode,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if gui:
        argv.append("--gui")
    return argv


def normalized_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Fairness-normalized argv: drop method and output-dir tokens."""
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--physical-grasp-method", "--output-dir"):
            index += 2
            continue
        normalized.append(token)
        index += 1
    return tuple(normalized)


def _trace_problems(path: Path) -> list[str]:
    if not path.is_file():
        return ["missing controller trace"]
    problems: list[str] = []
    previous_step = None
    record_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(
                    line, parse_constant=_reject_nonfinite_json
                )
            except (json.JSONDecodeError, ValueError) as error:
                problems.append(f"trace line {line_number}: {error}")
                continue
            if not isinstance(record, Mapping):
                problems.append(f"trace line {line_number}: not an object")
                continue
            step = record.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int):
                problems.append(
                    f"trace line {line_number}: global_step not int"
                )
            elif previous_step is not None and step <= previous_step:
                problems.append(
                    f"trace line {line_number}: global_step not increasing"
                )
            else:
                previous_step = step
            record_count += 1
    if record_count == 0:
        problems.append("controller trace is empty")
    return problems


def verify_side_evidence(
    side_dir: Path,
    console_path: Path,
    *,
    expected_seed: int,
    expected_method: str,
    expected_gui: bool,
    source_hashes: Mapping[str, str],
) -> list[str]:
    """Structural side check: report/trace/console presence, GPU markers,
    seed/method/mode/gui and source hashes.  Physical failure with a
    complete structured report+trace is NOT a structural failure."""
    problems: list[str] = []
    report_path = side_dir / "nominal_physics_report.json"
    trace_path = side_dir / "controller_steps.jsonl"
    if not report_path.is_file():
        problems.append("missing report")
    problems.extend(_trace_problems(trace_path))
    if not console_path.is_file():
        problems.append("missing side console")
        return problems
    console = console_path.read_text(encoding="utf-8", errors="replace")
    for marker in GPU_CONSOLE_REQUIRED_MARKERS:
        if marker not in console:
            problems.append(f"console missing GPU marker {marker!r}")
    for marker in GPU_CONSOLE_FORBIDDEN_MARKERS:
        if marker in console:
            problems.append(
                f"console contains forbidden GPU marker {marker!r}"
            )
    if expected_method not in console:
        problems.append("console missing method marker")
    if report_path.is_file():
        try:
            report = _load_json(report_path)
        except (json.JSONDecodeError, ValueError) as error:
            problems.append(f"report invalid json: {error}")
            return problems
        if not isinstance(report, Mapping):
            problems.append("report is not an object")
            return problems
        if report.get("seed") != expected_seed:
            problems.append("report seed mismatch")
        if report.get("physical_grasp_method") != expected_method:
            problems.append("report method mismatch")
        if report.get("formal_lift_mode") != "staged":
            problems.append("report mode must be staged")
        if report.get("gui") is not expected_gui:
            problems.append("report gui mismatch")
        provenance = report.get("provenance") or {}
        for key, expected in source_hashes.items():
            if provenance.get(key) != expected:
                problems.append(f"report {key} source mismatch")
    return problems


def _fresh_record_for_directory(
    side_dir: Path, execution: Mapping[str, Any]
) -> dict[str, Any]:
    report_path = side_dir / "nominal_physics_report.json"
    trace_path = side_dir / "controller_steps.jsonl"
    console_path = side_dir / "side_console.log"
    return {
        "method": execution["method"],
        "kind": "fresh",
        "argv": list(execution["argv"]),
        "normalized_argv": list(execution["normalized_argv"]),
        "exit_code": execution["exit_code"],
        "started_at_utc": execution["started_at_utc"],
        "ended_at_utc": execution["ended_at_utc"],
        "duration_s": execution["duration_s"],
        "side_console": str(console_path),
        "side_console_sha256": _sha256_file(console_path),
        "report_file": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "trace_file": str(trace_path),
        "trace_sha256": _sha256_file(trace_path),
    }


def _load_fresh_record(side_dir: Path, method: str) -> dict[str, Any]:
    execution_path = side_dir / EXECUTION_RECORD
    execution = _load_json(execution_path)
    if not isinstance(execution, Mapping):
        raise ValueError("execution record is not an object")
    if execution.get("method") != method:
        raise ValueError("execution record method mismatch")
    normalized = execution.get("normalized_argv")
    if not (
        isinstance(normalized, list)
        and normalized
        and all(isinstance(item, str) for item in normalized)
    ):
        raise ValueError("execution record normalized argv invalid")
    return _fresh_record_for_directory(side_dir, execution)


def _manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {
            "seed": manifest.get("seed"),
            "gui": manifest.get("gui"),
            "sides": manifest.get("sides"),
        },
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_side(
    *,
    wrapper: Path,
    runner: Path,
    method: str,
    seed: int,
    formal_lift_mode: str,
    gui: bool,
    inprogress_side_dir: Path,
    journal_path: Path,
) -> dict[str, Any]:
    inprogress_side_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(
        wrapper=wrapper,
        runner=runner,
        method=method,
        seed=seed,
        formal_lift_mode=formal_lift_mode,
        gui=gui,
        output_dir=inprogress_side_dir,
    )
    _journal(journal_path, "side_start", seed=seed, side=method, argv=argv)
    _status("side_start", seed=seed, side=method)
    started = _utc_now()
    started_monotonic = time.monotonic()
    console_path = inprogress_side_dir / "side_console.log"
    try:
        with console_path.open("wb") as console_handle:
            result = subprocess.run(
                argv,
                stdout=console_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            console_handle.flush()
            os.fsync(console_handle.fileno())
        exit_code = result.returncode
    except Exception as error:  # noqa: BLE001 - orchestration boundary
        with console_path.open("ab") as console_handle:
            console_handle.write(
                (
                    "orchestrator error: "
                    f"{type(error).__name__}: {error}\n"
                ).encode("utf-8")
            )
            console_handle.flush()
            os.fsync(console_handle.fileno())
        exit_code = -1
    ended = _utc_now()
    duration_s = time.monotonic() - started_monotonic
    _status("side_process_end", seed=seed, side=method)
    report_path = inprogress_side_dir / "nominal_physics_report.json"
    trace_path = inprogress_side_dir / "controller_steps.jsonl"
    return {
        "method": method,
        "kind": "fresh",
        "argv": argv,
        "normalized_argv": list(normalized_argv(argv)),
        "exit_code": exit_code,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "duration_s": duration_s,
        "side_console": str(console_path),
        "side_console_sha256": _sha256_file(console_path),
        "report_file": str(report_path),
        "report_sha256": (
            _sha256_file(report_path) if report_path.is_file() else None
        ),
        "trace_file": str(trace_path),
        "trace_sha256": (
            _sha256_file(trace_path) if trace_path.is_file() else None
        ),
    }


def reuse_side_record(
    *,
    seed: int,
    method: str,
    entry: Mapping[str, Any],
    expected_gui: bool,
    source_hashes: Mapping[str, str],
    wrapper: Path,
    runner: Path,
    formal_lift_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    try:
        episode = Path(entry["episode_dir"]).resolve()
        console = Path(entry["console_log"]).resolve()
    except (KeyError, TypeError) as error:
        return {}, [f"reuse entry path invalid: {error}"]
    problems = verify_side_evidence(
        episode,
        console,
        expected_seed=seed,
        expected_method=method,
        expected_gui=expected_gui,
        source_hashes=source_hashes,
    )
    declared = entry.get("declared_normalized_argv")
    if not (
        isinstance(declared, list)
        and declared
        and all(isinstance(item, str) for item in declared)
    ):
        problems.append("reuse side lacks valid declared_normalized_argv")
    else:
        expected = normalized_argv(
            build_argv(
                wrapper=wrapper,
                runner=runner,
                method=method,
                seed=seed,
                formal_lift_mode=formal_lift_mode,
                gui=expected_gui,
                output_dir=episode,
            )
        )
        if tuple(declared) != expected:
            problems.append("reuse declared_normalized_argv mismatch")
    return (
        {
            "method": method,
            "kind": "reuse",
            "reuse_episode_dir": str(episode),
            "reuse_console_log": str(console),
            "declared_normalized_argv": declared,
            "argv": [],
            "normalized_argv": declared or [],
            "exit_code": None,
            "started_at_utc": None,
            "ended_at_utc": None,
            "duration_s": None,
            "side_console": str(console),
            "side_console_sha256": _sha256_file(console),
            "report_file": str(episode / "nominal_physics_report.json"),
            "report_sha256": _sha256_file(
                episode / "nominal_physics_report.json"
            ),
            "trace_file": str(episode / "controller_steps.jsonl"),
            "trace_sha256": _sha256_file(
                episode / "controller_steps.jsonl"
            ),
        },
        problems,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", action="append", required=True,
        help="seed range like 0-29 or single seed, repeatable",
    )
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument(
        "--isaac-wrapper", default=str(DEFAULT_WRAPPER),
        help="path to run_isaac_python.sh",
    )
    parser.add_argument(
        "--runner-py", default=str(DEFAULT_RUNNER),
        help="path to the tabletop physical grasp wrapper",
    )
    parser.add_argument(
        "--formal-lift-mode", default="staged",
        help="staged for formal pairs",
    )
    parser.add_argument("--gui", action="store_true", help="GUI mode pair")
    parser.add_argument(
        "--reuse-index", default=None, help="reuse-index JSON path"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)

    seeds: list[int] = []
    try:
        for token in arguments.seeds:
            if "-" in token:
                low_raw, high_raw = token.split("-", 1)
                low, high = int(low_raw), int(high_raw)
                if high < low:
                    parser.error("seed range upper bound is below lower bound")
                seeds.extend(range(low, high + 1))
            else:
                seeds.append(int(token))
    except ValueError as error:
        parser.error(f"invalid seed expression: {error}")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        parser.error("duplicate seeds in range")
    if arguments.gui and len(seeds) != 1:
        parser.error("--gui pairs require exactly one seed")
    if arguments.formal_lift_mode != "staged":
        parser.error("formal paired batches require --formal-lift-mode staged")

    base_dir = Path(arguments.base_output_dir).resolve()
    wrapper = Path(arguments.isaac_wrapper).resolve()
    runner = Path(arguments.runner_py).resolve()
    if not wrapper.is_file():
        parser.error(f"isaac wrapper missing: {wrapper}")
    if not runner.is_file():
        parser.error(f"runner missing: {runner}")
    try:
        reuse_index = (
            _load_json(Path(arguments.reuse_index).resolve())
            if arguments.reuse_index
            else {}
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"reuse-index is invalid: {error}")
    if not isinstance(reuse_index, Mapping):
        parser.error("reuse-index must be a JSON object")
    source_hashes = current_source_hashes()
    batch_runner_sha256 = _sha256_file(Path(__file__).resolve())
    base_dir.mkdir(parents=True, exist_ok=True)
    journal_path = base_dir / "batch_state.jsonl"
    _journal(
        journal_path,
        "batch_start",
        seeds=sorted(seeds),
        source_hashes=source_hashes,
        batch_runner_sha256=batch_runner_sha256,
    )

    structural_failures = 0
    for seed in sorted(seeds):
        pair_dir = base_dir / f"seed{seed:03d}"
        manifest_path = pair_dir / "pair_manifest.json"
        inprogress_dir = pair_dir / ".inprogress"
        _journal(journal_path, "pair_start", seed=seed)

        if manifest_path.is_file():
            if not arguments.resume:
                _journal(
                    journal_path, "pair_fail", seed=seed,
                    reason="finalized pair exists without --resume",
                )
                structural_failures += 1
                continue
            try:
                manifest = _load_json(manifest_path)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                _journal(
                    journal_path,
                    "pair_fail",
                    seed=seed,
                    reason=f"finalized manifest invalid: {error}",
                )
                structural_failures += 1
                continue
            problems: list[str] = []
            if manifest.get("schema_version") != SCHEMA_VERSION:
                problems.append("manifest schema mismatch")
            if manifest.get("source_hashes") != source_hashes:
                problems.append("manifest source hashes changed")
            if manifest.get("batch_runner_sha256") != batch_runner_sha256:
                problems.append("manifest batch runner hash changed")
            if manifest.get("seed") != seed:
                problems.append("manifest seed mismatch")
            manifest_sides = manifest.get("sides")
            if not isinstance(manifest_sides, list):
                manifest_sides = []
                problems.append("manifest sides not a list")
            methods = [
                side.get("method")
                for side in manifest_sides
                if isinstance(side, Mapping)
            ]
            if methods != list(SIDE_METHODS):
                problems.append(
                    "manifest must contain exactly two ordered sides"
                )
            if manifest.get("manifest_content_sha256") != (
                _manifest_content_sha256(manifest)
            ):
                problems.append("manifest content hash mismatch")
            normalized_by_method = {}
            for side in manifest_sides:
                if not isinstance(side, Mapping):
                    problems.append("manifest side is not an object")
                    continue
                method = side.get("method")
                if method not in SIDE_METHODS:
                    problems.append("manifest side method invalid")
                    continue
                normalized = side.get("normalized_argv")
                if not (
                    isinstance(normalized, list)
                    and normalized
                    and all(isinstance(item, str) for item in normalized)
                ):
                    problems.append(f"{method} normalized argv invalid")
                else:
                    normalized_by_method[method] = tuple(normalized)
                report_file = side.get("report_file")
                side_console = side.get("side_console")
                if not isinstance(report_file, str) or not isinstance(
                    side_console, str
                ):
                    problems.append(f"{method} evidence paths invalid")
                    continue
                problems.extend(
                    verify_side_evidence(
                        Path(report_file).parent,
                        Path(side_console),
                        expected_seed=seed,
                        expected_method=method,
                        expected_gui=arguments.gui,
                        source_hashes=source_hashes,
                    )
                )
                for hash_key, file_key in (
                    ("report_sha256", "report_file"),
                    ("trace_sha256", "trace_file"),
                    ("side_console_sha256", "side_console"),
                ):
                    raw_path = side.get(file_key)
                    if not isinstance(raw_path, str):
                        problems.append(f"{method} {file_key} path invalid")
                        continue
                    path = Path(raw_path)
                    if not path.is_file():
                        problems.append(f"{method} {file_key} file missing")
                    elif _sha256_file(path) != side.get(hash_key):
                        problems.append(f"{method} {hash_key} hash mismatch")
            if (
                set(normalized_by_method) == set(SIDE_METHODS)
                and normalized_by_method[SIDE_METHODS[0]]
                != normalized_by_method[SIDE_METHODS[1]]
            ):
                problems.append("manifest normalized argv fairness mismatch")
            if problems:
                _journal(
                    journal_path, "pair_fail", seed=seed,
                    reason="finalized pair re-verification failed",
                    problems=problems,
                )
                structural_failures += 1
                continue
            _journal(
                journal_path, "pair_skip", seed=seed, reason="resume verified"
            )
            continue

        if inprogress_dir.exists():
            if not arguments.resume:
                _journal(
                    journal_path, "pair_fail", seed=seed,
                    reason="inprogress pair exists without --resume",
                )
                structural_failures += 1
                continue
            state_path = inprogress_dir / "state.json"
            try:
                state = (
                    _load_json(state_path)
                    if state_path.is_file()
                    else {}
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                _journal(
                    journal_path,
                    "pair_fail",
                    seed=seed,
                    reason=f"inprogress state invalid: {error}",
                )
                structural_failures += 1
                continue
            if not isinstance(state, Mapping):
                _journal(
                    journal_path,
                    "pair_fail",
                    seed=seed,
                    reason="inprogress state is not an object",
                )
                structural_failures += 1
                continue
            if state.get("source_hashes") != source_hashes:
                _journal(
                    journal_path, "pair_fail", seed=seed,
                    reason="inprogress source hashes changed",
                )
                structural_failures += 1
                continue
        else:
            inprogress_dir.mkdir(parents=True)
            _atomic_json(
                inprogress_dir / "state.json",
                {
                    "source_hashes": source_hashes,
                    "batch_runner_sha256": batch_runner_sha256,
                    "seed": seed,
                    "gui": arguments.gui,
                },
            )

        pair_dir.mkdir(parents=True, exist_ok=True)
        _status("pair_start", seed=seed)
        sides: dict[str, Any] = {}
        pair_problems: list[str] = []
        for method in SIDE_METHODS:
            final_side_dir = pair_dir / (
                "sync" if method == "synchronous" else "sequential"
            )
            inprogress_side = inprogress_dir / final_side_dir.name
            reuse_seed_entry = reuse_index.get(str(seed)) or {}
            if not isinstance(reuse_seed_entry, Mapping):
                pair_problems.append("reuse seed entry is not an object")
                break
            reuse_entry = reuse_seed_entry.get(method)
            if reuse_entry is not None:
                if not isinstance(reuse_entry, Mapping):
                    pair_problems.append(
                        f"reuse {method}: entry is not an object"
                    )
                    break
                record, problems = reuse_side_record(
                    seed=seed,
                    method=method,
                    entry=reuse_entry,
                    expected_gui=arguments.gui,
                    source_hashes=source_hashes,
                    wrapper=wrapper,
                    runner=runner,
                    formal_lift_mode=arguments.formal_lift_mode,
                )
                if problems:
                    pair_problems.append(f"reuse {method}: {problems}")
                    break
                sides[method] = record
                _journal(journal_path, "side_reuse", seed=seed, side=method)
                _status("side_reuse", seed=seed, side=method)
                continue
            if (
                inprogress_side.is_dir()
                and not final_side_dir.is_dir()
                and arguments.resume
            ):
                promote_problems = verify_side_evidence(
                    inprogress_side,
                    inprogress_side / "side_console.log",
                    expected_seed=seed,
                    expected_method=method,
                    expected_gui=arguments.gui,
                    source_hashes=source_hashes,
                )
                if not promote_problems:
                    os.rename(inprogress_side, final_side_dir)
                    _journal(
                        journal_path, "side_promoted",
                        seed=seed, side=method,
                    )
            if final_side_dir.is_dir():
                if not arguments.resume:
                    pair_problems.append(
                        f"existing side {method} without --resume"
                    )
                    break
                problems = verify_side_evidence(
                    final_side_dir,
                    final_side_dir / "side_console.log",
                    expected_seed=seed,
                    expected_method=method,
                    expected_gui=arguments.gui,
                    source_hashes=source_hashes,
                )
                if problems:
                    pair_problems.append(f"side {method}: {problems}")
                    break
                try:
                    sides[method] = _load_fresh_record(
                        final_side_dir, method
                    )
                except (OSError, KeyError, TypeError, ValueError) as error:
                    pair_problems.append(
                        f"side {method} execution record invalid: {error}"
                    )
                    break
                _journal(journal_path, "side_skip", seed=seed, side=method)
                continue
            record = run_side(
                wrapper=wrapper,
                runner=runner,
                method=method,
                seed=seed,
                formal_lift_mode=arguments.formal_lift_mode,
                gui=arguments.gui,
                inprogress_side_dir=inprogress_side,
                journal_path=journal_path,
            )
            side_problems = verify_side_evidence(
                inprogress_side,
                inprogress_side / "side_console.log",
                expected_seed=seed,
                expected_method=method,
                expected_gui=arguments.gui,
                source_hashes=source_hashes,
            )
            if side_problems:
                pair_problems.append(f"side {method}: {side_problems}")
                continue
            _atomic_json(
                inprogress_side / EXECUTION_RECORD,
                {
                    key: record[key]
                    for key in (
                        "method",
                        "argv",
                        "normalized_argv",
                        "exit_code",
                        "started_at_utc",
                        "ended_at_utc",
                        "duration_s",
                    )
                },
            )
            os.rename(inprogress_side, final_side_dir)
            sides[method] = _load_fresh_record(final_side_dir, method)
            _journal(journal_path, "side_done", seed=seed, side=method)
            _status("side_done", seed=seed, side=method)

        if pair_problems:
            _journal(
                journal_path, "pair_fail", seed=seed, problems=pair_problems
            )
            structural_failures += 1
            continue
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "gui": arguments.gui,
            "batch_runner_sha256": batch_runner_sha256,
            "source_hashes": source_hashes,
            "generated_at_utc": _utc_now(),
            "sides": [sides[m] for m in SIDE_METHODS if m in sides],
        }
        if [side.get("method") for side in manifest["sides"]] != list(
            SIDE_METHODS
        ):
            _journal(
                journal_path,
                "pair_fail",
                seed=seed,
                reason="pair did not produce exactly two ordered sides",
            )
            structural_failures += 1
            continue
        normalized_values = [
            tuple(side.get("normalized_argv") or [])
            for side in manifest["sides"]
        ]
        if not all(normalized_values) or len(set(normalized_values)) != 1:
            _journal(
                journal_path,
                "pair_fail",
                seed=seed,
                reason="pair normalized argv fairness mismatch",
            )
            structural_failures += 1
            continue
        manifest["manifest_content_sha256"] = _manifest_content_sha256(
            manifest
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = pair_dir / "pair_manifest.json.tmp"
        _atomic_json(tmp_path, manifest)
        os.replace(tmp_path, manifest_path)
        _journal(journal_path, "pair_finalized", seed=seed)
        _status("pair_finalized", seed=seed)

    _journal(
        journal_path, "batch_end", structural_failures=structural_failures
    )
    return 0 if structural_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
