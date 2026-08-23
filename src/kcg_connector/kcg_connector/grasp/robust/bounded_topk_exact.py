"""Hash-bound, wall-bounded exact evaluation of the frozen proxy Top-4.

The parent builds the real model once.  Each exact V9 invocation runs in its
own forked child so a wall-time overrun can be terminated without mutating a
production checkpoint.  A timeout is computationally unresolved, never a
geometric rejection.  The first exact static acceptance stops the sequence;
collision, wrench, route, dynamics, and formal selection remain downstream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .generation_checkpoint import CanonicalCheckpointCodec
from .production_candidate_generation import (
    DEVELOPMENT_OBJECT_ID,
    build_production_candidate_generation_runtime,
)
from .top_level_candidate_generator import (
    AttemptStatus,
    CanonicalV9Parameters,
    V9InvocationAuditBinding,
    canonicalize_v9_parameters,
)
from .grasp_optimizer import GraspCandidate
from .ray_closure import (
    CertifiedSequentialClosurePolicy,
    RayClosureAudit,
)


METHOD_ID = "CARTS_HASH_BOUND_SEQUENTIAL_TOP4_FORKED_EXACT_V1"
SCHEMA_VERSION = "carts_bounded_top4_exact_run_v1"
PROFILE_SCHEMA = "carts_multifidelity_proxy_rank_profile_v8"
PROFILE_STATUS = "COMPLETED_TOP4_PROXY_BOUND"
PER_CANDIDATE_WALL_LIMIT_SECONDS = 60.0
MAXIMUM_EXACT_CANDIDATE_COUNT = 4
ACCEPTED_STATUSES = frozenset(
    (
        AttemptStatus.STATIC_V9_ACCEPTED,
        AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
    )
)


class BoundedTopKExactError(ValueError):
    """Raised when bounded exact execution would violate its evidence wall."""


@dataclass(frozen=True)
class SelectedProxyCandidate:
    proxy_rank: int
    first_attempt_index: int
    canonical_key_hex: str
    parameters_unit: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ExactV9CompletedRecord:
    """Canonical exact result written only by a naturally completed child."""

    method_id: str
    profile_sha256: str
    run_id: str
    generator_contract_sha256: str
    v9_model_contract_sha256: str
    proxy_rank: int
    first_attempt_index: int
    canonical_parameters: CanonicalV9Parameters
    wall_seconds: float
    status: AttemptStatus
    candidate: GraspCandidate | None
    sequential_closure_policy: CertifiedSequentialClosurePolicy | None
    audit: RayClosureAudit | None
    invocation_binding: V9InvocationAuditBinding | None
    v9_failure_reason: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BoundedTopKExactError(
                f"profile JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def load_selected_proxy_candidates(
    profile_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], tuple[SelectedProxyCandidate, ...]]:
    """Validate the H50 evidence wall and return its immutable Top-4 order."""

    if _sha256_path(profile_path) != expected_sha256:
        raise BoundedTopKExactError("proxy profile SHA-256 does not match")
    try:
        document = json.loads(
            profile_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoundedTopKExactError("proxy profile is not valid JSON") from error
    rank_result = document.get("proxy_rank_result")
    if not isinstance(rank_result, dict):
        raise BoundedTopKExactError("proxy profile lacks a rank result")
    if (
        document.get("schema_version") != PROFILE_SCHEMA
        or document.get("status") != PROFILE_STATUS
        or document.get("object_id") != DEVELOPMENT_OBJECT_ID
        or document.get("exact_v9_evaluator_called") is not False
        or document.get("generation_checkpoint_written") is not False
        or document.get("isaac_launched") is not False
        or document.get("exact_top_k_ceiling")
        != MAXIMUM_EXACT_CANDIDATE_COUNT
        or rank_result.get("proxy_certifies_or_rejects") is not False
        or rank_result.get("formal_selected_candidate") is not None
        or rank_result.get("formal_selected_contact_range_policy") is not None
        or rank_result.get("full_hand_collision_state") != "NOT_CERTIFIABLE"
        or rank_result.get("dynamic_launch_allowed") is not False
        or rank_result.get("hardware_authorized") is not False
    ):
        raise BoundedTopKExactError(
            "proxy profile claim or execution boundary is not H50"
        )
    raw_rows = rank_result.get("ranked_survivors")
    raw_selected_keys = rank_result.get("exact_selected_keys")
    if not isinstance(raw_rows, list) or not isinstance(
        raw_selected_keys, list
    ):
        raise BoundedTopKExactError("proxy profile selected rows are malformed")
    selected_rows = [
        row
        for row in raw_rows
        if isinstance(row, dict) and row.get("selected_for_exact_v9") is True
    ]
    if (
        len(selected_rows) == 0
        or len(selected_rows) > MAXIMUM_EXACT_CANDIDATE_COUNT
        or len(selected_rows) != document.get("exact_selected_unique_count")
        or [row.get("canonical_key_hex") for row in selected_rows]
        != raw_selected_keys
        or [row.get("proxy_rank") for row in selected_rows]
        != list(range(1, len(selected_rows) + 1))
    ):
        raise BoundedTopKExactError(
            "proxy profile does not contain one contiguous frozen Top-4"
        )
    result: list[SelectedProxyCandidate] = []
    for row in selected_rows:
        parameters_value = row.get("parameters_unit")
        if not isinstance(parameters_value, list) or len(parameters_value) != 5:
            raise BoundedTopKExactError(
                "selected proxy parameters are malformed"
            )
        try:
            parameters_array = np.asarray(
                parameters_value, dtype=np.float64
            )
            key_bytes = bytes.fromhex(str(row.get("canonical_key_hex")))
        except (TypeError, ValueError) as error:
            raise BoundedTopKExactError(
                "selected proxy parameters are not canonical V9 values"
            ) from error
        if (
            parameters_array.shape != (5,)
            or not np.all(np.isfinite(parameters_array))
            or np.any(parameters_array < 0.0)
            or np.any(parameters_array > 1.0)
            or parameters_array[0] >= 1.0
            or key_bytes
            != np.asarray(parameters_array, dtype=">f8").tobytes(order="C")
        ):
            raise BoundedTopKExactError(
                "selected proxy parameters are not canonical V9 values"
            )
        parameters = tuple(float(value) for value in parameters_array)
        canonical = CanonicalV9Parameters(
            values=(
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
                parameters[4],
            ),
            exact_key=key_bytes,
        )
        key = row.get("canonical_key_hex")
        first_attempt_index = row.get("first_attempt_index")
        proxy_rank = row.get("proxy_rank")
        if (
            key != canonical.exact_key_hex
            or isinstance(first_attempt_index, bool)
            or not isinstance(first_attempt_index, int)
            or first_attempt_index < 0
            or isinstance(proxy_rank, bool)
            or not isinstance(proxy_rank, int)
        ):
            raise BoundedTopKExactError(
                "selected proxy identity differs from canonical parameters"
            )
        result.append(
            SelectedProxyCandidate(
                proxy_rank=proxy_rank,
                first_attempt_index=first_attempt_index,
                canonical_key_hex=canonical.exact_key_hex,
                parameters_unit=canonical.values,
            )
        )
    return document, tuple(result)


def _completed_codec() -> CanonicalCheckpointCodec:
    return CanonicalCheckpointCodec(
        additional_allowed_types=(
            CanonicalV9Parameters,
            ExactV9CompletedRecord,
        )
    )


def _exact_worker(
    runtime: Any,
    selected: SelectedProxyCandidate,
    profile_sha256: str,
    result_path: Path,
) -> None:
    started = time.perf_counter()
    canonical = canonicalize_v9_parameters(
        selected.parameters_unit,
        parameter_layout=runtime.generator.v9_parameter_layout,
    )
    (
        candidate,
        sequential_closure_policy,
        audit,
        invocation_binding,
        status,
        failure_reason,
    ) = runtime.generator._evaluate_v9(canonical)
    record = ExactV9CompletedRecord(
        method_id=METHOD_ID,
        profile_sha256=profile_sha256,
        run_id=runtime.run_id,
        generator_contract_sha256=runtime.generator.contract_hash_sha256,
        v9_model_contract_sha256=runtime.generator.v9_model_contract_sha256,
        proxy_rank=selected.proxy_rank,
        first_attempt_index=selected.first_attempt_index,
        canonical_parameters=canonical,
        wall_seconds=time.perf_counter() - started,
        status=status,
        candidate=candidate,
        sequential_closure_policy=sequential_closure_policy,
        audit=audit,
        invocation_binding=invocation_binding,
        v9_failure_reason=failure_reason,
    )
    _atomic_bytes(result_path, _completed_codec().canonical_bytes(record))


def _completed_summary(
    record: ExactV9CompletedRecord,
    *,
    record_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    value = record_path.read_bytes()
    try:
        relative_path = record_path.relative_to(repository_root).as_posix()
    except ValueError:
        relative_path = str(record_path)
    return {
        "proxy_rank": record.proxy_rank,
        "first_attempt_index": record.first_attempt_index,
        "canonical_key_hex": record.canonical_parameters.exact_key_hex,
        "execution_status": "EXACT_COMPLETED",
        "wall_seconds": record.wall_seconds,
        "exact_attempt_status": record.status.value,
        "v9_failure_reason": record.v9_failure_reason,
        "candidate_present": record.candidate is not None,
        "sequential_closure_policy_present": (
            record.sequential_closure_policy is not None
        ),
        "accepted_static": record.status in ACCEPTED_STATUSES,
        "completed_record_path": relative_path,
        "completed_record_sha256": _sha256_bytes(value),
        "timeout_is_geometric_rejection": False,
    }


def _timeout_summary(
    selected: SelectedProxyCandidate,
    *,
    wall_seconds: float,
) -> dict[str, object]:
    return {
        "proxy_rank": selected.proxy_rank,
        "first_attempt_index": selected.first_attempt_index,
        "canonical_key_hex": selected.canonical_key_hex,
        "execution_status": "COMPUTATION_UNRESOLVED_WALL_TIMEOUT",
        "wall_seconds": wall_seconds,
        "exact_attempt_status": None,
        "v9_failure_reason": None,
        "candidate_present": False,
        "sequential_closure_policy_present": False,
        "accepted_static": False,
        "completed_record_path": None,
        "completed_record_sha256": None,
        "timeout_is_geometric_rejection": False,
    }


def run(
    *,
    repository_root: Path,
    profile_path: Path,
    expected_profile_sha256: str,
    output_path: Path,
    result_directory: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile, selected_rows = load_selected_proxy_candidates(
        profile_path, expected_sha256=expected_profile_sha256
    )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "recorded_at_utc": _utc_now(),
        "status": "BUILDING_MODEL",
        "object_id": DEVELOPMENT_OBJECT_ID,
        "input_profile_path": str(profile_path),
        "input_profile_sha256": expected_profile_sha256,
        "per_candidate_wall_limit_seconds": (
            PER_CANDIDATE_WALL_LIMIT_SECONDS
        ),
        "maximum_exact_candidate_count": MAXIMUM_EXACT_CANDIDATE_COUNT,
        "selected_proxy_ranks": [row.proxy_rank for row in selected_rows],
        "selected_first_attempt_indices": [
            row.first_attempt_index for row in selected_rows
        ],
        "candidate_results": [],
        "generation_checkpoint_written": False,
        "isaac_launched": False,
        "formal_selected_contact_range_policy": None,
        "formal_selected_candidate": None,
        "full_hand_collision_state": "NOT_CERTIFIABLE",
        "dynamic_launch_allowed": False,
        "hardware_authorized": False,
    }
    _atomic_json(output_path, document)
    runtime = build_production_candidate_generation_runtime(
        repository_root=repository_root,
        object_id=DEVELOPMENT_OBJECT_ID,
    )
    if runtime.run_id != profile.get("run_id_if_generation_were_authorized"):
        raise BoundedTopKExactError(
            "current runtime differs from the H50 profile contract"
        )
    document.update(
        {
            "status": "MODEL_BUILT",
            "run_id": runtime.run_id,
            "generator_contract_sha256": (
                runtime.generator.contract_hash_sha256
            ),
            "v9_model_contract_sha256": (
                runtime.generator.v9_model_contract_sha256
            ),
            "model_build_and_input_validation_seconds": (
                time.perf_counter() - started
            ),
        }
    )
    _atomic_json(output_path, document)
    context = mp.get_context("fork")
    accepted_summary: dict[str, object] | None = None
    result_directory.mkdir(parents=True, exist_ok=True)
    for selected in selected_rows:
        record_path = result_directory / (
            f"rank_{selected.proxy_rank:02d}_attempt_"
            f"{selected.first_attempt_index:03d}_"
            f"{selected.canonical_key_hex[:16]}.canonical.json"
        )
        if record_path.exists():
            raise BoundedTopKExactError(
                "bounded exact result path already exists; overwrite refused"
            )
        attempt_started = time.perf_counter()
        document.update(
            {
                "status": "EXACT_CANDIDATE_RUNNING",
                "active_proxy_rank": selected.proxy_rank,
                "active_first_attempt_index": selected.first_attempt_index,
                "active_started_at_utc": _utc_now(),
            }
        )
        _atomic_json(output_path, document)
        print(
            json.dumps(
                {
                    "status": "EXACT_CANDIDATE_RUNNING",
                    "proxy_rank": selected.proxy_rank,
                    "first_attempt_index": selected.first_attempt_index,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        process = context.Process(
            target=_exact_worker,
            args=(
                runtime,
                selected,
                expected_profile_sha256,
                record_path,
            ),
            daemon=False,
        )
        process.start()
        deadline = attempt_started + PER_CANDIDATE_WALL_LIMIT_SECONDS
        while process.is_alive():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            process.join(timeout=min(1.0, remaining))
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover - OS failure boundary
                raise BoundedTopKExactError(
                    "timed-out exact child could not be reclaimed"
                )
            summary = _timeout_summary(
                selected,
                wall_seconds=time.perf_counter() - attempt_started,
            )
        else:
            if process.exitcode != 0 or not record_path.is_file():
                summary = {
                    **_timeout_summary(
                        selected,
                        wall_seconds=time.perf_counter() - attempt_started,
                    ),
                    "execution_status": "COMPUTATION_UNRESOLVED_CHILD_ERROR",
                    "child_exit_code": process.exitcode,
                }
            else:
                decoded = _completed_codec().decode_canonical_bytes(
                    record_path.read_bytes()
                )
                if type(decoded) is not ExactV9CompletedRecord:
                    raise BoundedTopKExactError(
                        "completed exact record has an unexpected type"
                    )
                if (
                    decoded.method_id != METHOD_ID
                    or decoded.profile_sha256 != expected_profile_sha256
                    or decoded.run_id != runtime.run_id
                    or decoded.proxy_rank != selected.proxy_rank
                    or decoded.first_attempt_index
                    != selected.first_attempt_index
                    or decoded.canonical_parameters.exact_key_hex
                    != selected.canonical_key_hex
                ):
                    raise BoundedTopKExactError(
                        "completed exact record differs from its request"
                    )
                summary = _completed_summary(
                    decoded,
                    record_path=record_path,
                    repository_root=repository_root,
                )
        document["candidate_results"].append(summary)
        document.update(
            {
                "status": "EXACT_CANDIDATE_COMMITTED",
                "active_proxy_rank": None,
                "active_first_attempt_index": None,
                "active_started_at_utc": None,
                "total_wall_seconds_so_far": time.perf_counter() - started,
            }
        )
        _atomic_json(output_path, document)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if summary["accepted_static"] is True:
            accepted_summary = summary
            break
    document.update(
        {
            "status": (
                "EXACT_STATIC_ACCEPTANCE_FOUND"
                if accepted_summary is not None
                else "NO_EXACT_ACCEPTANCE_WITHIN_TOP4_AND_WALL_BOUNDS"
            ),
            "active_proxy_rank": None,
            "active_first_attempt_index": None,
            "active_started_at_utc": None,
            "exact_invocation_count": len(document["candidate_results"]),
            "accepted_proxy_rank": (
                None
                if accepted_summary is None
                else accepted_summary["proxy_rank"]
            ),
            "accepted_first_attempt_index": (
                None
                if accepted_summary is None
                else accepted_summary["first_attempt_index"]
            ),
            "stopped_after_first_exact_acceptance": (
                accepted_summary is not None
            ),
            "total_wall_seconds": time.perf_counter() - started,
            "formal_selected_contact_range_policy": None,
            "formal_selected_candidate": None,
            "full_hand_collision_state": "NOT_CERTIFIABLE",
            "dynamic_launch_allowed": False,
            "hardware_authorized": False,
            "generation_checkpoint_written": False,
            "isaac_launched": False,
        }
    )
    _atomic_json(output_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        repository_root=arguments.repository_root.resolve(),
        profile_path=arguments.profile.resolve(),
        expected_profile_sha256=arguments.expected_profile_sha256,
        output_path=arguments.output.resolve(),
        result_directory=arguments.result_directory.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_invocation_count": result["exact_invocation_count"],
                "accepted_first_attempt_index": result[
                    "accepted_first_attempt_index"
                ],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedTopKExactError",
    "ExactV9CompletedRecord",
    "MAXIMUM_EXACT_CANDIDATE_COUNT",
    "METHOD_ID",
    "PER_CANDIDATE_WALL_LIMIT_SECONDS",
    "SelectedProxyCandidate",
    "load_selected_proxy_candidates",
    "run",
]
