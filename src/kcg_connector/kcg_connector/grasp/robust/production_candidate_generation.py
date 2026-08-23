"""Hash-bound, resumable production entry for CARTS candidate generation.

This module assembles only the audited object, hand, PAD and method contracts.
It persists an intent before evaluating a segment, so an interrupted evaluator
call remains visibly ambiguous and cannot be retried as if nothing happened.
The produced status is static generation evidence only; it never authorizes
formal selection, Isaac, or hardware execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

import mpmath
import numpy as np
import scipy

from .generation_checkpoint import (
    CheckpointLifecycle,
    GenerationCheckpointStore,
    StoredGenerationCheckpoint,
)
from .hand_contract import CARTSHandContract, load_carts_hand_contract
from .object_contract import LoadedObjectContract, load_object_contract
from .ray_closure import PreRegisteredTaskFrame, RayClosureSurfaceModel
from .study_contract import StudyContractAudit, audit_study_contract
from .surface_anchored_closure import SurfaceAnchoredRayClosureModel
from .top_level_candidate_generator import (
    MAIN_TOTAL_ATTEMPT_BUDGET,
    AttemptStatus,
    TopLevelCandidateGenerator,
)


METHOD_CONTRACT = Path("src/kcg_connector/config/carts_grasp_v1.yaml")
HAND_CONTRACT = Path("src/kcg_connector/config/carts_hand_contact_v1.yaml")
OBJECT_CONTRACT = Path("src/kcg_connector/config/carts_grasp_objects_v1.yaml")
DEVELOPMENT_OBJECT_ID = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT_ID = "te_deutsch_d38999_26fj35pn_step"
ALLOWED_OBJECT_IDS = (DEVELOPMENT_OBJECT_ID, TRANSFER_OBJECT_ID)
STATUS_SCHEMA = "carts_production_candidate_generation_status_v1"
STATIC_CLAIM = "STATIC_REAL_CONTRACT_CANDIDATE_GENERATION_PREFIX_ONLY"

_ENVIRONMENT_SOURCE_PATHS = (
    Path("src/kcg_connector/kcg_connector/grasp/robust/production_candidate_generation.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/generation_checkpoint.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/top_level_candidate_generator.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/surface_anchored_closure.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/ray_closure.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/interval_kinematics.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/hand_contract.py"),
    Path("src/kcg_connector/kcg_connector/grasp/robust/object_contract.py"),
)


class ProductionCandidateGenerationError(ValueError):
    """Raised when the production runtime cannot retain exact lineage."""


@dataclass(frozen=True)
class ProductionCandidateGenerationRuntime:
    """One fully bound object-specific runtime using the shared algorithm."""

    repository_root: Path
    object_id: str
    study_audit: StudyContractAudit
    hand_contract: CARTSHandContract
    object_contract: LoadedObjectContract
    closure_model: RayClosureSurfaceModel
    surface_proposer: SurfaceAnchoredRayClosureModel
    generator: TopLevelCandidateGenerator
    execution_environment_sha256: str
    run_id: str

    def default_checkpoint_root(self) -> Path:
        return (
            self.repository_root
            / "artifacts/carts_grasp/CARTS_GRASP_V1/production_candidate_generation"
            / self.object_id
            / self.run_id
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionCandidateGenerationError(f"{label} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_environment_sha256(
    *,
    repository_root: Path,
    study_audit: StudyContractAudit,
    generator: TopLevelCandidateGenerator,
) -> str:
    source_hashes: dict[str, str] = {}
    for relative in _ENVIRONMENT_SOURCE_PATHS:
        absolute = (repository_root / relative).resolve()
        if not absolute.is_file():
            raise ProductionCandidateGenerationError(
                f"execution source is unavailable: {relative.as_posix()}"
            )
        source_hashes[relative.as_posix()] = _sha256_file(absolute)
    document = {
        "schema_version": "carts_generation_execution_environment_v1",
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "mpmath_version": mpmath.__version__,
        "study_audit_sha256": study_audit.canonical_sha256,
        "generator_contract_sha256": generator.contract_hash_sha256,
        "v9_model_contract_sha256": generator.v9_model_contract_sha256,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_production_candidate_generation_runtime(
    *,
    repository_root: Path | str,
    object_id: str,
) -> ProductionCandidateGenerationRuntime:
    """Build one real-contract runtime without evaluating any candidate."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ProductionCandidateGenerationError(
            "repository_root must be an existing directory"
        )
    if object_id not in ALLOWED_OBJECT_IDS:
        raise ProductionCandidateGenerationError(
            "object_id is outside the preregistered development/transfer pair"
        )
    study_audit = audit_study_contract(
        METHOD_CONTRACT,
        HAND_CONTRACT,
        OBJECT_CONTRACT,
        repository_root=root,
    )
    method = _mapping(
        _mapping(
            study_audit.canonical_manifest["source_documents"],
            "study source documents",
        )["shared_method"],
        "shared method",
    )
    ray = _mapping(method["ray_closure"], "shared method ray_closure")
    interval = _mapping(ray["interval_backend"], "ray_closure interval_backend")

    hand_contract = load_carts_hand_contract(
        HAND_CONTRACT,
        repository_root=root,
    )
    object_contract = load_object_contract(
        OBJECT_CONTRACT,
        object_id=object_id,
        repository_root=root,
    )
    hand_model = hand_contract.build_hand_model()
    directions = hand_contract.closing_actuation_directions_unit(hand_model)
    task_rotation = np.asarray(
        object_contract.task_frame_rotation_object,
        dtype=np.float64,
    )
    if task_rotation.shape != (3, 3):
        raise ProductionCandidateGenerationError(
            "object task frame rotation must be 3 by 3"
        )
    closure_model = RayClosureSurfaceModel(
        object_model=object_contract.model,
        hand_model=hand_model,
        verified_pads=hand_contract.pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=tuple(float(row) for row in task_rotation[:, 0]),
            source=f"{object_contract.task_frame_source}:TASK_X_AXIS",
        ),
        closing_actuation_directions_unit=directions,
        object_contact_normal_policy=hand_contract.object_contact_normal_policy,
        pad_surface_normal_policy=hand_contract.pad_surface_normal_policy,
        maximum_subdivision_intervals=int(ray["maximum_subdivision_intervals"]),
        interval_decimal_precision=int(interval["decimal_precision"]),
        maximum_root_bisection_iterations=int(
            interval["maximum_root_bisection_iterations"]
        ),
    )
    surface_proposer = SurfaceAnchoredRayClosureModel(closure_model)
    pad_names = tuple(row.verified.name for row in closure_model.prepared_pads)
    generator = TopLevelCandidateGenerator(
        v9_evaluator=closure_model,
        surface_proposer=surface_proposer,
        hand_model=hand_model,
        anchor_pad_names=pad_names,
    )
    environment_sha256 = _execution_environment_sha256(
        repository_root=root,
        study_audit=study_audit,
        generator=generator,
    )
    run_id = (
        f"CARTS-{object_id}-BUDGET{MAIN_TOTAL_ATTEMPT_BUDGET}-"
        f"{generator.contract_hash_sha256[:16]}-{environment_sha256[:16]}"
    )
    return ProductionCandidateGenerationRuntime(
        repository_root=root,
        object_id=object_id,
        study_audit=study_audit,
        hand_contract=hand_contract,
        object_contract=object_contract,
        closure_model=closure_model,
        surface_proposer=surface_proposer,
        generator=generator,
        execution_environment_sha256=environment_sha256,
        run_id=run_id,
    )


def initialize_or_load_checkpoint(
    runtime: ProductionCandidateGenerationRuntime,
    *,
    checkpoint_root: Path | str | None = None,
) -> tuple[GenerationCheckpointStore, StoredGenerationCheckpoint]:
    """Create an empty formal prefix or load one exact committed prefix."""

    root = (
        runtime.default_checkpoint_root()
        if checkpoint_root is None
        else Path(checkpoint_root).resolve()
    )
    store = GenerationCheckpointStore(root)
    if store.latest_path.exists():
        stored = store.load_latest(
            generator=runtime.generator,
            run_id=runtime.run_id,
            execution_environment_sha256=runtime.execution_environment_sha256,
        )
    else:
        stored = store.initialize(
            generator=runtime.generator,
            state=runtime.generator.begin_resumable(MAIN_TOTAL_ATTEMPT_BUDGET),
            run_id=runtime.run_id,
            execution_environment_sha256=runtime.execution_environment_sha256,
        )
    write_generation_status(runtime, stored, checkpoint_root=store.root)
    return store, stored


def advance_checkpoint(
    runtime: ProductionCandidateGenerationRuntime,
    store: GenerationCheckpointStore,
    stored: StoredGenerationCheckpoint,
    *,
    stop_attempt_index_exclusive: int,
) -> StoredGenerationCheckpoint:
    """Advance one explicit prefix segment with an intent-first commit."""

    start = stored.state.completed_attempt_count
    stop = stop_attempt_index_exclusive
    if isinstance(stop, bool) or not isinstance(stop, int) or stop < start:
        raise ProductionCandidateGenerationError(
            "stop_attempt_index_exclusive cannot precede the committed prefix"
        )
    if stop == start:
        return stored
    intent = store.commit_intent(
        stored,
        generator=runtime.generator,
        stop_attempt_index_exclusive=stop,
    )
    advanced_state = runtime.generator.advance_resumable(
        intent.state,
        stop_attempt_index_exclusive=stop,
    )
    committed = store.commit_advanced(
        intent,
        generator=runtime.generator,
        advanced_state=advanced_state,
    )
    write_generation_status(runtime, committed, checkpoint_root=store.root)
    return committed


def advance_checkpoint_incrementally(
    runtime: ProductionCandidateGenerationRuntime,
    store: GenerationCheckpointStore,
    stored: StoredGenerationCheckpoint,
    *,
    stop_attempt_index_exclusive: int,
) -> StoredGenerationCheckpoint:
    """Reuse one loaded runtime while committing every attempt separately.

    This removes repeated CAD/hand/model construction from a long fixed-budget
    run without weakening the intent-first interruption boundary: an external
    stop can make only the currently executing attempt ambiguous, while every
    earlier attempt remains an independently committed prefix.
    """

    stop = stop_attempt_index_exclusive
    start = stored.state.completed_attempt_count
    if isinstance(stop, bool) or not isinstance(stop, int) or stop < start:
        raise ProductionCandidateGenerationError(
            "stop_attempt_index_exclusive cannot precede the committed prefix"
        )
    while stored.state.completed_attempt_count < stop:
        stored = advance_checkpoint(
            runtime,
            store,
            stored,
            stop_attempt_index_exclusive=(
                stored.state.completed_attempt_count + 1
            ),
        )
    return stored


def generation_status_document(
    runtime: ProductionCandidateGenerationRuntime,
    stored: StoredGenerationCheckpoint,
    *,
    checkpoint_root: Path,
) -> dict[str, object]:
    """Return a compact status that cannot be confused with formal pass."""

    counts = Counter(row.status.value for row in stored.state.attempts)
    accepted_candidate_keys = tuple(
        row.v9_parameter_key_hex
        for row in stored.state.unique_v9_evaluations
        if row.status is AttemptStatus.STATIC_V9_ACCEPTED
    )
    accepted_policy_keys = tuple(
        row.v9_parameter_key_hex
        for row in stored.state.unique_v9_evaluations
        if row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
    )
    complete = (
        stored.manifest.lifecycle is CheckpointLifecycle.PREFIX_COMPLETE
    )
    try:
        checkpoint_path = checkpoint_root.relative_to(runtime.repository_root).as_posix()
    except ValueError:
        checkpoint_path = str(checkpoint_root)
    return {
        "schema_version": STATUS_SCHEMA,
        "claim": STATIC_CLAIM,
        "object_id": runtime.object_id,
        "run_id": runtime.run_id,
        "checkpoint_root": checkpoint_path,
        "checkpoint_sha256": stored.checkpoint_sha256,
        "checkpoint_lifecycle": stored.manifest.lifecycle.value,
        "target_total_attempt_budget": stored.state.target_total_attempt_budget,
        "completed_attempt_count": stored.state.completed_attempt_count,
        "prefix_complete": complete,
        "attempt_status_counts": dict(sorted(counts.items())),
        "unique_v9_evaluation_count": len(stored.state.unique_v9_evaluations),
        "static_exact_candidate_count": len(accepted_candidate_keys),
        "static_contact_range_policy_count": len(accepted_policy_keys),
        "static_exact_candidate_keys": list(accepted_candidate_keys),
        "static_contact_range_policy_keys": list(accepted_policy_keys),
        "generator_contract_sha256": runtime.generator.contract_hash_sha256,
        "v9_model_contract_sha256": runtime.generator.v9_model_contract_sha256,
        "study_audit_sha256": runtime.study_audit.canonical_sha256,
        "execution_environment_sha256": runtime.execution_environment_sha256,
        "preregistration_blockers": list(
            runtime.study_audit.preregistration_blockers
        ),
        "production_joint_route_count": 0,
        "formal_selected_contact_range_policy": None,
        "formal_selected_candidate": None,
        "full_hand_collision_state": "NOT_CERTIFIABLE",
        "dynamic_launch_allowed": False,
        "hardware_authorized": False,
        "legacy_candidate_imported": False,
        "display_only_proposal_used_as_formal_evidence": False,
        "online_object_or_contact_truth_used": False,
    }


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_generation_status(
    runtime: ProductionCandidateGenerationRuntime,
    stored: StoredGenerationCheckpoint,
    *,
    checkpoint_root: Path,
) -> Path:
    status_path = checkpoint_root / "STATUS.json"
    _atomic_json(
        status_path,
        generation_status_document(
            runtime,
            stored,
            checkpoint_root=checkpoint_root,
        ),
    )
    return status_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one fail-closed CARTS production generation segment."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--object-id", choices=ALLOWED_OBJECT_IDS, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--stop-attempt-index-exclusive", type=int)
    parser.add_argument(
        "--run-through-attempt-index-exclusive",
        type=int,
        help=(
            "reuse one loaded runtime and commit one intent/result per attempt"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    selected_mode_count = sum(
        (
            int(arguments.initialize_only),
            int(arguments.stop_attempt_index_exclusive is not None),
            int(arguments.run_through_attempt_index_exclusive is not None),
        )
    )
    if selected_mode_count != 1:
        raise ProductionCandidateGenerationError(
            "choose exactly one of --initialize-only, "
            "--stop-attempt-index-exclusive, or "
            "--run-through-attempt-index-exclusive"
        )
    runtime = build_production_candidate_generation_runtime(
        repository_root=arguments.repository_root,
        object_id=arguments.object_id,
    )
    store, stored = initialize_or_load_checkpoint(
        runtime,
        checkpoint_root=arguments.checkpoint_root,
    )
    if arguments.stop_attempt_index_exclusive is not None:
        stored = advance_checkpoint(
            runtime,
            store,
            stored,
            stop_attempt_index_exclusive=(
                arguments.stop_attempt_index_exclusive
            ),
        )
    elif arguments.run_through_attempt_index_exclusive is not None:
        stored = advance_checkpoint_incrementally(
            runtime,
            store,
            stored,
            stop_attempt_index_exclusive=(
                arguments.run_through_attempt_index_exclusive
            ),
        )
    document = generation_status_document(
        runtime,
        stored,
        checkpoint_root=store.root,
    )
    print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_OBJECT_IDS",
    "DEVELOPMENT_OBJECT_ID",
    "ProductionCandidateGenerationError",
    "ProductionCandidateGenerationRuntime",
    "TRANSFER_OBJECT_ID",
    "advance_checkpoint",
    "advance_checkpoint_incrementally",
    "build_production_candidate_generation_runtime",
    "generation_status_document",
    "initialize_or_load_checkpoint",
    "main",
    "write_generation_status",
]
