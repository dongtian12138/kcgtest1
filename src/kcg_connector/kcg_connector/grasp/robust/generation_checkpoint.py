"""Fail-closed persistence for resumable top-level candidate generation.

The store persists only immutable, generator-validated prefixes.  An intent
is committed before an evaluation segment starts.  If a process disappears
before committing the advanced prefix, recovery reports an explicit blocked
state; it never retries the ambiguous evaluator invocation or fabricates a
terminal candidate attempt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    CertifiedImplicitRoot,
    IntervalBounds,
    IntervalTransverseRootCertificate,
)
from kcg_connector.grasp.robust.ray_closure import (
    CertifiedContactFeatureRoot,
    CertifiedSequentialClosurePolicy,
    PadClosureAudit,
    PossibleFirstContactSet,
    RayClosureAudit,
)
from kcg_connector.grasp.robust.surface_anchored_closure import (
    SurfaceAnchorAudit,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    METHOD_ID as GENERATOR_METHOD_ID,
    AttemptStatus,
    CandidateAttemptAudit,
    CandidateLane,
    CandidateLineage,
    ResumableGenerationState,
    StaticV9AcceptedCandidate,
    StaticV9AcceptedPolicy,
    TopLevelCandidateGenerator,
    TopLevelGenerationResult,
    UniqueV9Evaluation,
    V9InvocationAuditBinding,
)


CHECKPOINT_METHOD_ID = (
    "CARTS_TOP_LEVEL_GENERATION_CANONICAL_CAS_CHECKPOINT_V1"
)
CHECKPOINT_SCHEMA_ID = (
    "CARTS_TOP_LEVEL_GENERATION_CHECKPOINT_MANIFEST_SCHEMA_V1"
)
CANONICAL_CODEC_METHOD_ID = (
    "CARTS_EXPLICIT_WHITELIST_BINARY64_HEX_CANONICAL_JSON_V1"
)
STATE_BLOB_HASH_DOMAIN = b"CARTS_GENERATION_STATE_BLOB_V1\0"
MANIFEST_HASH_DOMAIN = b"CARTS_GENERATION_CHECKPOINT_MANIFEST_V1\0"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GenerationCheckpointError(ValueError):
    """Raised when persisted generation evidence is malformed or stale."""


class GenerationCheckpointRecoveryBlocked(GenerationCheckpointError):
    """Raised for an intent whose evaluator outcome was never committed."""


class CheckpointLifecycle(str, Enum):
    """Only committed states that may appear in a checkpoint manifest."""

    READY = "READY"
    PREFIX_COMPLETE = "PREFIX_COMPLETE"
    IN_FLIGHT = "RECOVERY_BLOCKED_IN_FLIGHT"


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


@dataclass(frozen=True)
class GenerationCheckpointManifest:
    """Canonical hash-chain node pointing at one typed state blob."""

    schema_id: str
    checkpoint_method_id: str
    generator_method_id: str
    generator_contract_sha256: str
    v9_model_contract_sha256: str
    execution_environment_sha256: str
    codec_method_id: str
    codec_registry_sha256: str
    run_id: str
    sequence_number: int
    previous_checkpoint_sha256: str | None
    lifecycle: CheckpointLifecycle
    target_total_attempt_budget: int
    completed_attempt_count: int
    pending_stop_attempt_index_exclusive: int | None
    state_blob_sha256: str

    def __post_init__(self) -> None:
        digests = (
            self.generator_contract_sha256,
            self.v9_model_contract_sha256,
            self.execution_environment_sha256,
            self.codec_registry_sha256,
            self.state_blob_sha256,
        )
        if (
            self.schema_id != CHECKPOINT_SCHEMA_ID
            or self.checkpoint_method_id != CHECKPOINT_METHOD_ID
            or self.generator_method_id != GENERATOR_METHOD_ID
            or self.codec_method_id != CANONICAL_CODEC_METHOD_ID
            or not isinstance(self.run_id, str)
            or not self.run_id
            or isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
            or any(not _valid_sha256(value) for value in digests)
            or (
                self.previous_checkpoint_sha256 is not None
                and not _valid_sha256(self.previous_checkpoint_sha256)
            )
            or not isinstance(self.lifecycle, CheckpointLifecycle)
            or self.target_total_attempt_budget not in (128, 256, 512)
            or isinstance(self.completed_attempt_count, bool)
            or not isinstance(self.completed_attempt_count, int)
            or self.completed_attempt_count < 0
            or self.completed_attempt_count
            > self.target_total_attempt_budget
        ):
            raise GenerationCheckpointError(
                "checkpoint manifest identity/counters are malformed"
            )
        pending = self.pending_stop_attempt_index_exclusive
        if self.lifecycle is CheckpointLifecycle.IN_FLIGHT:
            if (
                isinstance(pending, bool)
                or not isinstance(pending, int)
                or pending <= self.completed_attempt_count
                or pending > self.target_total_attempt_budget
            ):
                raise GenerationCheckpointError(
                    "in-flight manifest needs one future stop boundary"
                )
        elif pending is not None:
            raise GenerationCheckpointError(
                "committed prefix cannot carry an in-flight stop boundary"
            )
        expected_complete = (
            self.completed_attempt_count == self.target_total_attempt_budget
        )
        if expected_complete != (
            self.lifecycle is CheckpointLifecycle.PREFIX_COMPLETE
        ):
            if self.lifecycle is not CheckpointLifecycle.IN_FLIGHT:
                raise GenerationCheckpointError(
                    "checkpoint lifecycle contradicts prefix completeness"
                )


@dataclass(frozen=True)
class StoredGenerationCheckpoint:
    """A verified manifest, its immutable typed state and content hash."""

    checkpoint_sha256: str
    manifest: GenerationCheckpointManifest
    state: ResumableGenerationState

    def __post_init__(self) -> None:
        if not _valid_sha256(self.checkpoint_sha256):
            raise GenerationCheckpointError(
                "stored checkpoint SHA-256 is malformed"
            )


def _type_id(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _default_allowed_types() -> tuple[type[Any], ...]:
    return (
        AttemptStatus,
        CandidateLane,
        CandidateLineage,
        CandidateAttemptAudit,
        UniqueV9Evaluation,
        V9InvocationAuditBinding,
        ResumableGenerationState,
        StaticV9AcceptedCandidate,
        TopLevelGenerationResult,
        GraspCandidate,
        PlannedPadContact,
        IntervalBounds,
        CertifiedImplicitRoot,
        IntervalTransverseRootCertificate,
        CertifiedContactFeatureRoot,
        PossibleFirstContactSet,
        CertifiedSequentialClosurePolicy,
        PadClosureAudit,
        RayClosureAudit,
        SurfaceAnchorAudit,
        StaticV9AcceptedPolicy,
        CheckpointLifecycle,
        GenerationCheckpointManifest,
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationCheckpointError(
                f"canonical JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


class CanonicalCheckpointCodec:
    """Exact JSON codec accepting only an explicit immutable type registry."""

    def __init__(
        self,
        *,
        additional_allowed_types: Sequence[type[Any]] = (),
    ) -> None:
        allowed = _default_allowed_types() + tuple(additional_allowed_types)
        registry: dict[str, type[Any]] = {}
        for value_type in allowed:
            if not isinstance(value_type, type) or not (
                issubclass(value_type, Enum) or is_dataclass(value_type)
            ):
                raise GenerationCheckpointError(
                    "checkpoint whitelist accepts only Enum/dataclass types"
                )
            identifier = _type_id(value_type)
            previous = registry.get(identifier)
            if previous is not None and previous is not value_type:
                raise GenerationCheckpointError(
                    "checkpoint type identifier is ambiguous"
                )
            registry[identifier] = value_type
        self._registry = dict(sorted(registry.items()))
        registry_document = {
            "codec_method_id": CANONICAL_CODEC_METHOD_ID,
            "allowed_type_ids": list(self._registry),
        }
        self.registry_sha256 = hashlib.sha256(
            json.dumps(
                registry_document,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _encode(self, value: object) -> object:
        value_type = type(value)
        identifier = _type_id(value_type)
        if isinstance(value, Enum):
            if self._registry.get(identifier) is not value_type:
                raise GenerationCheckpointError(
                    f"checkpoint enum type is not whitelisted: {identifier}"
                )
            return {
                "$enum": identifier,
                "value": self._encode(value.value),
            }
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if not math.isfinite(number):
                raise GenerationCheckpointError(
                    "checkpoint evidence cannot contain NaN or infinity"
                )
            return {"$binary64": number.hex()}
        if isinstance(value, bytes):
            return {"$bytes": value.hex()}
        if is_dataclass(value) and not isinstance(value, type):
            if self._registry.get(identifier) is not value_type:
                raise GenerationCheckpointError(
                    f"checkpoint dataclass is not whitelisted: {identifier}"
                )
            return {
                "$type": identifier,
                "fields": [
                    [row.name, self._encode(getattr(value, row.name))]
                    for row in fields(value)
                ],
            }
        if isinstance(value, tuple):
            return {"$tuple": [self._encode(row) for row in value]}
        if isinstance(value, list):
            return {"$list": [self._encode(row) for row in value]}
        if isinstance(value, Mapping):
            rows: list[list[object]] = []
            keys = tuple(value)
            if any(not isinstance(key, str) for key in keys):
                raise GenerationCheckpointError(
                    "checkpoint mappings require string keys"
                )
            for key in sorted(keys):
                rows.append([key, self._encode(value[key])])
            return {"$mapping": rows}
        raise GenerationCheckpointError(
            "checkpoint evidence type is not explicitly whitelisted: "
            f"{identifier}"
        )

    def _decode(self, value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            raise GenerationCheckpointError(
                "canonical checkpoint JSON cannot contain decimal floats"
            )
        if isinstance(value, list):
            raise GenerationCheckpointError(
                "canonical checkpoint containers require an explicit tag"
            )
        if not isinstance(value, dict):
            raise GenerationCheckpointError(
                "canonical checkpoint JSON value is malformed"
            )
        keys = set(value)
        if keys == {"$binary64"}:
            token = value["$binary64"]
            if not isinstance(token, str):
                raise GenerationCheckpointError(
                    "binary64 checkpoint token must be text"
                )
            try:
                number = float.fromhex(token)
            except ValueError as error:
                raise GenerationCheckpointError(
                    "binary64 checkpoint token is invalid"
                ) from error
            if not math.isfinite(number) or number.hex() != token:
                raise GenerationCheckpointError(
                    "binary64 checkpoint token is noncanonical"
                )
            return number
        if keys == {"$bytes"}:
            token = value["$bytes"]
            if (
                not isinstance(token, str)
                or len(token) % 2
                or any(
                    character not in "0123456789abcdef"
                    for character in token
                )
            ):
                raise GenerationCheckpointError(
                    "checkpoint byte token is not lowercase hexadecimal"
                )
            return bytes.fromhex(token)
        if keys == {"$tuple"} or keys == {"$list"}:
            tag = "$tuple" if "$tuple" in value else "$list"
            rows = value[tag]
            if not isinstance(rows, list):
                raise GenerationCheckpointError(
                    "checkpoint sequence payload must be a list"
                )
            decoded = [self._decode(row) for row in rows]
            return tuple(decoded) if tag == "$tuple" else decoded
        if keys == {"$mapping"}:
            rows = value["$mapping"]
            if not isinstance(rows, list):
                raise GenerationCheckpointError(
                    "checkpoint mapping payload must be a list"
                )
            result: dict[str, object] = {}
            previous: str | None = None
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != 2
                    or not isinstance(row[0], str)
                    or (previous is not None and row[0] <= previous)
                ):
                    raise GenerationCheckpointError(
                        "checkpoint mapping keys are not unique/sorted"
                    )
                previous = row[0]
                result[row[0]] = self._decode(row[1])
            return result
        if keys == {"$enum", "value"}:
            identifier = value["$enum"]
            value_type = (
                self._registry.get(identifier)
                if isinstance(identifier, str)
                else None
            )
            if (
                not isinstance(identifier, str)
                or value_type is None
                or not issubclass(value_type, Enum)
            ):
                raise GenerationCheckpointError(
                    "checkpoint enum type is not in the explicit registry"
                )
            try:
                return value_type(self._decode(value["value"]))
            except (TypeError, ValueError) as error:
                raise GenerationCheckpointError(
                    "checkpoint enum value is invalid"
                ) from error
        if keys == {"$type", "fields"}:
            identifier = value["$type"]
            value_type = (
                self._registry.get(identifier)
                if isinstance(identifier, str)
                else None
            )
            rows = value["fields"]
            if (
                not isinstance(identifier, str)
                or value_type is None
                or not is_dataclass(value_type)
                or not isinstance(rows, list)
            ):
                raise GenerationCheckpointError(
                    "checkpoint dataclass type/fields are not registered"
                )
            expected_fields = tuple(row.name for row in fields(value_type))
            observed_fields: list[str] = []
            kwargs: dict[str, object] = {}
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != 2
                    or not isinstance(row[0], str)
                ):
                    raise GenerationCheckpointError(
                        "checkpoint dataclass field row is malformed"
                    )
                observed_fields.append(row[0])
                kwargs[row[0]] = self._decode(row[1])
            if tuple(observed_fields) != expected_fields:
                raise GenerationCheckpointError(
                    "checkpoint dataclass field schema/order changed"
                )
            try:
                return value_type(**kwargs)
            except Exception as error:
                raise GenerationCheckpointError(
                    f"checkpoint dataclass reconstruction failed: {identifier}"
                ) from error
        raise GenerationCheckpointError(
            "canonical checkpoint JSON object has an unknown tag"
        )

    def canonical_bytes(self, value: object) -> bytes:
        return json.dumps(
            self._encode(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode_canonical_bytes(self, value: bytes) -> object:
        try:
            document = json.loads(
                value.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise GenerationCheckpointError(
                "checkpoint bytes are not valid canonical JSON"
            ) from error
        decoded = self._decode(document)
        if self.canonical_bytes(decoded) != value:
            raise GenerationCheckpointError(
                "checkpoint JSON does not round-trip canonically"
            )
        return decoded


def _domain_sha256(domain: bytes, value: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(value)
    return digest.hexdigest()


class GenerationCheckpointStore:
    """Single-writer CAS store with immutable blobs and atomic LATEST."""

    def __init__(
        self,
        root: Path | str,
        *,
        codec: CanonicalCheckpointCodec | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.blob_directory = self.root / "blobs"
        self.manifest_directory = self.root / "manifests"
        self.latest_path = self.root / "LATEST"
        self.lock_path = self.root / "LOCK"
        self.codec = CanonicalCheckpointCodec() if codec is None else codec
        self.fault_injector = fault_injector
        self._live_intent_capabilities: set[str] = set()
        self.blob_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_directory.mkdir(parents=True, exist_ok=True)

    def _fault(self, stage: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage)

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_immutable(
        self,
        *,
        directory: Path,
        digest: str,
        value: bytes,
    ) -> Path:
        target = directory / f"{digest}.json"
        if target.exists():
            if target.read_bytes() != value:
                raise GenerationCheckpointError(
                    "content-addressed path contains contradictory bytes"
                )
            return target
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.",
            suffix=".tmp",
            dir=directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != value:
                    raise GenerationCheckpointError(
                        "concurrent CAS writer produced contradictory bytes"
                    )
            self._fsync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _read_latest_hash(self) -> str | None:
        if not self.latest_path.exists():
            return None
        try:
            value = self.latest_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise GenerationCheckpointError(
                "LATEST cannot be read as ASCII"
            ) from error
        if not value.endswith("\n") or not _valid_sha256(value[:-1]):
            raise GenerationCheckpointError("LATEST pointer is malformed")
        return value[:-1]

    def _replace_latest(self, digest: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".LATEST.",
            suffix=".tmp",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((digest + "\n").encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            self._fault("before_latest_replace")
            os.replace(temporary, self.latest_path)
            self._fsync_directory(self.root)
            self._fault("after_latest_replace")
        finally:
            temporary.unlink(missing_ok=True)

    def _write_state_blob(self, state: ResumableGenerationState) -> str:
        value = self.codec.canonical_bytes(state)
        digest = _domain_sha256(STATE_BLOB_HASH_DOMAIN, value)
        self._write_immutable(
            directory=self.blob_directory,
            digest=digest,
            value=value,
        )
        self._fault("after_state_blob_write")
        return digest

    def _write_manifest(
        self,
        manifest: GenerationCheckpointManifest,
    ) -> str:
        value = self.codec.canonical_bytes(manifest)
        digest = _domain_sha256(MANIFEST_HASH_DOMAIN, value)
        self._write_immutable(
            directory=self.manifest_directory,
            digest=digest,
            value=value,
        )
        self._fault("after_manifest_write")
        return digest

    def _read_cas(
        self,
        *,
        directory: Path,
        digest: str,
        domain: bytes,
    ) -> bytes:
        if not _valid_sha256(digest):
            raise GenerationCheckpointError("CAS digest is malformed")
        path = directory / f"{digest}.json"
        try:
            value = path.read_bytes()
        except OSError as error:
            raise GenerationCheckpointError(
                f"CAS object is missing: {digest}"
            ) from error
        if _domain_sha256(domain, value) != digest:
            raise GenerationCheckpointError(
                f"CAS object bytes contradict digest: {digest}"
            )
        return value

    def _read_manifest(
        self,
        checkpoint_sha256: str,
    ) -> GenerationCheckpointManifest:
        value = self._read_cas(
            directory=self.manifest_directory,
            digest=checkpoint_sha256,
            domain=MANIFEST_HASH_DOMAIN,
        )
        manifest = self.codec.decode_canonical_bytes(value)
        if type(manifest) is not GenerationCheckpointManifest:
            raise GenerationCheckpointError(
                "checkpoint CAS object is not a manifest"
            )
        return manifest

    def _read_state(
        self,
        manifest: GenerationCheckpointManifest,
    ) -> ResumableGenerationState:
        value = self._read_cas(
            directory=self.blob_directory,
            digest=manifest.state_blob_sha256,
            domain=STATE_BLOB_HASH_DOMAIN,
        )
        state = self.codec.decode_canonical_bytes(value)
        if type(state) is not ResumableGenerationState:
            raise GenerationCheckpointError(
                "checkpoint state blob has an unexpected type"
            )
        return state

    def _validate_chain(
        self,
        latest_sha256: str,
        *,
        generator: TopLevelCandidateGenerator,
    ) -> None:
        seen: set[str] = set()
        expected_sequence: int | None = None
        current: str | None = latest_sha256
        identity: tuple[str, ...] | None = None
        reverse_nodes: list[
            tuple[
                str,
                GenerationCheckpointManifest,
                ResumableGenerationState,
            ]
        ] = []
        while current is not None:
            if current in seen:
                raise GenerationCheckpointError("checkpoint hash chain cycles")
            seen.add(current)
            manifest = self._read_manifest(current)
            state = self._read_state(manifest)
            generator.validate_resumable_state(state)
            if (
                manifest.generator_contract_sha256
                != state.contract_hash_sha256
                or manifest.target_total_attempt_budget
                != state.target_total_attempt_budget
                or manifest.completed_attempt_count
                != state.completed_attempt_count
                or manifest.codec_registry_sha256
                != self.codec.registry_sha256
            ):
                raise GenerationCheckpointError(
                    "checkpoint chain node contradicts its typed state"
                )
            row_identity = (
                manifest.generator_contract_sha256,
                manifest.v9_model_contract_sha256,
                manifest.execution_environment_sha256,
                manifest.codec_registry_sha256,
                manifest.run_id,
            )
            if identity is None:
                identity = row_identity
                expected_sequence = manifest.sequence_number
            if row_identity != identity or manifest.sequence_number != (
                expected_sequence
            ):
                raise GenerationCheckpointError(
                    "checkpoint hash chain identity/sequence changed"
                )
            reverse_nodes.append((current, manifest, state))
            expected_sequence -= 1
            current = manifest.previous_checkpoint_sha256
        if expected_sequence != -1:
            raise GenerationCheckpointError(
                "checkpoint hash chain does not end at sequence zero"
            )
        nodes = tuple(reversed(reverse_nodes))
        if (
            not nodes
            or nodes[0][1].previous_checkpoint_sha256 is not None
            or nodes[0][1].sequence_number != 0
            or nodes[0][1].lifecycle is not CheckpointLifecycle.READY
            or nodes[0][2].completed_attempt_count != 0
        ):
            raise GenerationCheckpointError(
                "checkpoint chain does not start from an empty READY state"
            )
        for previous, observed in zip(nodes, nodes[1:]):
            previous_sha, previous_manifest, previous_state = previous
            _, observed_manifest, observed_state = observed
            if observed_manifest.previous_checkpoint_sha256 != previous_sha:
                raise GenerationCheckpointError(
                    "checkpoint previous-hash link changed"
                )
            if observed_manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT:
                if (
                    previous_manifest.lifecycle
                    is CheckpointLifecycle.IN_FLIGHT
                    or observed_manifest.state_blob_sha256
                    != previous_manifest.state_blob_sha256
                    or observed_state != previous_state
                ):
                    raise GenerationCheckpointError(
                        "in-flight intent changed committed prefix evidence"
                    )
                continue
            if previous_manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT:
                stop = (
                    previous_manifest.
                    pending_stop_attempt_index_exclusive
                )
                if (
                    observed_state.target_total_attempt_budget
                    != previous_state.target_total_attempt_budget
                    or observed_state.completed_attempt_count != stop
                    or observed_state.attempts[
                        : previous_state.completed_attempt_count
                    ]
                    != previous_state.attempts
                ):
                    raise GenerationCheckpointError(
                        "committed state does not resolve its exact intent"
                    )
                observed_unique = {
                    row.v9_parameter_key_hex: row
                    for row in observed_state.unique_v9_evaluations
                }
                for row in previous_state.unique_v9_evaluations:
                    child = observed_unique.get(row.v9_parameter_key_hex)
                    if (
                        child is None
                        or len(child.lineage) < len(row.lineage)
                        or child.lineage[: len(row.lineage)] != row.lineage
                        or self.codec.canonical_bytes(
                            replace(child, lineage=row.lineage)
                        )
                        != self.codec.canonical_bytes(row)
                    ):
                        raise GenerationCheckpointError(
                            "committed segment rewrote an earlier V9 outcome"
                        )
                continue
            if (
                observed_state.target_total_attempt_budget
                <= previous_state.target_total_attempt_budget
                or observed_state.completed_attempt_count
                != previous_state.completed_attempt_count
                or observed_state.attempts != previous_state.attempts
                or observed_state.unique_v9_evaluations
                != previous_state.unique_v9_evaluations
            ):
                raise GenerationCheckpointError(
                    "checkpoint transition is neither intent nor extension"
                )

    def _validate_loaded(
        self,
        checkpoint_sha256: str,
        *,
        generator: TopLevelCandidateGenerator,
        run_id: str,
        execution_environment_sha256: str,
    ) -> StoredGenerationCheckpoint:
        self._validate_chain(
            checkpoint_sha256,
            generator=generator,
        )
        manifest = self._read_manifest(checkpoint_sha256)
        state = self._read_state(manifest)
        generator.validate_resumable_state(state)
        if (
            manifest.generator_contract_sha256
            != generator.contract_hash_sha256
            or manifest.v9_model_contract_sha256
            != generator.v9_model_contract_sha256
            or manifest.execution_environment_sha256
            != execution_environment_sha256
            or manifest.codec_registry_sha256
            != self.codec.registry_sha256
            or manifest.run_id != run_id
            or manifest.target_total_attempt_budget
            != state.target_total_attempt_budget
            or manifest.completed_attempt_count
            != state.completed_attempt_count
        ):
            raise GenerationCheckpointError(
                "checkpoint manifest differs from generator/state binding"
            )
        return StoredGenerationCheckpoint(
            checkpoint_sha256=checkpoint_sha256,
            manifest=manifest,
            state=state,
        )

    def _authoritative_base(
        self,
        base: StoredGenerationCheckpoint,
        *,
        generator: TopLevelCandidateGenerator,
    ) -> StoredGenerationCheckpoint:
        if type(base) is not StoredGenerationCheckpoint:
            raise GenerationCheckpointError(
                "checkpoint commit base has an unexpected type"
            )
        verified = self._validate_loaded(
            base.checkpoint_sha256,
            generator=generator,
            run_id=base.manifest.run_id,
            execution_environment_sha256=(
                base.manifest.execution_environment_sha256
            ),
        )
        if (
            self.codec.canonical_bytes(base.manifest)
            != self.codec.canonical_bytes(verified.manifest)
            or self.codec.canonical_bytes(base.state)
            != self.codec.canonical_bytes(verified.state)
        ):
            raise GenerationCheckpointError(
                "checkpoint commit base differs from authoritative CAS bytes"
            )
        return verified

    def _commit(
        self,
        *,
        expected_latest_sha256: str | None,
        manifest: GenerationCheckpointManifest,
        state: ResumableGenerationState,
    ) -> StoredGenerationCheckpoint:
        with self._lock(exclusive=True):
            current = self._read_latest_hash()
            if current != expected_latest_sha256:
                raise GenerationCheckpointError(
                    "LATEST compare-and-swap detected a concurrent writer"
                )
            state_sha256 = self._write_state_blob(state)
            if state_sha256 != manifest.state_blob_sha256:
                raise GenerationCheckpointError(
                    "manifest state blob digest was not independently derived"
                )
            checkpoint_sha256 = self._write_manifest(manifest)
            self._replace_latest(checkpoint_sha256)
        return StoredGenerationCheckpoint(
            checkpoint_sha256=checkpoint_sha256,
            manifest=manifest,
            state=state,
        )

    def initialize(
        self,
        *,
        generator: TopLevelCandidateGenerator,
        state: ResumableGenerationState,
        run_id: str,
        execution_environment_sha256: str,
    ) -> StoredGenerationCheckpoint:
        generator.validate_resumable_state(state)
        if state.completed_attempt_count != 0:
            raise GenerationCheckpointError(
                "initial checkpoint must contain an empty prefix"
            )
        if not _valid_sha256(execution_environment_sha256):
            raise GenerationCheckpointError(
                "execution environment digest is malformed"
            )
        state_sha256 = _domain_sha256(
            STATE_BLOB_HASH_DOMAIN,
            self.codec.canonical_bytes(state),
        )
        manifest = GenerationCheckpointManifest(
            schema_id=CHECKPOINT_SCHEMA_ID,
            checkpoint_method_id=CHECKPOINT_METHOD_ID,
            generator_method_id=GENERATOR_METHOD_ID,
            generator_contract_sha256=generator.contract_hash_sha256,
            v9_model_contract_sha256=generator.v9_model_contract_sha256,
            execution_environment_sha256=execution_environment_sha256,
            codec_method_id=CANONICAL_CODEC_METHOD_ID,
            codec_registry_sha256=self.codec.registry_sha256,
            run_id=run_id,
            sequence_number=0,
            previous_checkpoint_sha256=None,
            lifecycle=CheckpointLifecycle.READY,
            target_total_attempt_budget=state.target_total_attempt_budget,
            completed_attempt_count=0,
            pending_stop_attempt_index_exclusive=None,
            state_blob_sha256=state_sha256,
        )
        return self._commit(
            expected_latest_sha256=None,
            manifest=manifest,
            state=state,
        )

    def commit_intent(
        self,
        base: StoredGenerationCheckpoint,
        *,
        generator: TopLevelCandidateGenerator,
        stop_attempt_index_exclusive: int,
    ) -> StoredGenerationCheckpoint:
        base = self._authoritative_base(base, generator=generator)
        if base.manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT:
            raise GenerationCheckpointError(
                "cannot start another segment from an in-flight intent"
            )
        stop = stop_attempt_index_exclusive
        if (
            isinstance(stop, bool)
            or not isinstance(stop, int)
            or stop <= base.state.completed_attempt_count
            or stop > base.state.target_total_attempt_budget
        ):
            raise GenerationCheckpointError(
                "intent stop must be inside the unprocessed target prefix"
            )
        manifest = GenerationCheckpointManifest(
            **{
                **{
                    row.name: getattr(base.manifest, row.name)
                    for row in fields(GenerationCheckpointManifest)
                },
                "sequence_number": base.manifest.sequence_number + 1,
                "previous_checkpoint_sha256": base.checkpoint_sha256,
                "lifecycle": CheckpointLifecycle.IN_FLIGHT,
                "pending_stop_attempt_index_exclusive": stop,
            }
        )
        committed = self._commit(
            expected_latest_sha256=base.checkpoint_sha256,
            manifest=manifest,
            state=base.state,
        )
        self._live_intent_capabilities.add(
            committed.checkpoint_sha256
        )
        return committed

    def commit_advanced(
        self,
        intent: StoredGenerationCheckpoint,
        *,
        generator: TopLevelCandidateGenerator,
        advanced_state: ResumableGenerationState,
    ) -> StoredGenerationCheckpoint:
        intent = self._authoritative_base(intent, generator=generator)
        if intent.manifest.lifecycle is not CheckpointLifecycle.IN_FLIGHT:
            raise GenerationCheckpointError(
                "advanced state must follow an in-flight intent"
            )
        if (
            intent.checkpoint_sha256
            not in self._live_intent_capabilities
        ):
            raise GenerationCheckpointRecoveryBlocked(
                "RECOVERY_BLOCKED_IN_FLIGHT: this process did not commit "
                "the evaluator intent; retry/fabricated completion is "
                "forbidden"
            )
        generator.validate_resumable_state(advanced_state)
        expected_stop = intent.manifest.pending_stop_attempt_index_exclusive
        if (
            advanced_state.contract_hash_sha256
            != intent.state.contract_hash_sha256
            or advanced_state.target_total_attempt_budget
            != intent.state.target_total_attempt_budget
            or advanced_state.attempts[: intent.state.completed_attempt_count]
            != intent.state.attempts
            or advanced_state.completed_attempt_count != expected_stop
        ):
            raise GenerationCheckpointError(
                "advanced state is not the exact requested prefix extension"
            )
        lifecycle = (
            CheckpointLifecycle.PREFIX_COMPLETE
            if advanced_state.completed_attempt_count
            == advanced_state.target_total_attempt_budget
            else CheckpointLifecycle.READY
        )
        state_sha256 = _domain_sha256(
            STATE_BLOB_HASH_DOMAIN,
            self.codec.canonical_bytes(advanced_state),
        )
        manifest = GenerationCheckpointManifest(
            **{
                **{
                    row.name: getattr(intent.manifest, row.name)
                    for row in fields(GenerationCheckpointManifest)
                },
                "sequence_number": intent.manifest.sequence_number + 1,
                "previous_checkpoint_sha256": intent.checkpoint_sha256,
                "lifecycle": lifecycle,
                "completed_attempt_count": (
                    advanced_state.completed_attempt_count
                ),
                "pending_stop_attempt_index_exclusive": None,
                "state_blob_sha256": state_sha256,
            }
        )
        committed = self._commit(
            expected_latest_sha256=intent.checkpoint_sha256,
            manifest=manifest,
            state=advanced_state,
        )
        self._live_intent_capabilities.discard(
            intent.checkpoint_sha256
        )
        return committed

    def commit_extension(
        self,
        base: StoredGenerationCheckpoint,
        *,
        generator: TopLevelCandidateGenerator,
        extended_state: ResumableGenerationState,
    ) -> StoredGenerationCheckpoint:
        base = self._authoritative_base(base, generator=generator)
        if base.manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT:
            raise GenerationCheckpointError(
                "cannot extend target from an in-flight intent"
            )
        generator.validate_resumable_state(extended_state)
        if (
            extended_state.target_total_attempt_budget
            <= base.state.target_total_attempt_budget
            or extended_state.completed_attempt_count
            != base.state.completed_attempt_count
            or self.codec.canonical_bytes(extended_state.attempts)
            != self.codec.canonical_bytes(base.state.attempts)
            or self.codec.canonical_bytes(
                extended_state.unique_v9_evaluations
            )
            != self.codec.canonical_bytes(
                base.state.unique_v9_evaluations
            )
        ):
            raise GenerationCheckpointError(
                "target extension changed committed generation evidence"
            )
        state_sha256 = _domain_sha256(
            STATE_BLOB_HASH_DOMAIN,
            self.codec.canonical_bytes(extended_state),
        )
        manifest = GenerationCheckpointManifest(
            **{
                **{
                    row.name: getattr(base.manifest, row.name)
                    for row in fields(GenerationCheckpointManifest)
                },
                "sequence_number": base.manifest.sequence_number + 1,
                "previous_checkpoint_sha256": base.checkpoint_sha256,
                "lifecycle": CheckpointLifecycle.READY,
                "target_total_attempt_budget": (
                    extended_state.target_total_attempt_budget
                ),
                "pending_stop_attempt_index_exclusive": None,
                "state_blob_sha256": state_sha256,
            }
        )
        return self._commit(
            expected_latest_sha256=base.checkpoint_sha256,
            manifest=manifest,
            state=extended_state,
        )

    def inspect_latest(
        self,
        *,
        generator: TopLevelCandidateGenerator,
        run_id: str,
        execution_environment_sha256: str,
    ) -> StoredGenerationCheckpoint:
        if not _valid_sha256(execution_environment_sha256):
            raise GenerationCheckpointError(
                "execution environment digest is malformed"
            )
        with self._lock(exclusive=False):
            latest = self._read_latest_hash()
            if latest is None:
                raise GenerationCheckpointError(
                    "checkpoint store has no LATEST state"
                )
            return self._validate_loaded(
                latest,
                generator=generator,
                run_id=run_id,
                execution_environment_sha256=execution_environment_sha256,
            )

    def load_latest(
        self,
        *,
        generator: TopLevelCandidateGenerator,
        run_id: str,
        execution_environment_sha256: str,
    ) -> StoredGenerationCheckpoint:
        stored = self.inspect_latest(
            generator=generator,
            run_id=run_id,
            execution_environment_sha256=execution_environment_sha256,
        )
        if stored.manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT:
            raise GenerationCheckpointRecoveryBlocked(
                "RECOVERY_BLOCKED_IN_FLIGHT: evaluator outcome is ambiguous; "
                "retry/fabricated failure is forbidden"
            )
        return stored


__all__ = [
    "CANONICAL_CODEC_METHOD_ID",
    "CHECKPOINT_METHOD_ID",
    "CHECKPOINT_SCHEMA_ID",
    "CanonicalCheckpointCodec",
    "CheckpointLifecycle",
    "GenerationCheckpointError",
    "GenerationCheckpointManifest",
    "GenerationCheckpointRecoveryBlocked",
    "GenerationCheckpointStore",
    "StoredGenerationCheckpoint",
]
