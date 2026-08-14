"""Reproducible SAC training and evaluation for connector residual v0.

This module owns algorithm configuration, run guards, artifact provenance and
evaluation reporting.  It deliberately does not create an Isaac Sim scene:
the caller must inject the one canonical ``ConnectorResidualEnv`` backed by
``ConnectorResidualIsaacBackend`` after starting ``SimulationApp``.

PyTorch and Stable-Baselines3 are imported only inside physical run functions,
so configuration and metadata logic remain testable in the ROS Python runtime.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import random
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_connector_residual_sac_run_v0"
INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
ACTION_SIZE = 4
OBSERVATION_SIZE = 24
MINIMUM_ACTOR_PARAMETER_DELTA = 1.0e-7
MINIMUM_PAIRED_EPISODES_FOR_CLAIM = 100
POSITIVE_CLAIM_CONFIDENCE_LEVEL = 0.95
MAXIMUM_PAIRED_P_VALUE = 0.05
MINIMUM_POSITIVE_CLAIM_IMPROVEMENT = 0.10
MINIMUM_POSITIVE_CLAIM_SUCCESS_RATE = 0.95
EVALUATION_RUNTIME_COMPATIBILITY_FIELDS = (
    "python",
    "isaacsim",
    "numpy",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "torch_cuda_build",
    "gpu",
)
SAFETY_FAILURE_REASONS = frozenset(
    {
        "cross_thread",
        "finger_overload",
        "invalid_physics",
        "lost_grasp",
        "nut_overspeed",
        "overtwist",
        "q7_limit",
        "q7_overspeed",
        "q7_tracking",
        "reverse_progress",
    }
)
RAW_SAFETY_METRIC_FIELDS = frozenset(
    {
        "physics_substep_max_abs_joint_velocity_rad_s",
        "physics_substep_max_abs_q7_velocity_rad_s",
        "physics_substep_max_joint_limit_violation_rad",
        "physics_substep_max_abs_finger_base_torque_nm",
        "policy_boundary_max_abs_nut_angular_velocity_rad_s",
        "policy_boundary_max_abs_q7_tracking_error_rad",
        "policy_boundary_max_grasp_translation_error_m",
        "policy_boundary_max_grasp_rotation_error_rad",
    }
)
RAW_SAFETY_LIMIT_FIELDS = frozenset(
    RAW_SAFETY_METRIC_FIELDS
    - {"physics_substep_max_abs_joint_velocity_rad_s"}
)
RAW_SAFETY_REPORT_FIELDS = frozenset(
    {
        "failure_reasons",
        "finite_throughout",
        "limits",
        "metrics",
        "passed",
        "sampling",
        "signal_source",
    }
)
TRAINING_RAW_SAFETY_SCHEMA_VERSION = (
    "kcg_training_raw_safety_audit_v1"
)
TRAINING_RAW_SAFETY_EPISODE_FIELDS = frozenset(
    {
        "complete",
        "evidence_valid_throughout",
        "failure_reasons",
        "finite_throughout",
        "last_sampling",
        "limits",
        "passed",
        "peaks",
        "policy_steps",
        "signal_source",
    }
)
TRAINING_RAW_SAFETY_REPORT_FIELDS = frozenset(
    {
        "complete_episode_count",
        "episode_reports",
        "evidence_valid_throughout",
        "failure_reasons",
        "finite_throughout",
        "limits",
        "partial_episode_count",
        "passed",
        "peaks",
        "policy_steps_audited",
        "schema_version",
        "signal_source",
    }
)


@dataclass(frozen=True)
class ConnectorResidualSACConfig:
    """Validated, immutable configuration for formal SAC runs."""

    schema_version: str
    interface_version: str
    algorithm: str
    policy: str
    device: str
    required_torch_version: str
    required_cuda_build: str
    action_size: int
    observation_size: int
    seed: int
    maximum_unconfirmed_timesteps: int
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq_steps: int
    gradient_steps: int
    network_architecture: tuple[int, ...]
    checkpoint_interval_steps: int
    save_replay_buffer: bool
    use_vecnormalize: bool
    evaluation_episodes: int
    evaluation_seed_start: int
    evaluation_deterministic: bool
    minimum_success_rate: float
    maximum_safety_failures: int


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_connector_residual_sac_config(
    config_path: str | Path,
) -> ConnectorResidualSACConfig:
    """Load and fail-closed validate the formal SAC configuration."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "configuration")
    contract = _mapping(document.get("contract"), "contract")
    runtime = _mapping(document.get("runtime"), "runtime")
    algorithm = _mapping(document.get("algorithm"), "algorithm")
    artifacts = _mapping(document.get("artifacts"), "artifacts")
    evaluation = _mapping(document.get("evaluation"), "evaluation")

    architecture = tuple(
        _positive_int(value, "algorithm.network_architecture")
        for value in algorithm.get("network_architecture", ())
    )
    config = ConnectorResidualSACConfig(
        schema_version=str(document.get("schema_version", "")),
        interface_version=str(contract.get("interface_version", "")),
        algorithm=str(algorithm.get("name", "")),
        policy=str(algorithm.get("policy", "")),
        device=str(runtime.get("device", "")),
        required_torch_version=str(
            runtime.get("required_torch_version", "")
        ),
        required_cuda_build=str(runtime.get("required_cuda_build", "")),
        action_size=_positive_int(
            contract.get("action_size"), "contract.action_size"
        ),
        observation_size=_positive_int(
            contract.get("observation_size"), "contract.observation_size"
        ),
        seed=_nonnegative_int(runtime.get("seed"), "runtime.seed"),
        maximum_unconfirmed_timesteps=_positive_int(
            runtime.get("maximum_unconfirmed_timesteps"),
            "runtime.maximum_unconfirmed_timesteps",
        ),
        learning_rate=_finite_float(
            algorithm.get("learning_rate"), "algorithm.learning_rate"
        ),
        buffer_size=_positive_int(
            algorithm.get("buffer_size"), "algorithm.buffer_size"
        ),
        learning_starts=_nonnegative_int(
            algorithm.get("learning_starts"), "algorithm.learning_starts"
        ),
        batch_size=_positive_int(
            algorithm.get("batch_size"), "algorithm.batch_size"
        ),
        tau=_finite_float(algorithm.get("tau"), "algorithm.tau"),
        gamma=_finite_float(algorithm.get("gamma"), "algorithm.gamma"),
        train_freq_steps=_positive_int(
            algorithm.get("train_freq_steps"),
            "algorithm.train_freq_steps",
        ),
        gradient_steps=_positive_int(
            algorithm.get("gradient_steps"), "algorithm.gradient_steps"
        ),
        network_architecture=architecture,
        checkpoint_interval_steps=_positive_int(
            artifacts.get("checkpoint_interval_steps"),
            "artifacts.checkpoint_interval_steps",
        ),
        save_replay_buffer=bool(artifacts.get("save_replay_buffer")),
        use_vecnormalize=bool(artifacts.get("use_vecnormalize")),
        evaluation_episodes=_positive_int(
            evaluation.get("episodes"), "evaluation.episodes"
        ),
        evaluation_seed_start=_nonnegative_int(
            evaluation.get("seed_start"), "evaluation.seed_start"
        ),
        evaluation_deterministic=bool(
            evaluation.get("deterministic")
        ),
        minimum_success_rate=_finite_float(
            evaluation.get("minimum_success_rate"),
            "evaluation.minimum_success_rate",
        ),
        maximum_safety_failures=_nonnegative_int(
            evaluation.get("maximum_safety_failures"),
            "evaluation.maximum_safety_failures",
        ),
    )

    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported SAC run schema: {config.schema_version!r}"
        )
    if config.interface_version != INTERFACE_VERSION:
        raise ValueError(
            "formal SAC config does not target connector residual v0"
        )
    if config.algorithm != "SAC" or config.policy != "MlpPolicy":
        raise ValueError(
            "formal connector runner currently supports SAC/MlpPolicy"
        )
    if config.device != "cuda:0":
        raise ValueError("formal connector SAC forbids CPU fallback")
    if not config.required_torch_version:
        raise ValueError("required torch version must be explicit")
    if not config.required_cuda_build:
        raise ValueError("required CUDA build must be explicit")
    if config.action_size != ACTION_SIZE:
        raise ValueError("formal SAC action size must be 4")
    if config.observation_size != OBSERVATION_SIZE:
        raise ValueError("formal SAC observation size must be 24")
    if config.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if config.buffer_size < config.batch_size:
        raise ValueError("buffer size must be at least the batch size")
    if config.learning_starts >= config.buffer_size:
        raise ValueError("learning_starts must be below buffer_size")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not config.network_architecture:
        raise ValueError("network architecture must not be empty")
    if not config.save_replay_buffer:
        raise ValueError("formal runs must preserve the replay buffer")
    if config.use_vecnormalize:
        raise ValueError(
            "VecNormalize is intentionally disabled for the already bounded "
            "24-D residual observation"
        )
    if not config.evaluation_deterministic:
        raise ValueError("formal evaluation must use deterministic=True")
    if not 0.0 <= config.minimum_success_rate <= 1.0:
        raise ValueError("minimum success rate must be in [0, 1]")
    return config


def resolved_config_document(
    config: ConnectorResidualSACConfig,
) -> dict[str, Any]:
    """Return a stable YAML/JSON-serializable representation."""

    values = asdict(config)
    values["network_architecture"] = list(config.network_architecture)
    return values


def resolve_training_timesteps(
    requested_timesteps: int | None,
    config: ConnectorResidualSACConfig,
    *,
    allow_long_training: bool,
) -> int:
    """Require an explicit step count and an explicit long-run opt-in."""

    if requested_timesteps is None:
        raise ValueError(
            "formal training requires an explicit --formal-timesteps value"
        )
    timesteps = _positive_int(requested_timesteps, "formal timesteps")
    if (
        timesteps > config.maximum_unconfirmed_timesteps
        and not allow_long_training
    ):
        raise ValueError(
            f"{timesteps} steps exceed the unconfirmed limit "
            f"{config.maximum_unconfirmed_timesteps}; add "
            "--allow-long-training only after explicitly approving the run"
        )
    return timesteps


def file_sha256(path: str | Path) -> str:
    """Hash one required provenance input."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def state_mapping_sha256(state: Mapping[str, Any]) -> str:
    """Hash a finite numeric state mapping independent of insertion order.

    Torch is intentionally not imported here. Tensor-like values are detached
    and copied to CPU through their public methods, while pure tests can supply
    ordinary NumPy arrays.
    """

    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a nonempty mapping")
    digest = hashlib.sha256()

    def update_framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)

    for raw_name in sorted(state):
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("state names must be nonempty strings")
        value = state[raw_name]
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if callable(getattr(value, "contiguous", None)):
            value = value.contiguous()
        if callable(getattr(value, "numpy", None)):
            value = value.numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError("state values must have numeric dtypes")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("state values must contain only finite numbers")
        contiguous = np.ascontiguousarray(array)
        update_framed(raw_name.encode("utf-8"))
        update_framed(contiguous.dtype.str.encode("ascii"))
        update_framed(
            json.dumps(
                list(contiguous.shape),
                allow_nan=False,
                separators=(",", ":"),
            ).encode("ascii")
        )
        update_framed(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: Any) -> str:
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("actor module must expose state_dict()")
    return state_mapping_sha256(state_dict())


def provenance_metadata(
    paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Flatten source/asset paths and hashes using smoke-compatible keys."""

    metadata, _ = capture_provenance_snapshot(paths)
    return metadata


def capture_provenance_snapshot(
    paths: Mapping[str, str | Path],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read every provenance input once and retain those exact bytes.

    Formal training calls this before any environment preflight or learning.
    The returned hashes and any archived copies must be derived only from this
    immutable in-memory snapshot, never by rereading a mutable source path at
    the end of a long run.
    """

    result: dict[str, Any] = {}
    contents: dict[str, bytes] = {}
    for name, path in sorted(paths.items()):
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload = resolved.read_bytes()
        result[f"source_{name}_path"] = str(resolved)
        result[f"source_{name}_sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
        contents[name] = payload
    return result, contents


def resolved_backend_randomization_document(
    backend: Any,
) -> dict[str, Any] | None:
    """Return the actual loaded randomization dataclass as finite JSON data."""

    scene = getattr(backend, "scene", None)
    config = getattr(scene, "randomization_config", None)
    if config is None:
        return None
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(
            "backend scene randomization_config must be a dataclass instance"
        )
    try:
        return json.loads(
            json.dumps(asdict(config), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "loaded randomization config must be finite JSON data"
        ) from error


def _resolved_randomization_yaml(
    document: Mapping[str, Any] | None,
) -> bytes | None:
    if document is None:
        return None
    return yaml.safe_dump(
        dict(document),
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")


def resolved_backend_curriculum_document(
    backend: Any,
) -> dict[str, Any]:
    """Return and cross-check the stage actually loaded by the backend."""

    scene = getattr(backend, "scene", None)
    document = getattr(scene, "resolved_curriculum_stage", None)
    if not isinstance(document, Mapping):
        raise TypeError(
            "backend scene must expose a resolved curriculum stage mapping"
        )
    try:
        normalized = json.loads(
            json.dumps(dict(document), allow_nan=False, sort_keys=True)
        )
        resolved_config = json.loads(
            json.dumps(
                asdict(scene.residual_config),
                allow_nan=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "resolved curriculum stage must contain finite JSON data"
        ) from error
    if normalized.get("interface_version") != INTERFACE_VERSION:
        raise ValueError(
            "resolved curriculum stage targets a different interface"
        )
    if normalized.get("resolved_residual_config") != resolved_config:
        raise ValueError(
            "resolved curriculum stage differs from backend residual config"
        )
    if normalized.get("maximum_episode_steps") != getattr(
        scene, "maximum_episode_steps", None
    ):
        raise ValueError(
            "resolved curriculum stage differs from backend episode limit"
        )
    if normalized.get("minimum_axial_progress_fraction") != (
        resolved_config.get("minimum_axial_progress_fraction")
    ):
        raise ValueError(
            "resolved curriculum axial-progress threshold is inconsistent"
        )
    try:
        initial_q7 = float(normalized["initial_q7_rad"])
        planned_final_q7 = float(normalized["planned_final_q7_rad"])
        safe_lower = float(normalized["q7_safe_lower_rad"])
        safe_upper = float(normalized["q7_safe_upper_rad"])
        reserve = float(normalized["q7_command_reserve_rad"])
        checkpoint_q7 = float(
            scene.checkpoint_positions[int(scene.q7_index)]
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(
            "resolved curriculum q7 reserve proof is incomplete"
        ) from error
    q7_values = (
        initial_q7,
        planned_final_q7,
        safe_lower,
        safe_upper,
        reserve,
        checkpoint_q7,
    )
    if not all(math.isfinite(value) for value in q7_values):
        raise ValueError("resolved curriculum q7 proof must be finite")
    expected_final_q7 = (
        initial_q7
        + int(resolved_config["tightening_direction"])
        * float(resolved_config["target_angle_rad"])
    )
    if (
        reserve + 1.0e-12 < math.radians(10.0)
        or not math.isclose(initial_q7, checkpoint_q7, abs_tol=1.0e-12)
        or not math.isclose(
            planned_final_q7, expected_final_q7, abs_tol=1.0e-12
        )
        or not safe_lower + reserve <= initial_q7 <= safe_upper - reserve
        or not safe_lower + reserve
        <= planned_final_q7
        <= safe_upper - reserve
    ):
        raise ValueError(
            "resolved curriculum q7 reserve proof does not match the backend"
        )
    return normalized


def _resolved_curriculum_yaml(document: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(document),
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")


def validate_resolved_curriculum_snapshot(
    training_metadata: Mapping[str, Any],
    current_document: Mapping[str, Any],
) -> None:
    """Require evaluation to use the exact stage resolved for training."""

    if "resolved_curriculum_stage" not in training_metadata:
        raise ValueError(
            "training metadata lacks a resolved curriculum stage snapshot"
        )
    current_normalized = json.loads(
        json.dumps(dict(current_document), allow_nan=False, sort_keys=True)
    )
    if training_metadata["resolved_curriculum_stage"] != current_normalized:
        raise ValueError(
            "evaluation resolved curriculum stage differs from training"
        )
    current_hash = hashlib.sha256(
        _resolved_curriculum_yaml(current_normalized)
    ).hexdigest()
    if training_metadata.get("resolved_curriculum_stage_sha256") != (
        current_hash
    ):
        raise ValueError(
            "evaluation resolved curriculum stage hash differs from training"
        )


def validate_curriculum_provenance(
    provenance: Mapping[str, Any],
) -> None:
    """Require both the versioned curriculum YAML and resolver source."""

    required = (
        "source_curriculum_config_path",
        "source_curriculum_config_sha256",
        "source_curriculum_contract_path",
        "source_curriculum_contract_sha256",
    )
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise ValueError(
            "formal curriculum provenance is missing: "
            + ", ".join(missing)
        )


def _archive_curriculum_snapshot(
    run_directory: Path,
    source_payload: bytes,
    resolved_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Archive exact source bytes and the canonical resolved stage."""

    requested_path = run_directory / "requested_curriculum_config.yaml"
    requested_path.write_bytes(source_payload)
    resolved_payload = _resolved_curriculum_yaml(resolved_document)
    resolved_path = run_directory / "resolved_curriculum_stage.yaml"
    resolved_path.write_bytes(resolved_payload)
    return {
        "requested_curriculum_config_path": str(requested_path),
        "requested_curriculum_config_sha256": hashlib.sha256(
            source_payload
        ).hexdigest(),
        "resolved_curriculum_stage": dict(resolved_document),
        "resolved_curriculum_stage_path": str(resolved_path),
        "resolved_curriculum_stage_sha256": hashlib.sha256(
            resolved_payload
        ).hexdigest(),
    }


def validate_resolved_randomization_snapshot(
    training_metadata: Mapping[str, Any],
    current_document: Mapping[str, Any] | None,
) -> None:
    """Require evaluation to use the exact loaded training distribution."""

    if "resolved_randomization_config" not in training_metadata:
        raise ValueError(
            "training metadata lacks a resolved randomization snapshot"
        )
    trained_document = training_metadata["resolved_randomization_config"]
    current_normalized = (
        None
        if current_document is None
        else json.loads(
            json.dumps(
                dict(current_document), allow_nan=False, sort_keys=True
            )
        )
    )
    if trained_document != current_normalized:
        raise ValueError(
            "evaluation resolved randomization config differs from training"
        )
    current_payload = _resolved_randomization_yaml(current_normalized)
    current_hash = (
        None
        if current_payload is None
        else hashlib.sha256(current_payload).hexdigest()
    )
    if training_metadata.get("resolved_randomization_config_sha256") != (
        current_hash
    ):
        raise ValueError(
            "evaluation resolved randomization hash differs from training"
        )


def backend_randomization_metadata(
    backend: Any,
    *,
    history_start: int = 0,
    history_end: int | None = None,
) -> dict[str, Any]:
    """Return bounded, JSON-safe randomization evidence from the backend."""

    enabled = bool(getattr(backend, "randomization_enabled", False))
    schema_version = getattr(
        backend, "randomization_schema_version", None
    )
    physics_applied = bool(
        getattr(backend, "physics_randomization_applied", False)
    )
    safety_source = str(
        getattr(backend, "safety_signal_source", "")
    )
    if enabled and not schema_version:
        raise ValueError(
            "enabled backend randomization must expose a schema version"
        )
    if physics_applied:
        raise ValueError(
            "residual randomization v1 forbids mass/friction/thread-lead "
            "randomization"
        )
    if safety_source != "raw_physics":
        raise ValueError(
            "formal residual safety must use raw_physics signals"
        )

    raw_history = getattr(backend, "episode_randomization_history", ())
    if not isinstance(raw_history, (list, tuple)):
        raise TypeError("backend randomization history must be a sequence")
    if isinstance(history_start, bool) or not isinstance(history_start, int):
        raise TypeError("randomization history_start must be an integer")
    if history_start < 0:
        raise ValueError("randomization history_start must be nonnegative")
    if history_end is not None:
        if isinstance(history_end, bool) or not isinstance(history_end, int):
            raise TypeError("randomization history_end must be an integer")
        if history_end < history_start:
            raise ValueError(
                "randomization history_end must not precede history_start"
            )
    selected_history = raw_history[history_start:history_end]
    try:
        history = json.loads(
            json.dumps(selected_history, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "backend randomization history must be finite JSON data"
        ) from error
    return {
        "control_observation_randomization_applied": enabled,
        "episode_randomization_count": len(history),
        "episode_randomization_first": (
            None if not history else history[0]
        ),
        "episode_randomization_last": (
            None if not history else history[-1]
        ),
        "physics_parameter_randomization": {
            "friction": False,
            "mass": False,
            "thread_lead": False,
        },
        "physics_randomization_applied": False,
        "randomization_enabled": enabled,
        "randomization_schema_version": schema_version,
        "safety_signal_source": safety_source,
        "seed_scope": (
            "python_numpy_gym_action_observation_sb3_torch_cuda_backend_"
            "control_observation_delay; mass_friction_thread_lead_fixed"
            if enabled
            else "python_numpy_gym_action_observation_sb3_torch_cuda; "
            "control_observation_and_physics_fixed"
        ),
    }


def validate_randomization_provenance(
    randomization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Require the exact v1 YAML and sampler source when it is enabled."""

    if not bool(
        randomization.get(
            "control_observation_randomization_applied", False
        )
    ):
        return
    required = (
        "source_randomization_config_path",
        "source_randomization_config_sha256",
        "source_randomization_contract_path",
        "source_randomization_contract_sha256",
    )
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise ValueError(
            "enabled randomization is missing provenance: "
            + ", ".join(missing)
        )


def training_randomization_phase_verified(
    randomization: Mapping[str, Any],
    *,
    training_reset_count: int,
    expected_seed: int,
) -> bool:
    """Verify that learn-phase resets have their own seeded history slice."""

    count = int(randomization.get("episode_randomization_count", -1))
    enabled = bool(randomization.get("randomization_enabled", False))
    if not enabled:
        return count == 0
    first = randomization.get("episode_randomization_first")
    return bool(
        training_reset_count > 0
        and count == training_reset_count
        and isinstance(first, Mapping)
        and first.get("seed") == expected_seed
    )


def _new_run_directory(
    output_root: str | Path, kind: str, seed: int
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"{kind}_seed{seed}_{timestamp}"
    directory.mkdir(parents=False, exist_ok=False)
    return directory


def _runtime_modules(config: ConnectorResidualSACConfig) -> Any:
    """Load optional training dependencies and enforce the CUDA contract."""

    import gymnasium
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    import stable_baselines3
    import torch

    if torch.__version__ != config.required_torch_version:
        raise RuntimeError(
            "Isaac GPU torch changed: "
            f"{torch.__version__}; expected {config.required_torch_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if torch.version.cuda != config.required_cuda_build:
        raise RuntimeError(
            "Isaac CUDA build changed: "
            f"{torch.version.cuda}; expected {config.required_cuda_build}"
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeError("formal connector SAC requires CUDA device 0")
    probe = torch.ones(8, dtype=torch.float32, device="cuda:0")
    probe.requires_grad_(True)
    probe_loss = (probe.square().sum() / probe.numel())
    probe_loss.backward()
    if probe.grad is None or not bool(torch.all(torch.isfinite(probe.grad))):
        raise RuntimeError("finite CUDA backward probe failed")
    return {
        "CheckpointCallback": CheckpointCallback,
        "Monitor": Monitor,
        "SAC": SAC,
        "check_env": check_env,
        "gymnasium": gymnasium,
        "stable_baselines3": stable_baselines3,
        "torch": torch,
    }


def _seed_runtime(
    environment: Any, modules: Mapping[str, Any], seed: int
) -> None:
    """Seed every software RNG while stating that v0 physics is fixed."""

    random.seed(seed)
    np.random.seed(seed)
    torch = modules["torch"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)


def _model_device_metadata(model: Any, prefix: str) -> dict[str, Any]:
    """Record and fail closed on every SAC actor/critic parameter device."""

    result: dict[str, Any] = {}
    all_devices: list[str] = []
    for component_name in ("actor", "critic", "critic_target"):
        component = getattr(model, component_name)
        devices = sorted(
            {str(parameter.device) for parameter in component.parameters()}
        )
        if not devices:
            raise RuntimeError(f"SAC {component_name} has no parameters")
        result[f"{prefix}_{component_name}_parameter_devices"] = devices
        all_devices.extend(devices)
    if any(device != "cuda:0" for device in all_devices):
        unique_devices = sorted(set(all_devices))
        raise RuntimeError(
            f"{prefix} SAC parameters escaped cuda:0: {unique_devices}"
        )
    result[f"{prefix}_all_trainable_parameters_on_cuda0"] = True
    return result


def _runtime_metadata(modules: Mapping[str, Any]) -> dict[str, Any]:
    torch = modules["torch"]
    device = torch.device("cuda:0")
    return {
        "actor_device_requested": "cuda:0",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(device),
        "gymnasium": modules["gymnasium"].__version__,
        "isaacsim": importlib.metadata.version("isaacsim"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "stable_baselines3": modules["stable_baselines3"].__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }


def validate_evaluation_runtime(
    training_metadata: Mapping[str, Any] | Any,
    current_runtime: Mapping[str, Any] | Any,
) -> None:
    """Reject evaluation if a recorded runtime field is absent or changed."""

    if not isinstance(training_metadata, Mapping):
        raise TypeError("training metadata must be a mapping")
    if not isinstance(current_runtime, Mapping):
        raise TypeError("current runtime metadata must be a mapping")
    for field in EVALUATION_RUNTIME_COMPATIBILITY_FIELDS:
        trained_value = training_metadata.get(field)
        current_value = current_runtime.get(field)
        if (
            isinstance(trained_value, bool)
            or not isinstance(trained_value, str)
            or not trained_value
        ):
            raise ValueError(
                f"training runtime field {field!r} is missing or invalid"
            )
        if (
            isinstance(current_value, bool)
            or not isinstance(current_value, str)
            or not current_value
        ):
            raise ValueError(
                f"current runtime field {field!r} is missing or invalid"
            )
        if trained_value != current_value:
            raise ValueError(
                f"evaluation runtime mismatch for {field}: "
                f"trained={trained_value!r}, current={current_value!r}"
            )


def _backend_reset_summary(backend: Any) -> tuple[dict[str, float], bool]:
    from kcg_connector.isaac_residual_backend import (
        summarize_reset_diagnostics,
    )

    return summarize_reset_diagnostics(backend.reset_diagnostics)


def _runtime_thread_prim_count(backend: Any) -> int:
    prefix = backend.scene.thread_spec.runtime_root + "/"
    return sum(
        1
        for prim in backend.scene.stage.Traverse()
        if str(prim.GetPath()).startswith(prefix)
    )


def _episode_buffer_summary(model: Any) -> dict[str, Any]:
    records = list(model.ep_info_buffer or ())
    if not records:
        return {
            "completed_training_episodes": 0,
            "mean_training_episode_length": None,
            "mean_training_episode_return": None,
        }
    return {
        "completed_training_episodes": len(records),
        "mean_training_episode_length": float(
            np.mean([record["l"] for record in records])
        ),
        "mean_training_episode_return": float(
            np.mean([record["r"] for record in records])
        ),
    }


def run_formal_training(
    environment: Any,
    backend: Any,
    config: ConnectorResidualSACConfig,
    *,
    requested_timesteps: int | None,
    allow_long_training: bool,
    output_root: str | Path,
    provenance_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Run one guarded single-environment SAC training job."""

    timesteps = resolve_training_timesteps(
        requested_timesteps,
        config,
        allow_long_training=allow_long_training,
    )
    modules = _runtime_modules(config)
    torch = modules["torch"]
    _seed_runtime(environment, modules, config.seed)
    initial_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    run_directory = _new_run_directory(output_root, "train", config.seed)
    run_provenance, frozen_source_bytes = capture_provenance_snapshot(
        provenance_paths
    )
    validate_curriculum_provenance(run_provenance)
    if "curriculum_config" not in frozen_source_bytes:
        raise ValueError(
            "formal training did not freeze the curriculum YAML bytes"
        )
    resolved_curriculum_document = (
        resolved_backend_curriculum_document(backend)
    )
    curriculum_snapshot_metadata = _archive_curriculum_snapshot(
        run_directory,
        frozen_source_bytes["curriculum_config"],
        resolved_curriculum_document,
    )
    initial_randomization_metadata = backend_randomization_metadata(backend)
    validate_randomization_provenance(
        initial_randomization_metadata, run_provenance
    )
    requested_config_path = run_directory / "requested_training_config.yaml"
    requested_config_payload = frozen_source_bytes["training_config"]
    requested_config_path.write_bytes(requested_config_payload)
    resolved_config_path = run_directory / "resolved_training_config.yaml"
    resolved_config_payload = yaml.safe_dump(
        resolved_config_document(config),
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    resolved_config_path.write_bytes(resolved_config_payload)
    resolved_randomization_document = (
        resolved_backend_randomization_document(backend)
    )
    resolved_randomization_payload = _resolved_randomization_yaml(
        resolved_randomization_document
    )
    resolved_randomization_path = None
    resolved_randomization_sha256 = None
    if resolved_randomization_payload is not None:
        resolved_randomization_path = (
            run_directory / "resolved_randomization_config.yaml"
        )
        resolved_randomization_path.write_bytes(
            resolved_randomization_payload
        )
        resolved_randomization_sha256 = hashlib.sha256(
            resolved_randomization_payload
        ).hexdigest()

    preflight_reset_start = int(backend.reset_count)
    preflight_history_start = len(backend.episode_randomization_history)
    modules["check_env"](
        environment, warn=True, skip_render_check=True
    )
    preflight_reset_count = int(backend.reset_count) - preflight_reset_start
    preflight_history_end = len(backend.episode_randomization_history)
    preflight_randomization_count = (
        preflight_history_end - preflight_history_start
    )
    _seed_runtime(environment, modules, config.seed)
    training_raw_safety_audit = TrainingRawSafetyAudit()
    audited_environment = _wrap_training_environment_for_raw_safety(
        environment,
        backend,
        training_raw_safety_audit,
        modules["gymnasium"],
    )
    monitor = modules["Monitor"](
        audited_environment, filename=str(run_directory / "monitor")
    )
    checkpoint = None
    if timesteps >= config.checkpoint_interval_steps:
        checkpoint = modules["CheckpointCallback"](
            save_freq=config.checkpoint_interval_steps,
            save_path=str(run_directory / "checkpoints"),
            name_prefix="connector_residual_sac",
            save_replay_buffer=config.save_replay_buffer,
            save_vecnormalize=False,
        )
    model = modules["SAC"](
        config.policy,
        monitor,
        device=config.device,
        seed=config.seed,
        verbose=1,
        learning_rate=config.learning_rate,
        learning_starts=config.learning_starts,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq_steps,
        gradient_steps=config.gradient_steps,
        policy_kwargs={"net_arch": list(config.network_architecture)},
    )
    model_device_metadata = _model_device_metadata(model, "training")
    actor_before = torch.cat(
        [
            parameter.detach().flatten().cpu()
            for parameter in model.actor.parameters()
        ]
    )
    actor_initial_state_sha256 = _module_state_sha256(model.actor)
    training_reset_start = int(backend.reset_count)
    training_history_start = len(backend.episode_randomization_history)
    model.learn(total_timesteps=timesteps, callback=checkpoint)
    training_raw_safety_report = (
        training_raw_safety_audit.finalize()
    )
    training_reset_count = int(backend.reset_count) - training_reset_start
    training_history_end = len(backend.episode_randomization_history)
    actor_after = torch.cat(
        [
            parameter.detach().flatten().cpu()
            for parameter in model.actor.parameters()
        ]
    )
    actor_parameter_delta = float(
        torch.max(torch.abs(actor_after - actor_before))
    )
    actor_final_state_sha256 = _module_state_sha256(model.actor)

    model_base_path = run_directory / "final_model"
    model.save(str(model_base_path))
    model_path = model_base_path.with_suffix(".zip")
    replay_path = run_directory / "replay_buffer.pkl"
    model.save_replay_buffer(str(replay_path))
    reloaded = modules["SAC"].load(
        str(model_path), env=monitor, device=config.device
    )
    reloaded_device_metadata = _model_device_metadata(reloaded, "reloaded")
    reloaded_actor_state_sha256 = _module_state_sha256(reloaded.actor)
    actor_reload_verified = bool(
        reloaded_actor_state_sha256 == actor_final_state_sha256
    )
    actor_device = str(next(model.actor.parameters()).device)
    reloaded_actor_device = str(next(reloaded.actor.parameters()).device)
    replay_size = int(model.replay_buffer.size())
    reset_snap_maxima, reset_snap_ok = _backend_reset_summary(backend)
    final_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    optimization_expected = bool(timesteps > config.learning_starts)
    optimization_verified = bool(
        not optimization_expected
        or (
            int(model._n_updates) > 0
            and math.isfinite(actor_parameter_delta)
            and actor_parameter_delta > 0.0
        )
    )
    randomization_metadata = backend_randomization_metadata(
        backend,
        history_start=training_history_start,
        history_end=training_history_end,
    )
    training_randomization_verified = (
        training_randomization_phase_verified(
            randomization_metadata,
            training_reset_count=training_reset_count,
            expected_seed=config.seed,
        )
    )
    structural_passed = bool(
        int(model.num_timesteps) >= timesteps
        and replay_size > 0
        and actor_device.startswith("cuda")
        and reloaded_actor_device.startswith("cuda")
        and model_path.is_file()
        and replay_path.is_file()
        and reset_snap_ok
        and backend.thread_proxy_rebuild_count == backend.reset_count
        and final_runtime_thread_prim_count
        == initial_runtime_thread_prim_count
        and optimization_verified
        and actor_reload_verified
        and training_randomization_verified
        and training_raw_safety_report["passed"] is True
        and training_raw_safety_report["policy_steps_audited"]
        == int(model.num_timesteps)
    )
    metadata: dict[str, Any] = {
        "action_size": ACTION_SIZE,
        "actor_device": actor_device,
        "actor_final_state_sha256": actor_final_state_sha256,
        "actor_initial_state_sha256": actor_initial_state_sha256,
        "actor_parameter_max_delta": actor_parameter_delta,
        "actor_reload_verified": actor_reload_verified,
        "algorithm": config.algorithm,
        "dry_run": bool(
            timesteps <= config.maximum_unconfirmed_timesteps
        ),
        "environment_checker": "passed",
        "environment_hard_resets": backend.reset_count,
        "interface_version": config.interface_version,
        "learning_starts": config.learning_starts,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "model_timesteps": int(model.num_timesteps),
        "observation_size": OBSERVATION_SIZE,
        "optimization_expected": optimization_expected,
        "optimization_verified": optimization_verified,
        "optimizer_updates": int(model._n_updates),
        "passed": structural_passed,
        "policy_competence_claim": False,
        "policy_seed": config.seed,
        "preflight_hard_resets": preflight_reset_count,
        "preflight_randomization_count": (
            preflight_randomization_count
        ),
        "random_seed": config.seed,
        "reloaded_actor_device": reloaded_actor_device,
        "reloaded_actor_state_sha256": reloaded_actor_state_sha256,
        "replay_buffer_path": str(replay_path),
        "replay_buffer_sha256": file_sha256(replay_path),
        "replay_size": replay_size,
        "requested_timesteps": timesteps,
        "residual_curriculum_stage": (
            resolved_curriculum_document["stage_name"]
        ),
        "resolved_randomization_config": (
            resolved_randomization_document
        ),
        "resolved_randomization_config_path": (
            None
            if resolved_randomization_path is None
            else str(resolved_randomization_path)
        ),
        "resolved_randomization_config_sha256": (
            resolved_randomization_sha256
        ),
        "reset_checkpoint_snap_maxima": reset_snap_maxima,
        "reset_checkpoint_snap_ok": reset_snap_ok,
        "run_directory": str(run_directory),
        "run_kind": "formal_train",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config.schema_version,
        "scene_build_count": 1,
        "simulation_app_count": 1,
        "thread_proxy_rebuild_count": backend.thread_proxy_rebuild_count,
        "thread_proxy_reset_strategy": (
            "remove_hard_reset_contact_recovery_recreate"
        ),
        "runtime_thread_prim_count": final_runtime_thread_prim_count,
        "runtime_thread_prim_count_initial": (
            initial_runtime_thread_prim_count
        ),
        "physicsusd_disjoint_warning_count": None,
        "physicsusd_disjoint_warning_count_source": (
            "not_captured_in_process; inspect the matching Kit log"
        ),
        "training_completed": True,
        "training_config": resolved_config_document(config),
        "training_hard_resets": training_reset_count,
        "training_randomization_phase_verified": (
            training_randomization_verified
        ),
        "training_raw_safety_complete_episode_count": (
            training_raw_safety_report["complete_episode_count"]
        ),
        "training_raw_safety_failure_reasons": (
            training_raw_safety_report["failure_reasons"]
        ),
        "training_raw_safety_partial_episode_count": (
            training_raw_safety_report["partial_episode_count"]
        ),
        "training_raw_safety_passed": (
            training_raw_safety_report["passed"]
        ),
        "training_raw_safety_peaks": (
            training_raw_safety_report["peaks"]
        ),
        "training_raw_safety_policy_steps_audited": (
            training_raw_safety_report["policy_steps_audited"]
        ),
        "training_raw_safety_report": training_raw_safety_report,
        "vecnormalize_path": None,
        "vecnormalize_used": False,
    }
    metadata.update(_episode_buffer_summary(model))
    metadata.update(randomization_metadata)
    metadata.update(model_device_metadata)
    metadata.update(reloaded_device_metadata)
    metadata.update(_runtime_metadata(modules))
    metadata.update(run_provenance)
    metadata.update(curriculum_snapshot_metadata)
    metadata["source_resolved_training_config_path"] = str(
        resolved_config_path
    )
    metadata["source_resolved_training_config_sha256"] = hashlib.sha256(
        resolved_config_payload
    ).hexdigest()
    metadata["source_requested_training_config_path"] = str(
        requested_config_path
    )
    metadata["source_requested_training_config_sha256"] = hashlib.sha256(
        requested_config_payload
    ).hexdigest()
    metadata_path = run_directory / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    metadata["metadata_path"] = str(metadata_path)
    monitor.close()
    return metadata


def load_training_metadata_for_model(
    model_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Load the mandatory provenance record beside a formal model."""

    model = Path(model_path).expanduser().resolve()
    metadata_path = model.parent / "training_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"formal model metadata is missing: {metadata_path}"
        )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("model metadata has an incompatible schema")
    if document.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("model metadata targets a different task interface")
    if document.get("action_size") != ACTION_SIZE:
        raise ValueError("model metadata action size is incompatible")
    if document.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("model metadata observation size is incompatible")
    if document.get("vecnormalize_used") is not False:
        raise ValueError("formal v0 evaluation does not support VecNormalize")
    if document.get("model_sha256") != file_sha256(model):
        raise ValueError("model hash does not match its training metadata")
    return metadata_path, document


def validate_loaded_actor_training_binding(
    training_metadata: Mapping[str, Any],
    config: ConnectorResidualSACConfig,
    loaded_actor_state_sha256: str,
) -> None:
    """Bind an evaluated actor to its resolved training record."""

    if not isinstance(training_metadata, Mapping):
        raise TypeError("training metadata must be a mapping")
    if not _valid_sha256(loaded_actor_state_sha256):
        raise ValueError("loaded actor state hash is invalid")
    final_hash = training_metadata.get("actor_final_state_sha256")
    reloaded_hash = training_metadata.get(
        "reloaded_actor_state_sha256"
    )
    if not _valid_sha256(final_hash) or final_hash != (
        loaded_actor_state_sha256
    ):
        raise ValueError(
            "loaded actor state does not match training metadata"
        )
    if (
        training_metadata.get("actor_reload_verified") is not True
        or reloaded_hash != final_hash
    ):
        raise ValueError(
            "training metadata lacks a verified saved-actor reload"
        )

    expected_config = resolved_config_document(config)
    trained_config = training_metadata.get("training_config")
    if trained_config != expected_config:
        raise ValueError(
            "evaluation SAC configuration differs from training"
        )
    archive_path_value = training_metadata.get(
        "source_resolved_training_config_path"
    )
    archive_hash = training_metadata.get(
        "source_resolved_training_config_sha256"
    )
    if (
        type(archive_path_value) is not str
        or not archive_path_value
        or not _valid_sha256(archive_hash)
    ):
        raise ValueError(
            "training metadata lacks a resolved config archive"
        )
    archive_path = Path(archive_path_value).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    payload = archive_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != archive_hash:
        raise ValueError("resolved training config archive hash changed")
    archived_document = yaml.safe_load(payload)
    if archived_document != expected_config:
        raise ValueError(
            "resolved training config archive content differs"
        )


def validate_evaluation_provenance(
    training_metadata: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> None:
    """Reject evaluation with changed code, config or physical assets."""

    for key, value in current_provenance.items():
        if not key.endswith("_sha256"):
            continue
        trained_value = training_metadata.get(key)
        if trained_value != value:
            raise ValueError(
                f"evaluation provenance mismatch for {key}: "
                f"trained={trained_value!r}, current={value!r}"
            )


def _strict_finite_number(
    value: Any, name: str, *, positive: bool = False
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def validate_raw_safety_report(value: Any) -> dict[str, Any]:
    """Strictly validate one backend raw-physics safety report."""

    if not isinstance(value, Mapping):
        raise ValueError("raw safety report must be a mapping")
    if set(value) != RAW_SAFETY_REPORT_FIELDS:
        raise ValueError("raw safety report has an incompatible schema")
    passed = value["passed"]
    finite_throughout = value["finite_throughout"]
    if type(passed) is not bool:
        raise ValueError("raw safety passed must be a boolean")
    if type(finite_throughout) is not bool:
        raise ValueError(
            "raw safety finite_throughout must be a boolean"
        )
    if value["signal_source"] != "raw_physics":
        raise ValueError("raw safety signal source must be raw_physics")

    reasons = value["failure_reasons"]
    if not isinstance(reasons, list) or any(
        type(reason) is not str or not reason for reason in reasons
    ):
        raise ValueError(
            "raw safety failure reasons must be nonempty strings"
        )
    if len(reasons) != len(set(reasons)):
        raise ValueError("raw safety failure reasons must be unique")

    raw_metrics = value["metrics"]
    if (
        not isinstance(raw_metrics, Mapping)
        or set(raw_metrics) != RAW_SAFETY_METRIC_FIELDS
    ):
        raise ValueError("raw safety metrics have an incompatible schema")
    metrics = {
        name: _strict_finite_number(
            raw_metrics[name], f"raw safety metric {name}"
        )
        for name in sorted(RAW_SAFETY_METRIC_FIELDS)
    }

    raw_limits = value["limits"]
    if (
        not isinstance(raw_limits, Mapping)
        or set(raw_limits) != RAW_SAFETY_LIMIT_FIELDS
    ):
        raise ValueError("raw safety limits have an incompatible schema")
    limits = {
        name: _strict_finite_number(
            raw_limits[name], f"raw safety limit {name}"
        )
        for name in sorted(RAW_SAFETY_LIMIT_FIELDS)
    }

    raw_sampling = value["sampling"]
    if not isinstance(raw_sampling, Mapping) or set(raw_sampling) != {
        "physics_substep",
        "policy_boundary",
    }:
        raise ValueError("raw safety sampling has an incompatible schema")
    sampling: dict[str, dict[str, Any]] = {}
    for name in ("physics_substep", "policy_boundary"):
        entry = raw_sampling[name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "includes_episode_initial_snapshot",
            "rate_hz",
            "samples",
        }:
            raise ValueError(
                f"raw safety sampling {name} has an incompatible schema"
            )
        if entry["includes_episode_initial_snapshot"] is not True:
            raise ValueError(
                "raw safety sampling must include the initial snapshot"
            )
        samples = entry["samples"]
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, Integral)
            or int(samples) <= 0
        ):
            raise ValueError(
                f"raw safety sampling {name} samples must be positive"
            )
        sampling[name] = {
            "includes_episode_initial_snapshot": True,
            "rate_hz": _strict_finite_number(
                entry["rate_hz"],
                f"raw safety sampling {name} rate",
                positive=True,
            ),
            "samples": int(samples),
        }

    within_limits = all(
        metrics[name] <= limits[name]
        for name in RAW_SAFETY_LIMIT_FIELDS
    )
    computed_passed = bool(
        finite_throughout and within_limits and not reasons
    )
    if passed != computed_passed:
        raise ValueError(
            "raw safety passed flag is inconsistent with its evidence"
        )
    normalized = {
        "failure_reasons": list(reasons),
        "finite_throughout": finite_throughout,
        "limits": limits,
        "metrics": metrics,
        "passed": passed,
        "sampling": sampling,
        "signal_source": "raw_physics",
    }
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return normalized


def _validate_raw_safety_projection(
    info: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    if not isinstance(info, Mapping):
        raise ValueError("final episode info must be a mapping")
    if info.get("raw_safety_passed") is not report["passed"]:
        raise ValueError("final info raw safety passed flag differs")
    if info.get("raw_safety_failure_reasons") != report[
        "failure_reasons"
    ]:
        raise ValueError("final info raw safety failure reasons differ")
    if info.get("raw_safety_peaks") != report["metrics"]:
        raise ValueError("final info raw safety peaks differ")
    if info.get("safety_signal_source") != "raw_physics":
        raise ValueError("final info safety source must be raw_physics")


def _copy_json_safe(value: Any) -> Any:
    """Return a detached JSON-safe copy while rejecting NaN and infinity."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _empty_raw_safety_values(fields: frozenset[str]) -> dict[str, None]:
    return {name: None for name in sorted(fields)}


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _validate_nullable_raw_values(
    value: Any, fields: frozenset[str], name: str
) -> dict[str, float | None]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an incompatible schema")
    normalized: dict[str, float | None] = {}
    for field in sorted(fields):
        item = value[field]
        if item is None:
            normalized[field] = None
        else:
            normalized[field] = _strict_finite_number(
                item, f"{name} {field}"
            )
    return normalized


def _validate_training_sampling(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "physics_substep",
        "policy_boundary",
    }:
        raise ValueError("training raw safety sampling schema is invalid")
    result: dict[str, dict[str, Any]] = {}
    for name in ("physics_substep", "policy_boundary"):
        entry = value[name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "includes_episode_initial_snapshot",
            "rate_hz",
            "samples",
        }:
            raise ValueError(
                "training raw safety sampling entry is invalid"
            )
        samples = entry["samples"]
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, Integral)
            or int(samples) <= 0
        ):
            raise ValueError(
                "training raw safety sampling count is invalid"
            )
        if entry["includes_episode_initial_snapshot"] is not True:
            raise ValueError(
                "training raw safety sampling lacks initial snapshot"
            )
        result[name] = {
            "includes_episode_initial_snapshot": True,
            "rate_hz": _strict_finite_number(
                entry["rate_hz"],
                f"training raw safety {name} rate",
                positive=True,
            ),
            "samples": int(samples),
        }
    return result


def validate_training_raw_safety_report(value: Any) -> dict[str, Any]:
    """Validate the exact, JSON-safe raw evidence for one learn call."""

    if not isinstance(value, Mapping):
        raise ValueError("training raw safety report must be a mapping")
    if set(value) != TRAINING_RAW_SAFETY_REPORT_FIELDS:
        raise ValueError(
            "training raw safety report has an incompatible schema"
        )
    if value["schema_version"] != TRAINING_RAW_SAFETY_SCHEMA_VERSION:
        raise ValueError("training raw safety schema version is invalid")
    if value["signal_source"] != "raw_physics":
        raise ValueError("training raw safety source must be raw_physics")
    for field in (
        "passed",
        "finite_throughout",
        "evidence_valid_throughout",
    ):
        if type(value[field]) is not bool:
            raise ValueError(f"training raw safety {field} is invalid")

    counts: dict[str, int] = {}
    for field in (
        "policy_steps_audited",
        "complete_episode_count",
        "partial_episode_count",
    ):
        item = value[field]
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, Integral)
            or int(item) < 0
        ):
            raise ValueError(
                f"training raw safety {field} is invalid"
            )
        counts[field] = int(item)

    reasons = value["failure_reasons"]
    if not isinstance(reasons, list) or any(
        type(reason) is not str or not reason for reason in reasons
    ):
        raise ValueError("training raw safety reasons are invalid")
    if len(reasons) != len(set(reasons)):
        raise ValueError("training raw safety reasons are not unique")
    peaks = _validate_nullable_raw_values(
        value["peaks"], RAW_SAFETY_METRIC_FIELDS,
        "training raw safety peaks",
    )
    limits = _validate_nullable_raw_values(
        value["limits"], RAW_SAFETY_LIMIT_FIELDS,
        "training raw safety limits",
    )

    raw_episodes = value["episode_reports"]
    if not isinstance(raw_episodes, list):
        raise ValueError("training raw safety episodes must be a list")
    episodes: list[dict[str, Any]] = []
    for raw_episode in raw_episodes:
        if (
            not isinstance(raw_episode, Mapping)
            or set(raw_episode) != TRAINING_RAW_SAFETY_EPISODE_FIELDS
        ):
            raise ValueError(
                "training raw safety episode has an incompatible schema"
            )
        for field in (
            "complete",
            "passed",
            "finite_throughout",
            "evidence_valid_throughout",
        ):
            if type(raw_episode[field]) is not bool:
                raise ValueError(
                    f"training raw safety episode {field} is invalid"
                )
        policy_steps = raw_episode["policy_steps"]
        if (
            isinstance(policy_steps, (bool, np.bool_))
            or not isinstance(policy_steps, Integral)
            or int(policy_steps) <= 0
        ):
            raise ValueError(
                "training raw safety episode policy_steps is invalid"
            )
        if raw_episode["signal_source"] != "raw_physics":
            raise ValueError(
                "training raw safety episode source is invalid"
            )
        episode_reasons = raw_episode["failure_reasons"]
        if not isinstance(episode_reasons, list) or any(
            type(reason) is not str or not reason
            for reason in episode_reasons
        ):
            raise ValueError(
                "training raw safety episode reasons are invalid"
            )
        if len(episode_reasons) != len(set(episode_reasons)):
            raise ValueError(
                "training raw safety episode reasons are not unique"
            )
        episode_peaks = _validate_nullable_raw_values(
            raw_episode["peaks"], RAW_SAFETY_METRIC_FIELDS,
            "training raw safety episode peaks",
        )
        episode_limits = _validate_nullable_raw_values(
            raw_episode["limits"], RAW_SAFETY_LIMIT_FIELDS,
            "training raw safety episode limits",
        )
        raw_sampling = raw_episode["last_sampling"]
        sampling = (
            None
            if raw_sampling is None
            else _validate_training_sampling(raw_sampling)
        )
        complete_evidence = all(
            item is not None for item in episode_peaks.values()
        ) and all(item is not None for item in episode_limits.values())
        within_limits = complete_evidence and all(
            episode_peaks[name] <= episode_limits[name]
            for name in RAW_SAFETY_LIMIT_FIELDS
        )
        computed_episode_passed = bool(
            raw_episode["evidence_valid_throughout"]
            and raw_episode["finite_throughout"]
            and complete_evidence
            and within_limits
            and not episode_reasons
        )
        if raw_episode["passed"] != computed_episode_passed:
            raise ValueError(
                "training raw safety episode passed is inconsistent"
            )
        episodes.append(
            {
                "complete": raw_episode["complete"],
                "evidence_valid_throughout": raw_episode[
                    "evidence_valid_throughout"
                ],
                "failure_reasons": list(episode_reasons),
                "finite_throughout": raw_episode["finite_throughout"],
                "last_sampling": sampling,
                "limits": episode_limits,
                "passed": raw_episode["passed"],
                "peaks": episode_peaks,
                "policy_steps": int(policy_steps),
                "signal_source": "raw_physics",
            }
        )

    complete_count = sum(episode["complete"] for episode in episodes)
    partial_count = len(episodes) - complete_count
    policy_steps = sum(episode["policy_steps"] for episode in episodes)
    if complete_count != counts["complete_episode_count"]:
        raise ValueError(
            "training raw safety complete episode count differs"
        )
    if partial_count != counts["partial_episode_count"]:
        raise ValueError(
            "training raw safety partial episode count differs"
        )
    if policy_steps != counts["policy_steps_audited"]:
        raise ValueError("training raw safety policy step count differs")
    complete_evidence = bool(episodes) and all(
        item is not None for item in peaks.values()
    ) and all(item is not None for item in limits.values())
    within_limits = complete_evidence and all(
        peaks[name] <= limits[name]
        for name in RAW_SAFETY_LIMIT_FIELDS
    )
    computed_passed = bool(
        policy_steps > 0
        and value["evidence_valid_throughout"]
        and value["finite_throughout"]
        and complete_evidence
        and within_limits
        and all(episode["passed"] for episode in episodes)
        and not reasons
    )
    if value["passed"] != computed_passed:
        raise ValueError("training raw safety passed is inconsistent")
    normalized = {
        "complete_episode_count": complete_count,
        "episode_reports": episodes,
        "evidence_valid_throughout": value[
            "evidence_valid_throughout"
        ],
        "failure_reasons": list(reasons),
        "finite_throughout": value["finite_throughout"],
        "limits": limits,
        "partial_episode_count": partial_count,
        "passed": value["passed"],
        "peaks": peaks,
        "policy_steps_audited": policy_steps,
        "schema_version": TRAINING_RAW_SAFETY_SCHEMA_VERSION,
        "signal_source": "raw_physics",
    }
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return normalized


class TrainingRawSafetyAudit:
    """Latch raw-physics evidence at every policy step of ``learn``."""

    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []
        self._current = self._new_episode()
        self._finalized_report: dict[str, Any] | None = None

    @staticmethod
    def _new_episode() -> dict[str, Any]:
        return {
            "evidence_valid_throughout": True,
            "failure_reasons": [],
            "finite_throughout": True,
            "last_report": None,
            "last_sampling": None,
            "limits": _empty_raw_safety_values(
                RAW_SAFETY_LIMIT_FIELDS
            ),
            "peaks": _empty_raw_safety_values(
                RAW_SAFETY_METRIC_FIELDS
            ),
            "policy_steps": 0,
            "unsafe_seen": False,
        }

    def _failure(
        self,
        reason: str,
        *,
        evidence_invalid: bool = False,
        finite_unverified: bool = False,
    ) -> None:
        _append_unique(self._current["failure_reasons"], reason)
        if evidence_invalid:
            self._current["evidence_valid_throughout"] = False
        if finite_unverified:
            self._current["finite_throughout"] = False

    def note_reset(self) -> None:
        """Mark an unexpected reset that did not follow done."""

        if self._finalized_report is not None:
            raise RuntimeError("training raw safety audit is finalized")
        if self._current["policy_steps"] > 0:
            self._failure(
                "unexpected_reset_before_episode_end",
                evidence_invalid=True,
            )
            self._close_episode(complete=False)

    def record_step(
        self,
        raw_report: Any,
        info: Any,
        *,
        episode_done: bool,
    ) -> None:
        """Validate and latch one inner Gym step before VecEnv autoreset."""

        if self._finalized_report is not None:
            raise RuntimeError("training raw safety audit is finalized")
        self._current["policy_steps"] += 1
        try:
            report = validate_raw_safety_report(raw_report)
        except (TypeError, ValueError):
            self._failure(
                "backend_raw_safety_report_invalid",
                evidence_invalid=True,
                finite_unverified=True,
            )
            if episode_done:
                self._close_episode(complete=True)
            return

        previous = self._current["last_report"]
        if previous is not None:
            if previous["passed"] is False and report["passed"] is True:
                self._failure(
                    "backend_raw_safety_passed_recovered",
                    evidence_invalid=True,
                )
            for name in RAW_SAFETY_METRIC_FIELDS:
                if report["metrics"][name] < previous["metrics"][name]:
                    self._failure(
                        "backend_raw_safety_peak_decreased",
                        evidence_invalid=True,
                    )
            for name in ("physics_substep", "policy_boundary"):
                prior_sampling = previous["sampling"][name]
                sampling = report["sampling"][name]
                if (
                    sampling["rate_hz"] != prior_sampling["rate_hz"]
                    or sampling["samples"] < prior_sampling["samples"]
                ):
                    self._failure(
                        "backend_raw_safety_sampling_regressed",
                        evidence_invalid=True,
                    )

        for name, value in report["metrics"].items():
            current = self._current["peaks"][name]
            self._current["peaks"][name] = (
                value if current is None else max(current, value)
            )
        for name, value in report["limits"].items():
            current = self._current["limits"][name]
            if current is not None and current != value:
                self._failure(
                    "backend_raw_safety_limits_changed",
                    evidence_invalid=True,
                )
            self._current["limits"][name] = (
                value if current is None else min(current, value)
            )
        self._current["finite_throughout"] = bool(
            self._current["finite_throughout"]
            and report["finite_throughout"]
        )
        self._current["last_sampling"] = report["sampling"]
        self._current["last_report"] = report
        if report["passed"] is False:
            self._current["unsafe_seen"] = True
            self._failure("backend_raw_safety_failed")
            for reason in report["failure_reasons"]:
                self._failure(reason)
        try:
            _validate_raw_safety_projection(info, report)
        except (TypeError, ValueError):
            self._failure(
                "step_info_raw_safety_projection_invalid",
                evidence_invalid=True,
            )
        if episode_done:
            self._close_episode(complete=True)

    def _close_episode(self, *, complete: bool) -> None:
        current = self._current
        if current["policy_steps"] <= 0:
            return
        complete_evidence = all(
            item is not None for item in current["peaks"].values()
        ) and all(
            item is not None for item in current["limits"].values()
        )
        within_limits = complete_evidence and all(
            current["peaks"][name] <= current["limits"][name]
            for name in RAW_SAFETY_LIMIT_FIELDS
        )
        passed = bool(
            current["evidence_valid_throughout"]
            and current["finite_throughout"]
            and complete_evidence
            and within_limits
            and not current["failure_reasons"]
        )
        self._episodes.append(
            {
                "complete": bool(complete),
                "evidence_valid_throughout": current[
                    "evidence_valid_throughout"
                ],
                "failure_reasons": list(current["failure_reasons"]),
                "finite_throughout": current["finite_throughout"],
                "last_sampling": current["last_sampling"],
                "limits": dict(current["limits"]),
                "passed": passed,
                "peaks": dict(current["peaks"]),
                "policy_steps": current["policy_steps"],
                "signal_source": "raw_physics",
            }
        )
        self._current = self._new_episode()

    def finalize(self) -> dict[str, Any]:
        """Close the final partial episode and return exact evidence."""

        if self._finalized_report is not None:
            return _copy_json_safe(self._finalized_report)
        self._close_episode(complete=False)
        reasons: list[str] = []
        peaks = _empty_raw_safety_values(RAW_SAFETY_METRIC_FIELDS)
        limits = _empty_raw_safety_values(RAW_SAFETY_LIMIT_FIELDS)
        limits_changed = False
        for episode in self._episodes:
            for reason in episode["failure_reasons"]:
                _append_unique(reasons, reason)
            for name, value in episode["peaks"].items():
                if value is not None:
                    current = peaks[name]
                    peaks[name] = (
                        value if current is None else max(current, value)
                    )
            for name, value in episode["limits"].items():
                if value is None:
                    continue
                current = limits[name]
                if current is not None and current != value:
                    limits_changed = True
                limits[name] = (
                    value if current is None else min(current, value)
                )
        if limits_changed:
            _append_unique(
                reasons,
                "backend_raw_safety_limits_changed_between_episodes",
            )
        policy_steps = sum(
            episode["policy_steps"] for episode in self._episodes
        )
        complete_count = sum(
            episode["complete"] for episode in self._episodes
        )
        partial_count = len(self._episodes) - complete_count
        evidence_valid = bool(self._episodes) and all(
            episode["evidence_valid_throughout"]
            for episode in self._episodes
        ) and not limits_changed
        finite_throughout = bool(self._episodes) and all(
            episode["finite_throughout"]
            for episode in self._episodes
        )
        complete_evidence = all(
            item is not None for item in peaks.values()
        ) and all(item is not None for item in limits.values())
        within_limits = complete_evidence and all(
            peaks[name] <= limits[name]
            for name in RAW_SAFETY_LIMIT_FIELDS
        )
        if policy_steps == 0:
            _append_unique(reasons, "no_policy_steps_audited")
        passed = bool(
            policy_steps > 0
            and evidence_valid
            and finite_throughout
            and complete_evidence
            and within_limits
            and all(episode["passed"] for episode in self._episodes)
            and not reasons
        )
        report = {
            "complete_episode_count": complete_count,
            "episode_reports": self._episodes,
            "evidence_valid_throughout": evidence_valid,
            "failure_reasons": reasons,
            "finite_throughout": finite_throughout,
            "limits": limits,
            "partial_episode_count": partial_count,
            "passed": passed,
            "peaks": peaks,
            "policy_steps_audited": policy_steps,
            "schema_version": TRAINING_RAW_SAFETY_SCHEMA_VERSION,
            "signal_source": "raw_physics",
        }
        self._finalized_report = validate_training_raw_safety_report(
            report
        )
        return _copy_json_safe(self._finalized_report)


def _wrap_training_environment_for_raw_safety(
    environment: Any,
    backend: Any,
    audit: TrainingRawSafetyAudit,
    gymnasium: Any,
) -> Any:
    """Place the audit inside Monitor and VecEnv autoreset boundaries."""

    class RuntimeRawSafetyAuditWrapper(gymnasium.Wrapper):
        def reset(self, **kwargs: Any) -> Any:
            audit.note_reset()
            return self.env.reset(**kwargs)

        def step(self, action: Any) -> Any:
            transition = self.env.step(action)
            if not isinstance(transition, tuple) or len(transition) != 5:
                raise RuntimeError(
                    "raw safety audit requires a Gymnasium 5-tuple"
                )
            observation, reward, terminated, truncated, info = transition
            try:
                report = backend.raw_safety_report
            except Exception:
                report = None
            audit.record_step(
                report,
                info,
                episode_done=bool(terminated) or bool(truncated),
            )
            return observation, reward, terminated, truncated, info

    return RuntimeRawSafetyAuditWrapper(environment)


def physical_episode_report(
    backend: Any,
    observation: np.ndarray,
    *,
    episode: int,
    episode_return: float,
    episode_steps: int,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
    final_info: Mapping[str, Any],
    reset_info: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Produce the same physical success/safety fields as zero residual."""

    from kcg_connector.residual_rl import loaded_torque_channels

    raw_safety = validate_raw_safety_report(backend.raw_safety_report)
    _validate_raw_safety_projection(final_info, raw_safety)
    if reset_info.get("safety_signal_source") != "raw_physics":
        raise ValueError("reset safety source must be raw_physics")
    state = backend.previous_state
    config = backend.scene.residual_config
    expected_axial = (
        config.helical_lead_m * state.nut_angle_rad / (2.0 * math.pi)
    )
    helical_error = state.axial_travel_m - expected_axial
    minimum_axial_travel = (
        config.minimum_axial_progress_fraction * expected_axial
    )
    axial_progress_fraction = (
        0.0
        if expected_axial <= 0.0
        else state.axial_travel_m / expected_axial
    )
    axial_progress_gate_passed = bool(
        expected_axial > 0.0
        and state.axial_travel_m >= minimum_axial_travel
    )
    q7_delta = state.q7_position_rad - backend.start_q7
    safety = backend.episode_safety
    physical_success = bool(
        terminated
        and not truncated
        and termination_reason == "success"
        and observation.shape == (OBSERVATION_SIZE,)
        and np.all(np.isfinite(observation))
        and abs(state.nut_angle_rad - config.target_angle_rad)
        <= config.success_angle_tolerance_rad
        and abs(q7_delta + state.nut_angle_rad) <= math.radians(3.0)
        and abs(helical_error)
        <= config.helical_tolerance_m(state.nut_angle_rad)
        and axial_progress_gate_passed
        and loaded_torque_channels(state, config)
        >= config.minimum_loaded_torque_channels
        and safety.max_finger_torque_delta
        <= config.maximum_absolute_finger_torque_nm
        and safety.finite_throughout
        and safety.max_limit_violation <= 0.02
        and raw_safety["passed"]
    )
    return {
        "axial_travel_m": state.axial_travel_m,
        "axial_progress_fraction": axial_progress_fraction,
        "axial_progress_gate_passed": axial_progress_gate_passed,
        "episode": episode,
        "episode_randomization": reset_info.get(
            "episode_randomization"
        ),
        "helical_error_m": helical_error,
        "loaded_channels": loaded_torque_channels(state, config),
        "maximum_finger_torque_nm": safety.max_finger_torque_delta,
        "maximum_joint_limit_violation_rad": safety.max_limit_violation,
        "maximum_joint_speed_rad_s": safety.max_abs_velocity,
        "minimum_axial_progress_fraction": (
            config.minimum_axial_progress_fraction
        ),
        "minimum_axial_travel_m": minimum_axial_travel,
        "nut_angle_degrees": math.degrees(state.nut_angle_rad),
        "passed": physical_success,
        "policy_steps": episode_steps,
        "q7_delta_degrees": math.degrees(q7_delta),
        "raw_safety_failure_reasons": raw_safety[
            "failure_reasons"
        ],
        "raw_safety_passed": raw_safety["passed"],
        "raw_safety_report": raw_safety,
        "reset": reset_info.get("reset"),
        "reset_checkpoint": reset_info.get("reset_checkpoint"),
        "return": episode_return,
        "safety_signal_source": raw_safety["signal_source"],
        "seed": seed,
        "stable_hold_seconds": state.stable_hold_seconds,
        "terminated": bool(terminated),
        "termination_reason": termination_reason or "time_limit",
        "truncated": bool(truncated),
    }


def _validate_evaluation_report(report: Any) -> dict[str, Any]:
    """Normalize the fields used by formal aggregate gates."""

    if not isinstance(report, Mapping):
        raise ValueError("evaluation report must be a mapping")
    passed = report.get("passed")
    raw_safety_passed = report.get("raw_safety_passed")
    if type(passed) is not bool:
        raise ValueError("evaluation report passed must be a boolean")
    if type(raw_safety_passed) is not bool:
        raise ValueError(
            "evaluation report raw_safety_passed must be a boolean"
        )
    safety_source = report.get("safety_signal_source")
    if safety_source != "raw_physics":
        raise ValueError(
            "evaluation report safety source must be raw_physics"
        )
    failure_reasons = report.get("raw_safety_failure_reasons")
    if not isinstance(failure_reasons, list) or any(
        type(reason) is not str or not reason
        for reason in failure_reasons
    ):
        raise ValueError(
            "evaluation raw safety reasons must be a list of strings"
        )
    if len(failure_reasons) != len(set(failure_reasons)):
        raise ValueError(
            "evaluation raw safety reasons must be unique"
        )
    if raw_safety_passed != (not failure_reasons):
        raise ValueError(
            "evaluation raw safety flag and reasons are inconsistent"
        )
    if passed and not raw_safety_passed:
        raise ValueError(
            "an unsafe evaluation report cannot be marked passed"
        )
    policy_steps = report.get("policy_steps")
    seed = report.get("seed")
    if (
        isinstance(policy_steps, (bool, np.bool_))
        or not isinstance(policy_steps, Integral)
        or int(policy_steps) <= 0
    ):
        raise ValueError(
            "evaluation report policy_steps must be a positive integer"
        )
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, Integral)
        or int(seed) < 0
    ):
        raise ValueError(
            "evaluation report seed must be a nonnegative integer"
        )
    episode_return = report.get("return")
    if (
        isinstance(episode_return, (bool, np.bool_))
        or not isinstance(episode_return, Real)
        or not math.isfinite(float(episode_return))
    ):
        raise ValueError("evaluation report return must be finite")
    termination_reason = report.get("termination_reason")
    if type(termination_reason) is not str or not termination_reason:
        raise ValueError(
            "evaluation termination reason must be a nonempty string"
        )
    if passed and termination_reason != "success":
        raise ValueError(
            "a passed evaluation report must terminate with success"
        )
    return {
        "passed": passed,
        "policy_steps": int(policy_steps),
        "raw_safety_failure_reasons": list(failure_reasons),
        "raw_safety_passed": raw_safety_passed,
        "return": float(episode_return),
        "safety_signal_source": safety_source,
        "seed": int(seed),
        "termination_reason": termination_reason,
    }


def aggregate_evaluation_reports(
    reports: list[Mapping[str, Any]],
    minimum_success_rate: float,
    maximum_safety_failures: int = 0,
) -> dict[str, Any]:
    """Aggregate deterministic episode reports without simulator imports."""

    if not reports:
        raise ValueError("evaluation requires at least one episode report")
    normalized_reports = [
        _validate_evaluation_report(report) for report in reports
    ]
    successes = sum(report["passed"] for report in normalized_reports)
    reasons = Counter(
        str(report["termination_reason"])
        for report in normalized_reports
        if not report["passed"]
    )
    raw_safety_failures = sum(
        not report["raw_safety_passed"]
        for report in normalized_reports
    )
    termination_safety_failures = sum(
        report["termination_reason"] in SAFETY_FAILURE_REASONS
        for report in normalized_reports
    )
    safety_failures = sum(
        not report["raw_safety_passed"]
        or report["termination_reason"] in SAFETY_FAILURE_REASONS
        for report in normalized_reports
    )
    success_rate = successes / len(normalized_reports)
    return {
        "acceptance_passed": bool(
            success_rate >= minimum_success_rate
            and safety_failures <= maximum_safety_failures
        ),
        "episodes": len(normalized_reports),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "mean_episode_length": float(
            np.mean(
                [report["policy_steps"] for report in normalized_reports]
            )
        ),
        "mean_episode_return": float(
            np.mean([report["return"] for report in normalized_reports])
        ),
        "minimum_success_rate": minimum_success_rate,
        "maximum_safety_failures": maximum_safety_failures,
        "raw_safety_failure_count": raw_safety_failures,
        "safety_failure_count": safety_failures,
        "success_count": successes,
        "success_rate": success_rate,
        "termination_safety_failure_count": (
            termination_safety_failures
        ),
    }


def comparable_episode_randomization(value: Any) -> dict[str, Any] | None:
    """Normalize one sampled domain while ignoring only its episode counter."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("episode randomization must be a mapping or None")
    try:
        normalized = json.loads(
            json.dumps(dict(value), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "episode randomization must contain finite JSON data"
        ) from error
    normalized.pop("episode", None)
    return normalized


def normalized_reset_initial_signature(value: Any) -> dict[str, Any]:
    """Copy the minimal physical reset signature into strict JSON data."""

    if not isinstance(value, Mapping) or set(value) != {
        "body_position",
        "nut_position",
        "q7",
    }:
        raise ValueError("reset initial signature has an invalid schema")
    result: dict[str, Any] = {}
    for field in ("body_position", "nut_position"):
        array = np.asarray(value[field], dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"reset initial signature {field} must be finite 3D"
            )
        result[field] = [float(item) for item in array]
    q7 = value["q7"]
    if (
        isinstance(q7, (bool, np.bool_))
        or not isinstance(q7, Real)
        or not math.isfinite(float(q7))
    ):
        raise ValueError("reset initial signature q7 must be finite")
    result["q7"] = float(q7)
    return result


def compare_reset_initial_signatures(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare paired reset states with explicit physical tolerances."""

    first_normalized = normalized_reset_initial_signature(first)
    second_normalized = normalized_reset_initial_signature(second)
    body_error = float(
        np.linalg.norm(
            np.asarray(first_normalized["body_position"])
            - np.asarray(second_normalized["body_position"])
        )
    )
    nut_error = float(
        np.linalg.norm(
            np.asarray(first_normalized["nut_position"])
            - np.asarray(second_normalized["nut_position"])
        )
    )
    q7_error = abs(first_normalized["q7"] - second_normalized["q7"])
    return {
        "body_position_error_m": body_error,
        "maximum_body_position_error_m": 0.0001,
        "maximum_nut_position_error_m": 0.0001,
        "maximum_q7_error_rad": math.radians(0.1),
        "nut_position_error_m": nut_error,
        "passed": bool(
            body_error <= 0.0001
            and nut_error <= 0.0001
            and q7_error <= math.radians(0.1)
        ),
        "q7_error_rad": q7_error,
    }


def paired_execution_order(
    pair_index: int,
) -> tuple[tuple[str, bool], tuple[str, bool]]:
    """Counterbalance zero/model order using a one-based pair index."""

    if (
        isinstance(pair_index, bool)
        or not isinstance(pair_index, Integral)
        or pair_index < 1
    ):
        raise ValueError("pair index must be a positive integer")
    zero = ("zero", False)
    trained = ("trained_deterministic", True)
    if int(pair_index) % 2 == 1:
        return zero, trained
    return trained, zero


def _binomial_upper_tail_probability(
    trials: int, minimum_successes: int, probability: float
) -> float:
    """Return P[X >= minimum_successes] for a binomial random variable."""

    if (
        isinstance(trials, bool)
        or not isinstance(trials, Integral)
        or trials < 0
    ):
        raise ValueError("binomial trials must be a nonnegative integer")
    if (
        isinstance(minimum_successes, bool)
        or not isinstance(minimum_successes, Integral)
    ):
        raise ValueError("binomial successes must be an integer")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("binomial probability must be finite and in [0, 1]")
    trial_count = int(trials)
    threshold = int(minimum_successes)
    chance = float(probability)
    if threshold <= 0:
        return 1.0
    if threshold > trial_count:
        return 0.0
    if chance == 0.0:
        return 0.0
    if chance == 1.0:
        return 1.0

    log_chance = math.log(chance)
    log_failure = math.log1p(-chance)
    terms = []
    for successes in range(threshold, trial_count + 1):
        terms.append(
            math.lgamma(trial_count + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trial_count - successes + 1)
            + successes * log_chance
            + (trial_count - successes) * log_failure
        )
    maximum = max(terms)
    probability_sum = math.exp(maximum) * math.fsum(
        math.exp(term - maximum) for term in terms
    )
    return float(min(1.0, max(0.0, probability_sum)))


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    *,
    confidence_level: float = POSITIVE_CLAIM_CONFIDENCE_LEVEL,
) -> float:
    """Compute the exact one-sided binomial lower confidence bound."""

    if (
        isinstance(trials, bool)
        or not isinstance(trials, Integral)
        or trials <= 0
    ):
        raise ValueError("Clopper-Pearson trials must be a positive integer")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, Integral)
        or not 0 <= successes <= trials
    ):
        raise ValueError("Clopper-Pearson successes must be in [0, trials]")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence level must be finite and in (0, 1)")
    success_count = int(successes)
    trial_count = int(trials)
    if success_count == 0:
        return 0.0
    alpha = 1.0 - float(confidence_level)
    lower = 0.0
    upper = success_count / trial_count
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        tail = _binomial_upper_tail_probability(
            trial_count, success_count, midpoint
        )
        if tail < alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def exact_mcnemar_one_sided_p_value(
    improvements: int, regressions: int
) -> float:
    """Test whether paired binary improvements exceed regressions."""

    for value, name in (
        (improvements, "improvements"),
        (regressions, "regressions"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
        ):
            raise ValueError(f"paired {name} must be a nonnegative integer")
    discordant = int(improvements) + int(regressions)
    if discordant == 0:
        return 1.0
    return _binomial_upper_tail_probability(
        discordant, int(improvements), 0.5
    )


def positive_claim_training_evidence(
    training_metadata: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Validate that a candidate actor actually received optimizer updates.

    This is deliberately non-throwing: missing, malformed or non-finite
    metadata must make a positive policy-improvement claim false.
    """

    failures: list[str] = []
    minimum_optimizer_updates = None
    if not isinstance(training_metadata, Mapping):
        failures.append("training_metadata_not_mapping")
        return {
            "minimum_actor_parameter_delta": (
                MINIMUM_ACTOR_PARAMETER_DELTA
            ),
            "minimum_required_optimizer_updates": None,
            "policy_improvement_training_evidence_failures": failures,
            "policy_improvement_training_evidence_verified": False,
        }

    for field in (
        "actor_reload_verified",
        "training_completed",
        "passed",
        "optimization_expected",
        "optimization_verified",
        "training_raw_safety_passed",
    ):
        if training_metadata.get(field) is not True:
            failures.append(f"{field}_not_true")

    training_raw_safety_report = None
    try:
        training_raw_safety_report = (
            validate_training_raw_safety_report(
                training_metadata.get("training_raw_safety_report")
            )
        )
    except (TypeError, ValueError):
        failures.append("training_raw_safety_report_invalid")
    if training_raw_safety_report is not None:
        if training_raw_safety_report["passed"] is not True:
            failures.append("training_raw_safety_report_not_passed")
        projections = (
            (
                "training_raw_safety_passed",
                training_raw_safety_report["passed"],
            ),
            (
                "training_raw_safety_failure_reasons",
                training_raw_safety_report["failure_reasons"],
            ),
            (
                "training_raw_safety_peaks",
                training_raw_safety_report["peaks"],
            ),
            (
                "training_raw_safety_policy_steps_audited",
                training_raw_safety_report["policy_steps_audited"],
            ),
            (
                "training_raw_safety_complete_episode_count",
                training_raw_safety_report["complete_episode_count"],
            ),
            (
                "training_raw_safety_partial_episode_count",
                training_raw_safety_report["partial_episode_count"],
            ),
        )
        for field, expected in projections:
            if training_metadata.get(field) != expected:
                failures.append(f"{field}_projection_mismatch")

    def strict_nonnegative_integer(field: str) -> int | None:
        value = training_metadata.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            failures.append(f"{field}_invalid")
            return None
        return int(value)

    model_timesteps = strict_nonnegative_integer("model_timesteps")
    learning_starts = strict_nonnegative_integer("learning_starts")
    optimizer_updates = strict_nonnegative_integer("optimizer_updates")
    if (
        model_timesteps is not None
        and training_raw_safety_report is not None
        and training_raw_safety_report["policy_steps_audited"]
        != model_timesteps
    ):
        failures.append("training_raw_safety_policy_steps_mismatch")
    training_config = training_metadata.get("training_config")
    train_freq_steps = None
    gradient_steps = None
    if not isinstance(training_config, Mapping):
        failures.append("training_config_invalid")
    else:
        for field in ("train_freq_steps", "gradient_steps"):
            value = training_config.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
            ):
                failures.append(f"training_config_{field}_invalid")
            elif field == "train_freq_steps":
                train_freq_steps = int(value)
            else:
                gradient_steps = int(value)
        configured_learning_starts = training_config.get("learning_starts")
        if (
            learning_starts is None
            or isinstance(configured_learning_starts, bool)
            or not isinstance(configured_learning_starts, Integral)
            or int(configured_learning_starts) != learning_starts
        ):
            failures.append("training_config_learning_starts_mismatch")

    if model_timesteps is not None and learning_starts is not None:
        if model_timesteps <= learning_starts:
            failures.append("model_timesteps_not_beyond_learning_starts")
        elif train_freq_steps is not None and gradient_steps is not None:
            eligible_steps = model_timesteps - learning_starts
            minimum_optimizer_updates = max(
                1,
                (eligible_steps // train_freq_steps) * gradient_steps - 1,
            )
            if (
                optimizer_updates is None
                or optimizer_updates < minimum_optimizer_updates
            ):
                failures.append("optimizer_updates_below_minimum")

    actor_delta = training_metadata.get("actor_parameter_max_delta")
    if (
        isinstance(actor_delta, bool)
        or not isinstance(actor_delta, Real)
        or not math.isfinite(float(actor_delta))
        or float(actor_delta) < MINIMUM_ACTOR_PARAMETER_DELTA
    ):
        failures.append("actor_parameter_max_delta_invalid")

    initial_hash = training_metadata.get("actor_initial_state_sha256")
    final_hash = training_metadata.get("actor_final_state_sha256")
    reloaded_hash = training_metadata.get(
        "reloaded_actor_state_sha256"
    )
    if not _valid_sha256(initial_hash):
        failures.append("actor_initial_state_sha256_invalid")
    if not _valid_sha256(final_hash):
        failures.append("actor_final_state_sha256_invalid")
    if not _valid_sha256(reloaded_hash):
        failures.append("reloaded_actor_state_sha256_invalid")
    if _valid_sha256(initial_hash) and initial_hash == final_hash:
        failures.append("actor_state_sha256_unchanged")
    if (
        _valid_sha256(final_hash)
        and _valid_sha256(reloaded_hash)
        and final_hash != reloaded_hash
    ):
        failures.append("reloaded_actor_state_sha256_mismatch")

    return {
        "minimum_actor_parameter_delta": MINIMUM_ACTOR_PARAMETER_DELTA,
        "minimum_required_optimizer_updates": minimum_optimizer_updates,
        "policy_improvement_training_evidence_failures": failures,
        "policy_improvement_training_evidence_verified": not failures,
    }


def aggregate_paired_evaluation_reports(
    zero_reports: list[Mapping[str, Any]],
    trained_reports: list[Mapping[str, Any]],
    randomization_matches: list[bool],
    *,
    minimum_trained_success_rate: float = 0.95,
    minimum_improvement_margin: float = 0.10,
    maximum_safety_failures: int = 0,
    training_evidence_verified: bool = False,
    minimum_paired_episodes: int = MINIMUM_PAIRED_EPISODES_FOR_CLAIM,
    confidence_level: float = POSITIVE_CLAIM_CONFIDENCE_LEVEL,
    maximum_paired_p_value: float = MAXIMUM_PAIRED_P_VALUE,
) -> dict[str, Any]:
    """Compare zero and trained policies.

    Engineering integrity and evidence of policy improvement remain separate.
    """

    if not zero_reports or len(zero_reports) != len(trained_reports):
        raise ValueError(
            "paired evaluation requires equal nonempty report lists"
        )
    if len(randomization_matches) != len(zero_reports):
        raise ValueError(
            "paired evaluation needs one randomization match per pair"
        )
    if any(type(value) is not bool for value in randomization_matches):
        raise ValueError(
            "paired randomization matches must be exact booleans"
        )
    normalized_zero_reports = [
        _validate_evaluation_report(report) for report in zero_reports
    ]
    normalized_trained_reports = [
        _validate_evaluation_report(report) for report in trained_reports
    ]
    if (
        isinstance(minimum_trained_success_rate, bool)
        or not isinstance(minimum_trained_success_rate, Real)
        or not math.isfinite(float(minimum_trained_success_rate))
        or not 0.0 <= float(minimum_trained_success_rate) <= 1.0
    ):
        raise ValueError("minimum trained success rate must be in [0, 1]")
    if (
        isinstance(minimum_improvement_margin, bool)
        or not isinstance(minimum_improvement_margin, Real)
        or not math.isfinite(float(minimum_improvement_margin))
        or not 0.0 <= float(minimum_improvement_margin) <= 1.0
    ):
        raise ValueError("minimum improvement margin must be in [0, 1]")
    if (
        isinstance(maximum_safety_failures, bool)
        or not isinstance(maximum_safety_failures, Integral)
        or maximum_safety_failures < 0
    ):
        raise ValueError("maximum safety failures must be nonnegative")
    if (
        isinstance(minimum_paired_episodes, bool)
        or not isinstance(minimum_paired_episodes, Integral)
        or minimum_paired_episodes < 1
    ):
        raise ValueError("minimum paired episodes must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence level must be finite and in (0, 1)")
    if (
        isinstance(maximum_paired_p_value, bool)
        or not isinstance(maximum_paired_p_value, Real)
        or not math.isfinite(float(maximum_paired_p_value))
        or not 0.0 < float(maximum_paired_p_value) <= 1.0
    ):
        raise ValueError(
            "maximum paired p-value must be finite and in (0, 1]"
        )
    minimum_trained_success_rate = max(
        float(minimum_trained_success_rate),
        MINIMUM_POSITIVE_CLAIM_SUCCESS_RATE,
    )
    minimum_improvement_margin = max(
        float(minimum_improvement_margin),
        MINIMUM_POSITIVE_CLAIM_IMPROVEMENT,
    )
    minimum_paired_episodes = max(
        int(minimum_paired_episodes), MINIMUM_PAIRED_EPISODES_FOR_CLAIM
    )
    confidence_level = max(
        float(confidence_level), POSITIVE_CLAIM_CONFIDENCE_LEVEL
    )
    maximum_paired_p_value = min(
        float(maximum_paired_p_value), MAXIMUM_PAIRED_P_VALUE
    )

    zero_summary = aggregate_evaluation_reports(
        normalized_zero_reports,
        minimum_trained_success_rate,
        maximum_safety_failures,
    )
    trained_summary = aggregate_evaluation_reports(
        normalized_trained_reports,
        minimum_trained_success_rate,
        maximum_safety_failures,
    )
    improvements = 0
    regressions = []
    paired_seed_mismatches = []
    for index, (zero_report, trained_report) in enumerate(
        zip(normalized_zero_reports, normalized_trained_reports), start=1
    ):
        zero_passed = zero_report["passed"]
        trained_passed = trained_report["passed"]
        if zero_report["seed"] != trained_report["seed"]:
            paired_seed_mismatches.append(
                {
                    "episode": index,
                    "trained_seed": trained_report["seed"],
                    "zero_seed": zero_report["seed"],
                }
            )
        if not zero_passed and trained_passed:
            improvements += 1
        if zero_passed and not trained_passed:
            regressions.append(
                {
                    "episode": index,
                    "seed": trained_report.get("seed"),
                    "trained_termination_reason": trained_report.get(
                        "termination_reason"
                    ),
                    "zero_termination_reason": zero_report.get(
                        "termination_reason"
                    ),
                }
            )
    improvement_margin = (
        trained_summary["success_rate"] - zero_summary["success_rate"]
    )
    paired_data_integrity_passed = bool(
        all(randomization_matches)
        and not paired_seed_mismatches
        and zero_summary["safety_failure_count"]
        <= maximum_safety_failures
        and trained_summary["safety_failure_count"]
        <= maximum_safety_failures
    )
    exact_p_value = exact_mcnemar_one_sided_p_value(
        improvements, len(regressions)
    )
    trained_success_rate_lower_bound = clopper_pearson_lower_bound(
        trained_summary["success_count"],
        len(normalized_trained_reports),
        confidence_level=float(confidence_level),
    )
    paired_statistical_evidence_passed = bool(
        len(normalized_trained_reports) >= int(minimum_paired_episodes)
        and trained_summary["success_rate"]
        >= minimum_trained_success_rate
        and trained_success_rate_lower_bound + 1.0e-12
        >= minimum_trained_success_rate
        and improvement_margin + 1.0e-12
        >= minimum_improvement_margin
        and exact_p_value <= float(maximum_paired_p_value)
    )
    improvement_criteria_passed = bool(
        paired_data_integrity_passed
        and training_evidence_verified is True
        and paired_statistical_evidence_passed
        and zero_summary["safety_failure_count"] == 0
        and trained_summary["safety_failure_count"] == 0
        and not regressions
    )
    return {
        "confidence_level": float(confidence_level),
        "maximum_paired_p_value": float(maximum_paired_p_value),
        "minimum_improvement_margin": minimum_improvement_margin,
        "minimum_paired_episodes": int(minimum_paired_episodes),
        "minimum_trained_success_rate": minimum_trained_success_rate,
        "paired_data_integrity_passed": paired_data_integrity_passed,
        "paired_discordant_count": improvements + len(regressions),
        "paired_exact_mcnemar_one_sided_p_value": exact_p_value,
        "paired_improvement_count": improvements,
        "paired_randomization_match_count": sum(
            randomization_matches
        ),
        "paired_randomization_mismatch_count": sum(
            not value for value in randomization_matches
        ),
        "paired_regression_count": len(regressions),
        "paired_regressions": regressions,
        "paired_seed_mismatch_count": len(paired_seed_mismatches),
        "paired_seed_mismatches": paired_seed_mismatches,
        "paired_statistical_evidence_passed": (
            paired_statistical_evidence_passed
        ),
        "policy_improvement_criteria_passed": (
            improvement_criteria_passed
        ),
        "policy_improvement_training_evidence_verified": (
            training_evidence_verified is True
        ),
        "success_rate_improvement": improvement_margin,
        "trained_success_rate_clopper_pearson_lower_bound": (
            trained_success_rate_lower_bound
        ),
        "trained_policy_summary": trained_summary,
        "zero_policy_summary": zero_summary,
    }


def run_formal_evaluation(
    environment: Any,
    backend: Any,
    config: ConnectorResidualSACConfig,
    *,
    model_path: str | Path,
    episodes: int | None,
    output_root: str | Path,
    provenance_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Evaluate one formal model with deterministic actions and fixed seeds."""

    modules = _runtime_modules(config)
    initial_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    model = Path(model_path).expanduser().resolve()
    metadata_path, training_metadata = load_training_metadata_for_model(model)
    runtime_metadata = _runtime_metadata(modules)
    validate_evaluation_runtime(training_metadata, runtime_metadata)
    current_provenance, current_source_bytes = capture_provenance_snapshot(
        provenance_paths
    )
    validate_curriculum_provenance(current_provenance)
    if "curriculum_config" not in current_source_bytes:
        raise ValueError(
            "formal evaluation did not freeze the curriculum YAML bytes"
        )
    validate_evaluation_provenance(training_metadata, current_provenance)
    initial_randomization_metadata = backend_randomization_metadata(backend)
    validate_randomization_provenance(
        initial_randomization_metadata, current_provenance
    )
    trained_randomization_enabled = bool(
        training_metadata.get("randomization_enabled", False)
    )
    if trained_randomization_enabled != bool(
        initial_randomization_metadata["randomization_enabled"]
    ):
        raise ValueError(
            "evaluation randomization mode differs from training metadata"
        )
    resolved_randomization_document = (
        resolved_backend_randomization_document(backend)
    )
    validate_resolved_randomization_snapshot(
        training_metadata, resolved_randomization_document
    )
    resolved_curriculum_document = (
        resolved_backend_curriculum_document(backend)
    )
    validate_resolved_curriculum_snapshot(
        training_metadata, resolved_curriculum_document
    )
    resolved_randomization_payload = _resolved_randomization_yaml(
        resolved_randomization_document
    )
    resolved_randomization_sha256 = (
        None
        if resolved_randomization_payload is None
        else hashlib.sha256(resolved_randomization_payload).hexdigest()
    )
    episode_count = (
        config.evaluation_episodes
        if episodes is None
        else _positive_int(episodes, "evaluation episodes")
    )
    run_directory = _new_run_directory(
        output_root, "evaluate", config.evaluation_seed_start
    )
    curriculum_snapshot_metadata = _archive_curriculum_snapshot(
        run_directory,
        current_source_bytes["curriculum_config"],
        resolved_curriculum_document,
    )
    resolved_randomization_path = None
    if resolved_randomization_payload is not None:
        resolved_randomization_path = (
            run_directory / "resolved_randomization_config.yaml"
        )
        resolved_randomization_path.write_bytes(
            resolved_randomization_payload
        )
    _seed_runtime(environment, modules, config.evaluation_seed_start)
    loaded = modules["SAC"].load(
        str(model), env=environment, device=config.device
    )
    loaded_device_metadata = _model_device_metadata(loaded, "evaluation")
    loaded_actor_state_sha256 = _module_state_sha256(loaded.actor)
    validate_loaded_actor_training_binding(
        training_metadata, config, loaded_actor_state_sha256
    )
    reports: list[dict[str, Any]] = []
    for index in range(episode_count):
        seed = config.evaluation_seed_start + index
        observation, reset_info = environment.reset(seed=seed)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        final_info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = loaded.predict(
                observation, deterministic=config.evaluation_deterministic
            )
            observation, reward, terminated, truncated, final_info = (
                environment.step(np.asarray(action, dtype=np.float32))
            )
            episode_return += float(reward)
            episode_steps += 1
        reports.append(
            physical_episode_report(
                backend,
                observation,
                episode=index + 1,
                episode_return=episode_return,
                episode_steps=episode_steps,
                terminated=terminated,
                truncated=truncated,
                termination_reason=str(
                    final_info.get("termination_reason", "")
                ),
                final_info=final_info,
                reset_info=reset_info,
                seed=seed,
            )
        )

    summary = aggregate_evaluation_reports(
        reports,
        config.minimum_success_rate,
        config.maximum_safety_failures,
    )
    reset_snap_maxima, reset_snap_ok = _backend_reset_summary(backend)
    final_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    randomization_metadata = backend_randomization_metadata(backend)
    summary["acceptance_passed"] = bool(
        summary["acceptance_passed"]
        and reset_snap_ok
        and backend.thread_proxy_rebuild_count == backend.reset_count
        and final_runtime_thread_prim_count
        == initial_runtime_thread_prim_count
    )
    actor_device = str(next(loaded.actor.parameters()).device)
    evaluation_metadata: dict[str, Any] = {
        "action_size": ACTION_SIZE,
        "actor_device": actor_device,
        "deterministic": True,
        "environment_hard_resets": backend.reset_count,
        "episode_reports": reports,
        "evaluation_completed": True,
        "interface_version": config.interface_version,
        "loaded_actor_state_sha256": loaded_actor_state_sha256,
        "model_path": str(model),
        "model_sha256": file_sha256(model),
        "observation_size": OBSERVATION_SIZE,
        "generalization_claim": False,
        "policy_competence_claim": False,
        "policy_seed": training_metadata.get("policy_seed"),
        "random_seed_start": config.evaluation_seed_start,
        "residual_curriculum_stage": (
            resolved_curriculum_document["stage_name"]
        ),
        "resolved_randomization_config": (
            resolved_randomization_document
        ),
        "resolved_randomization_config_path": (
            None
            if resolved_randomization_path is None
            else str(resolved_randomization_path)
        ),
        "resolved_randomization_config_sha256": (
            resolved_randomization_sha256
        ),
        "reset_checkpoint_snap_maxima": reset_snap_maxima,
        "reset_checkpoint_snap_ok": reset_snap_ok,
        "run_directory": str(run_directory),
        "run_kind": "formal_evaluation",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config.schema_version,
        "scene_build_count": 1,
        "simulation_app_count": 1,
        "thread_proxy_rebuild_count": backend.thread_proxy_rebuild_count,
        "thread_proxy_reset_strategy": (
            "remove_hard_reset_contact_recovery_recreate"
        ),
        "runtime_thread_prim_count": final_runtime_thread_prim_count,
        "runtime_thread_prim_count_initial": (
            initial_runtime_thread_prim_count
        ),
        "physicsusd_disjoint_warning_count": None,
        "physicsusd_disjoint_warning_count_source": (
            "not_captured_in_process; inspect the matching Kit log"
        ),
        "training_metadata_path": str(metadata_path),
        "training_metadata_sha256": file_sha256(metadata_path),
        "vecnormalize_path": None,
        "vecnormalize_used": False,
    }
    evaluation_metadata.update(summary)
    evaluation_metadata.update(randomization_metadata)
    evaluation_metadata.update(loaded_device_metadata)
    evaluation_metadata.update(runtime_metadata)
    evaluation_metadata.update(current_provenance)
    evaluation_metadata.update(curriculum_snapshot_metadata)
    output_path = run_directory / "evaluation_metadata.json"
    output_path.write_text(
        json.dumps(
            evaluation_metadata, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_metadata["metadata_path"] = str(output_path)
    environment.close()
    return evaluation_metadata


def run_formal_paired_evaluation(
    environment: Any,
    backend: Any,
    config: ConnectorResidualSACConfig,
    *,
    model_path: str | Path,
    episodes: int | None,
    output_root: str | Path,
    provenance_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Benchmark zero versus deterministic model actions on paired domains."""

    modules = _runtime_modules(config)
    initial_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    initial_reset_count = int(backend.reset_count)
    initial_thread_rebuild_count = int(backend.thread_proxy_rebuild_count)
    initial_history_count = len(backend.episode_randomization_history)
    model = Path(model_path).expanduser().resolve()
    metadata_path, training_metadata = load_training_metadata_for_model(model)
    training_evidence = positive_claim_training_evidence(training_metadata)
    runtime_metadata = _runtime_metadata(modules)
    validate_evaluation_runtime(training_metadata, runtime_metadata)
    current_provenance, current_source_bytes = capture_provenance_snapshot(
        provenance_paths
    )
    validate_curriculum_provenance(current_provenance)
    if "curriculum_config" not in current_source_bytes:
        raise ValueError(
            "paired evaluation did not freeze the curriculum YAML bytes"
        )
    validate_evaluation_provenance(training_metadata, current_provenance)
    initial_randomization_metadata = backend_randomization_metadata(backend)
    validate_randomization_provenance(
        initial_randomization_metadata, current_provenance
    )
    trained_randomization_enabled = bool(
        training_metadata.get("randomization_enabled", False)
    )
    if trained_randomization_enabled != bool(
        initial_randomization_metadata["randomization_enabled"]
    ):
        raise ValueError(
            "paired evaluation randomization mode differs from training"
        )
    resolved_randomization_document = (
        resolved_backend_randomization_document(backend)
    )
    validate_resolved_randomization_snapshot(
        training_metadata, resolved_randomization_document
    )
    resolved_curriculum_document = (
        resolved_backend_curriculum_document(backend)
    )
    validate_resolved_curriculum_snapshot(
        training_metadata, resolved_curriculum_document
    )
    resolved_randomization_payload = _resolved_randomization_yaml(
        resolved_randomization_document
    )
    resolved_randomization_sha256 = (
        None
        if resolved_randomization_payload is None
        else hashlib.sha256(resolved_randomization_payload).hexdigest()
    )
    episode_count = (
        config.evaluation_episodes
        if episodes is None
        else _positive_int(episodes, "evaluation episodes")
    )
    run_directory = _new_run_directory(
        output_root, "paired_evaluate", config.evaluation_seed_start
    )
    curriculum_snapshot_metadata = _archive_curriculum_snapshot(
        run_directory,
        current_source_bytes["curriculum_config"],
        resolved_curriculum_document,
    )
    resolved_randomization_path = None
    if resolved_randomization_payload is not None:
        resolved_randomization_path = (
            run_directory / "resolved_randomization_config.yaml"
        )
        resolved_randomization_path.write_bytes(
            resolved_randomization_payload
        )

    _seed_runtime(environment, modules, config.evaluation_seed_start)
    loaded = modules["SAC"].load(
        str(model), env=environment, device=config.device
    )
    loaded_device_metadata = _model_device_metadata(loaded, "evaluation")
    loaded_actor_state_sha256 = _module_state_sha256(loaded.actor)
    validate_loaded_actor_training_binding(
        training_metadata, config, loaded_actor_state_sha256
    )
    zero_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def run_episode(
        *, seed: int, episode: int, use_model: bool
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        observation, reset_info = environment.reset(seed=seed)
        reset_initial_signature = normalized_reset_initial_signature(
            backend.initial_signature
        )
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        final_info: dict[str, Any] = {}
        while not (terminated or truncated):
            if use_model:
                action, _ = loaded.predict(
                    observation,
                    deterministic=config.evaluation_deterministic,
                )
                selected_action = np.asarray(action, dtype=np.float32)
            else:
                selected_action = zero_action
            observation, reward, terminated, truncated, final_info = (
                environment.step(selected_action)
            )
            episode_return += float(reward)
            episode_steps += 1
        report = physical_episode_report(
            backend,
            observation,
            episode=episode,
            episode_return=episode_return,
            episode_steps=episode_steps,
            terminated=terminated,
            truncated=truncated,
            termination_reason=str(
                final_info.get("termination_reason", "")
            ),
            final_info=final_info,
            reset_info=reset_info,
            seed=seed,
        )
        report["reset_initial_signature"] = reset_initial_signature
        return report, reset_info

    zero_reports: list[dict[str, Any]] = []
    trained_reports: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    randomization_matches: list[bool] = []
    seed_pairs_verified: list[bool] = []
    reset_initial_signature_matches: list[bool] = []
    execution_order_counts = {
        "zero_then_trained_deterministic": 0,
        "trained_deterministic_then_zero": 0,
    }
    for index in range(episode_count):
        pair_index = index + 1
        seed = config.evaluation_seed_start + index
        execution_order = paired_execution_order(pair_index)
        execution_labels = [label for label, _ in execution_order]
        execution_order_key = "_then_".join(execution_labels)
        execution_order_counts[execution_order_key] += 1
        executed_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for label, use_model in execution_order:
            executed_reports[label] = run_episode(
                seed=seed,
                episode=pair_index,
                use_model=use_model,
            )
        zero_report, zero_reset_info = executed_reports["zero"]
        trained_report, trained_reset_info = executed_reports[
            "trained_deterministic"
        ]
        zero_randomization = comparable_episode_randomization(
            zero_reset_info.get("episode_randomization")
        )
        trained_randomization = comparable_episode_randomization(
            trained_reset_info.get("episode_randomization")
        )
        if trained_randomization_enabled:
            randomization_records_present = bool(
                isinstance(
                    zero_reset_info.get("episode_randomization"), Mapping
                )
                and isinstance(
                    trained_reset_info.get("episode_randomization"),
                    Mapping,
                )
            )
        else:
            randomization_records_present = bool(
                zero_reset_info.get("episode_randomization") is None
                and trained_reset_info.get("episode_randomization") is None
            )
        randomization_match = bool(
            randomization_records_present
            and zero_randomization == trained_randomization
        )
        seed_pair_verified = bool(
            zero_reset_info.get("seed") == seed
            and trained_reset_info.get("seed") == seed
        )
        reset_signature_comparison = compare_reset_initial_signatures(
            zero_report["reset_initial_signature"],
            trained_report["reset_initial_signature"],
        )
        randomization_matches.append(randomization_match)
        seed_pairs_verified.append(seed_pair_verified)
        reset_initial_signature_matches.append(
            reset_signature_comparison["passed"]
        )
        zero_reports.append(zero_report)
        trained_reports.append(trained_report)
        pair_reports.append(
            {
                "episode": pair_index,
                "execution_order": execution_labels,
                "randomization_match": randomization_match,
                "reset_initial_signature_comparison": (
                    reset_signature_comparison
                ),
                "seed": seed,
                "seed_pair_verified": seed_pair_verified,
                "trained": trained_report,
                "trained_episode_randomization_comparable": (
                    trained_randomization
                ),
                "zero": zero_report,
                "zero_episode_randomization_comparable": (
                    zero_randomization
                ),
            }
        )

    paired_summary = aggregate_paired_evaluation_reports(
        zero_reports,
        trained_reports,
        randomization_matches,
        minimum_trained_success_rate=config.minimum_success_rate,
        minimum_improvement_margin=0.10,
        maximum_safety_failures=config.maximum_safety_failures,
        training_evidence_verified=training_evidence[
            "policy_improvement_training_evidence_verified"
        ],
    )
    reset_snap_maxima, reset_snap_ok = _backend_reset_summary(backend)
    final_runtime_thread_prim_count = _runtime_thread_prim_count(backend)
    evaluation_reset_count = int(backend.reset_count) - initial_reset_count
    evaluation_thread_rebuild_count = (
        int(backend.thread_proxy_rebuild_count)
        - initial_thread_rebuild_count
    )
    randomization_metadata = backend_randomization_metadata(
        backend, history_start=initial_history_count
    )
    expected_randomization_count = (
        2 * episode_count if backend.randomization_enabled else 0
    )
    paired_history_verified = bool(
        randomization_metadata["episode_randomization_count"]
        == expected_randomization_count
    )
    raw_safety_reports = all(
        report.get("safety_signal_source") == "raw_physics"
        and report.get("raw_safety_passed") is True
        for report in zero_reports + trained_reports
    )
    execution_order_verified = bool(
        execution_order_counts["zero_then_trained_deterministic"]
        == (episode_count + 1) // 2
        and execution_order_counts[
            "trained_deterministic_then_zero"
        ]
        == episode_count // 2
    )
    benchmark_integrity_passed = bool(
        paired_summary["paired_data_integrity_passed"]
        and all(seed_pairs_verified)
        and all(reset_initial_signature_matches)
        and execution_order_verified
        and paired_history_verified
        and raw_safety_reports
        and evaluation_reset_count == 2 * episode_count
        and evaluation_thread_rebuild_count == evaluation_reset_count
        and backend.thread_proxy_rebuild_count == backend.reset_count
        and reset_snap_ok
        and final_runtime_thread_prim_count
        == initial_runtime_thread_prim_count
    )
    policy_improvement_claim = bool(
        benchmark_integrity_passed
        and paired_summary["policy_improvement_criteria_passed"]
    )
    actor_device = str(next(loaded.actor.parameters()).device)
    paired_metadata: dict[str, Any] = {
        "action_size": ACTION_SIZE,
        "actor_device": actor_device,
        "benchmark_integrity_passed": benchmark_integrity_passed,
        "deterministic_model_actions": True,
        "environment_hard_resets": backend.reset_count,
        "evaluation_completed": True,
        "evaluation_order_counts": execution_order_counts,
        "evaluation_order_strategy": (
            "counterbalanced_one_based_pair_index_odd_zero_first_"
            "even_trained_deterministic_first"
        ),
        "evaluation_order_verified": execution_order_verified,
        "evaluation_pair_reset_count": evaluation_reset_count,
        "evaluation_thread_proxy_rebuild_count": (
            evaluation_thread_rebuild_count
        ),
        "generalization_claim": False,
        "interface_version": config.interface_version,
        "loaded_actor_state_sha256": loaded_actor_state_sha256,
        "model_path": str(model),
        "model_sha256": file_sha256(model),
        "observation_size": OBSERVATION_SIZE,
        "pair_reports": pair_reports,
        "paired_history_verified": paired_history_verified,
        "paired_seed_stream_verified": all(seed_pairs_verified),
        "paired_reset_initial_signature_match_count": sum(
            reset_initial_signature_matches
        ),
        "paired_reset_initial_signature_mismatch_count": sum(
            not value for value in reset_initial_signature_matches
        ),
        "paired_reset_initial_signatures_verified": all(
            reset_initial_signature_matches
        ),
        "passed": benchmark_integrity_passed,
        "policy_competence_claim": False,
        "policy_improvement_claim": policy_improvement_claim,
        "policy_seed": training_metadata.get("policy_seed"),
        "provenance_verified": True,
        "random_seed_start": config.evaluation_seed_start,
        "residual_curriculum_stage": (
            resolved_curriculum_document["stage_name"]
        ),
        "reset_checkpoint_snap_maxima": reset_snap_maxima,
        "reset_checkpoint_snap_ok": reset_snap_ok,
        "resolved_randomization_config": (
            resolved_randomization_document
        ),
        "resolved_randomization_config_path": (
            None
            if resolved_randomization_path is None
            else str(resolved_randomization_path)
        ),
        "resolved_randomization_config_sha256": (
            resolved_randomization_sha256
        ),
        "run_directory": str(run_directory),
        "run_kind": "formal_paired_zero_vs_model_evaluation",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config.schema_version,
        "scene_build_count": 1,
        "simulation_app_count": 1,
        "training_metadata_path": str(metadata_path),
        "training_metadata_sha256": file_sha256(metadata_path),
        "vecnormalize_path": None,
        "vecnormalize_used": False,
    }
    paired_metadata.update(paired_summary)
    paired_metadata.update(randomization_metadata)
    paired_metadata.update(loaded_device_metadata)
    paired_metadata.update(runtime_metadata)
    paired_metadata.update(current_provenance)
    paired_metadata.update(curriculum_snapshot_metadata)
    paired_metadata.update(training_evidence)
    output_path = run_directory / "paired_evaluation_metadata.json"
    output_path.write_text(
        json.dumps(
            paired_metadata, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    paired_metadata["metadata_path"] = str(output_path)
    environment.close()
    return paired_metadata


__all__ = [
    "ACTION_SIZE",
    "EVALUATION_RUNTIME_COMPATIBILITY_FIELDS",
    "INTERFACE_VERSION",
    "MAXIMUM_PAIRED_P_VALUE",
    "MINIMUM_ACTOR_PARAMETER_DELTA",
    "MINIMUM_PAIRED_EPISODES_FOR_CLAIM",
    "MINIMUM_POSITIVE_CLAIM_IMPROVEMENT",
    "MINIMUM_POSITIVE_CLAIM_SUCCESS_RATE",
    "OBSERVATION_SIZE",
    "POSITIVE_CLAIM_CONFIDENCE_LEVEL",
    "RAW_SAFETY_LIMIT_FIELDS",
    "RAW_SAFETY_METRIC_FIELDS",
    "RAW_SAFETY_REPORT_FIELDS",
    "SAFETY_FAILURE_REASONS",
    "SCHEMA_VERSION",
    "TRAINING_RAW_SAFETY_EPISODE_FIELDS",
    "TRAINING_RAW_SAFETY_REPORT_FIELDS",
    "TRAINING_RAW_SAFETY_SCHEMA_VERSION",
    "ConnectorResidualSACConfig",
    "TrainingRawSafetyAudit",
    "aggregate_evaluation_reports",
    "aggregate_paired_evaluation_reports",
    "backend_randomization_metadata",
    "capture_provenance_snapshot",
    "clopper_pearson_lower_bound",
    "compare_reset_initial_signatures",
    "comparable_episode_randomization",
    "exact_mcnemar_one_sided_p_value",
    "file_sha256",
    "load_connector_residual_sac_config",
    "load_training_metadata_for_model",
    "normalized_reset_initial_signature",
    "paired_execution_order",
    "physical_episode_report",
    "positive_claim_training_evidence",
    "provenance_metadata",
    "resolve_training_timesteps",
    "resolved_backend_curriculum_document",
    "resolved_config_document",
    "resolved_backend_randomization_document",
    "run_formal_evaluation",
    "run_formal_paired_evaluation",
    "run_formal_training",
    "state_mapping_sha256",
    "training_randomization_phase_verified",
    "validate_evaluation_provenance",
    "validate_evaluation_runtime",
    "validate_loaded_actor_training_binding",
    "validate_raw_safety_report",
    "validate_training_raw_safety_report",
    "validate_curriculum_provenance",
    "validate_randomization_provenance",
    "validate_resolved_curriculum_snapshot",
    "validate_resolved_randomization_snapshot",
]
