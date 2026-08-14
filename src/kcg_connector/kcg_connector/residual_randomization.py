"""Seeded, low-risk randomization for connector residual v0.

This module is deliberately pure Python/NumPy.  It owns only configuration
validation and deterministic sampling; Isaac-specific application lives in
``isaac_residual_backend``.  In particular, the first randomization contract
does not alter connector mass, contact friction, geometry, or thread lead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .residual_rl import ConnectorResidualConfig


RANDOMIZATION_SCHEMA_VERSION = (
    "kcg_connector_residual_randomization_v1"
)
RESIDUAL_INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
RESIDUAL_ACTION_SIZE = 4
RESIDUAL_OBSERVATION_SIZE = 24
TORQUE_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")


def reproducible_stream_reset_seed(
    base_seed: int | None, episode_index: int
) -> int | None:
    """Seed the first reset, then advance the backend RNG stream naturally."""

    if isinstance(episode_index, bool) or not isinstance(
        episode_index, (int, np.integer)
    ):
        raise TypeError("episode_index must be a nonnegative integer")
    if int(episode_index) < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    if base_seed is None:
        return None
    if isinstance(base_seed, bool) or not isinstance(
        base_seed, (int, np.integer)
    ):
        raise TypeError("base_seed must be a nonnegative integer or None")
    if int(base_seed) < 0:
        raise ValueError("base_seed must be a nonnegative integer or None")
    return int(base_seed) if int(episode_index) == 0 else None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_tuple(value: Any, name: str, size: int) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], name: str):
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} keys differ; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class ConnectorResidualRandomizationConfig:
    """Validated distribution for the already-engaged 20-degree stage."""

    schema_version: str
    interface_version: str
    action_size: int
    observation_size: int
    enabled: bool
    clamp_joint_names: tuple[str, ...]
    clamp_offset_lower_rad: tuple[float, ...]
    clamp_offset_upper_rad: tuple[float, ...]
    hand_pd_joint_names: tuple[str, ...]
    hand_kp_scale_lower: float
    hand_kp_scale_upper: float
    hand_kd_scale_lower: float
    hand_kd_scale_upper: float
    torque_joint_names: tuple[str, ...]
    torque_bias_lower_nm: tuple[float, ...]
    torque_bias_upper_nm: tuple[float, ...]
    torque_noise_std_nm: tuple[float, ...]
    torque_noise_clip_sigma: float
    action_delay_choices: tuple[int, ...]
    safety_signal_source: str
    excluded_physics_parameters: tuple[str, ...]


@dataclass(frozen=True)
class ConnectorResidualEpisodeRandomization:
    """One sampled episode with a JSON-serializable representation."""

    clamp_nominal_offsets_rad: tuple[float, ...]
    hand_kp_scale: float
    hand_kd_scale: float
    finger_torque_bias_nm: tuple[float, ...]
    finger_torque_noise_std_nm: tuple[float, ...]
    finger_torque_noise_clip_sigma: float
    action_delay_policy_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_connector_residual_randomization_config(
    config_path: str | Path,
) -> ConnectorResidualRandomizationConfig:
    """Load and fail-closed validate the versioned v1 distribution."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "configuration")
    _exact_keys(
        document,
        {
            "schema_version",
            "contract",
            "randomization",
            "safety",
            "excluded_physics_parameters",
        },
        "configuration",
    )
    contract = _mapping(document["contract"], "contract")
    _exact_keys(
        contract,
        {"interface_version", "action_size", "observation_size"},
        "contract",
    )
    randomization = _mapping(
        document["randomization"], "randomization"
    )
    _exact_keys(
        randomization,
        {
            "enabled",
            "clamp_nominal_offset",
            "hand_pd_scale",
            "finger_torque_observation",
            "action_delay_policy_steps",
        },
        "randomization",
    )
    clamp = _mapping(
        randomization["clamp_nominal_offset"],
        "randomization.clamp_nominal_offset",
    )
    _exact_keys(
        clamp,
        {"joint_names", "lower_rad", "upper_rad"},
        "randomization.clamp_nominal_offset",
    )
    gains = _mapping(
        randomization["hand_pd_scale"],
        "randomization.hand_pd_scale",
    )
    _exact_keys(
        gains,
        {"joint_names", "kp", "kd"},
        "randomization.hand_pd_scale",
    )
    kp_range = _finite_tuple(
        gains["kp"], "randomization.hand_pd_scale.kp", 2
    )
    kd_range = _finite_tuple(
        gains["kd"], "randomization.hand_pd_scale.kd", 2
    )
    torque = _mapping(
        randomization["finger_torque_observation"],
        "randomization.finger_torque_observation",
    )
    _exact_keys(
        torque,
        {
            "joint_names",
            "bias_lower_nm",
            "bias_upper_nm",
            "gaussian_noise_std_nm",
            "gaussian_noise_clip_sigma",
        },
        "randomization.finger_torque_observation",
    )
    delay = _mapping(
        randomization["action_delay_policy_steps"],
        "randomization.action_delay_policy_steps",
    )
    _exact_keys(
        delay,
        {"choices"},
        "randomization.action_delay_policy_steps",
    )
    safety = _mapping(document["safety"], "safety")
    _exact_keys(safety, {"signal_source"}, "safety")

    try:
        action_size = int(contract["action_size"])
        observation_size = int(contract["observation_size"])
        action_delay_choices = tuple(
            int(value) for value in delay["choices"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "contract sizes and delay choices must be integers"
        ) from error
    config = ConnectorResidualRandomizationConfig(
        schema_version=str(document["schema_version"]),
        interface_version=str(contract["interface_version"]),
        action_size=action_size,
        observation_size=observation_size,
        enabled=randomization["enabled"],
        clamp_joint_names=tuple(str(value) for value in clamp["joint_names"]),
        clamp_offset_lower_rad=_finite_tuple(
            clamp["lower_rad"],
            "randomization.clamp_nominal_offset.lower_rad",
            3,
        ),
        clamp_offset_upper_rad=_finite_tuple(
            clamp["upper_rad"],
            "randomization.clamp_nominal_offset.upper_rad",
            3,
        ),
        hand_pd_joint_names=tuple(
            str(value) for value in gains["joint_names"]
        ),
        hand_kp_scale_lower=kp_range[0],
        hand_kp_scale_upper=kp_range[1],
        hand_kd_scale_lower=kd_range[0],
        hand_kd_scale_upper=kd_range[1],
        torque_joint_names=tuple(
            str(value) for value in torque["joint_names"]
        ),
        torque_bias_lower_nm=_finite_tuple(
            torque["bias_lower_nm"],
            "randomization.finger_torque_observation.bias_lower_nm",
            3,
        ),
        torque_bias_upper_nm=_finite_tuple(
            torque["bias_upper_nm"],
            "randomization.finger_torque_observation.bias_upper_nm",
            3,
        ),
        torque_noise_std_nm=_finite_tuple(
            torque["gaussian_noise_std_nm"],
            "randomization.finger_torque_observation.gaussian_noise_std_nm",
            3,
        ),
        torque_noise_clip_sigma=_finite_float(
            torque["gaussian_noise_clip_sigma"],
            "randomization.finger_torque_observation."
            "gaussian_noise_clip_sigma",
        ),
        action_delay_choices=action_delay_choices,
        safety_signal_source=str(safety["signal_source"]),
        excluded_physics_parameters=tuple(
            str(value) for value in document["excluded_physics_parameters"]
        ),
    )
    _validate_randomization_config(config)
    return config


def _validate_randomization_config(
    config: ConnectorResidualRandomizationConfig,
) -> None:
    if config.schema_version != RANDOMIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported residual randomization schema: "
            f"{config.schema_version!r}"
        )
    if config.interface_version != RESIDUAL_INTERFACE_VERSION:
        raise ValueError("randomization does not target residual v0")
    if config.action_size != RESIDUAL_ACTION_SIZE:
        raise ValueError("randomization action size must remain 4")
    if config.observation_size != RESIDUAL_OBSERVATION_SIZE:
        raise ValueError("randomization observation size must remain 24")
    if not isinstance(config.enabled, bool):
        raise ValueError("randomization.enabled must be boolean")
    if config.clamp_joint_names != TORQUE_JOINT_NAMES:
        raise ValueError("clamp randomization must target f1j2/f2j1/f3j2")
    if config.hand_pd_joint_names != TORQUE_JOINT_NAMES:
        raise ValueError("PD randomization must target f1j2/f2j1/f3j2")
    if config.torque_joint_names != TORQUE_JOINT_NAMES:
        raise ValueError("torque randomization must target f1j2/f2j1/f3j2")

    for lower, upper in zip(
        config.clamp_offset_lower_rad,
        config.clamp_offset_upper_rad,
    ):
        if not lower <= 0.0 <= upper or lower > upper:
            raise ValueError("clamp offsets must bracket zero")
        if max(abs(lower), abs(upper)) > 0.010000001:
            raise ValueError("v1 clamp offsets must stay within +/-0.01 rad")
    for name, lower, upper in (
        (
            "hand kp scale",
            config.hand_kp_scale_lower,
            config.hand_kp_scale_upper,
        ),
        (
            "hand kd scale",
            config.hand_kd_scale_lower,
            config.hand_kd_scale_upper,
        ),
    ):
        if not lower <= 1.0 <= upper or lower > upper:
            raise ValueError(f"{name} must bracket one")
        if lower < 0.95 - 1.0e-12 or upper > 1.05 + 1.0e-12:
            raise ValueError(
                f"{name} must stay within +/-5 percent"
            )
    for lower, upper in zip(
        config.torque_bias_lower_nm,
        config.torque_bias_upper_nm,
    ):
        if not lower <= 0.0 <= upper or lower > upper:
            raise ValueError("torque bias ranges must bracket zero")
        if max(abs(lower), abs(upper)) > 0.010000001:
            raise ValueError("v1 torque bias must stay within +/-0.01 Nm")
    if any(
        value < 0.0 or value > 0.005000001
        for value in config.torque_noise_std_nm
    ):
        raise ValueError("v1 torque noise std must be in [0, 0.005] Nm")
    if not 1.0 <= config.torque_noise_clip_sigma <= 6.0:
        raise ValueError("torque noise clip must be in [1, 6] sigma")
    if config.action_delay_choices != (0, 1):
        raise ValueError("v1 action delay choices must be exactly [0, 1]")
    if config.safety_signal_source != "raw_physics":
        raise ValueError("safety must use raw_physics")
    excluded = set(config.excluded_physics_parameters)
    if excluded != {"mass", "friction", "thread_lead"}:
        raise ValueError(
            "v1 must explicitly exclude mass, friction, and thread_lead"
        )


def sample_connector_residual_randomization(
    config: ConnectorResidualRandomizationConfig,
    rng: np.random.Generator,
) -> ConnectorResidualEpisodeRandomization:
    """Sample one episode from ``rng`` using a fixed, documented draw order."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not config.enabled:
        return ConnectorResidualEpisodeRandomization(
            clamp_nominal_offsets_rad=(0.0, 0.0, 0.0),
            hand_kp_scale=1.0,
            hand_kd_scale=1.0,
            finger_torque_bias_nm=(0.0, 0.0, 0.0),
            finger_torque_noise_std_nm=(0.0, 0.0, 0.0),
            finger_torque_noise_clip_sigma=(
                config.torque_noise_clip_sigma
            ),
            action_delay_policy_steps=0,
        )
    clamp_offsets = rng.uniform(
        np.asarray(config.clamp_offset_lower_rad, dtype=np.float64),
        np.asarray(config.clamp_offset_upper_rad, dtype=np.float64),
    )
    hand_kp_scale = float(
        rng.uniform(config.hand_kp_scale_lower, config.hand_kp_scale_upper)
    )
    hand_kd_scale = float(
        rng.uniform(config.hand_kd_scale_lower, config.hand_kd_scale_upper)
    )
    torque_bias = rng.uniform(
        np.asarray(config.torque_bias_lower_nm, dtype=np.float64),
        np.asarray(config.torque_bias_upper_nm, dtype=np.float64),
    )
    delay_index = int(rng.integers(0, len(config.action_delay_choices)))
    return ConnectorResidualEpisodeRandomization(
        clamp_nominal_offsets_rad=tuple(
            float(value) for value in clamp_offsets
        ),
        hand_kp_scale=hand_kp_scale,
        hand_kd_scale=hand_kd_scale,
        finger_torque_bias_nm=tuple(float(value) for value in torque_bias),
        finger_torque_noise_std_nm=config.torque_noise_std_nm,
        finger_torque_noise_clip_sigma=config.torque_noise_clip_sigma,
        action_delay_policy_steps=config.action_delay_choices[delay_index],
    )


def randomized_residual_config(
    base: ConnectorResidualConfig,
    episode: ConnectorResidualEpisodeRandomization,
) -> ConnectorResidualConfig:
    """Return episode-local action/observation scale without resizing it."""

    nominal = np.asarray(
        base.clamp_nominal_positions_rad, dtype=np.float64
    ) + np.asarray(episode.clamp_nominal_offsets_rad, dtype=np.float64)
    return replace(
        base,
        clamp_nominal_positions_rad=tuple(float(value) for value in nominal),
    )


def randomized_finger_torque_sample(
    raw_torques_nm: Any,
    episode: ConnectorResidualEpisodeRandomization,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply observation-only torque bias/noise to one raw 3-axis sample."""

    raw = np.asarray(raw_torques_nm, dtype=np.float64)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError("raw finger torques must be three finite values")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    standard_deviation = np.asarray(
        episode.finger_torque_noise_std_nm, dtype=np.float64
    )
    noise = rng.normal(0.0, standard_deviation, size=3)
    limit = episode.finger_torque_noise_clip_sigma * standard_deviation
    noise = np.clip(noise, -limit, limit)
    return (
        raw
        + np.asarray(episode.finger_torque_bias_nm, dtype=np.float64)
        + noise
    )
