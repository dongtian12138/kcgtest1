"""Pure tests for seeded residual-v0 episode randomization."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.residual_randomization import (
    RANDOMIZATION_SCHEMA_VERSION,
    load_connector_residual_randomization_config,
    randomized_finger_torque_sample,
    randomized_residual_config,
    reproducible_stream_reset_seed,
    sample_connector_residual_randomization,
)
from kcg_connector.residual_rl import load_connector_residual_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RANDOMIZATION_PATH = (
    PACKAGE_ROOT / "config/connector_residual_randomization_v1.yaml"
)
TASK_PATH = PACKAGE_ROOT / "config/connector_task.yaml"


def _load():
    return load_connector_residual_randomization_config(
        RANDOMIZATION_PATH
    )


def test_versioned_config_preserves_v0_shape_and_excludes_physics():
    config = _load()
    assert config.schema_version == RANDOMIZATION_SCHEMA_VERSION
    assert config.interface_version == "kcg_connector_twist_residual_v0"
    assert config.action_size == 4
    assert config.observation_size == 24
    assert config.safety_signal_source == "raw_physics"
    assert set(config.excluded_physics_parameters) == {
        "mass",
        "friction",
        "thread_lead",
    }


def test_same_seed_is_identical_and_different_seed_changes_sample():
    config = _load()
    first = sample_connector_residual_randomization(
        config, np.random.default_rng(1201)
    )
    repeated = sample_connector_residual_randomization(
        config, np.random.default_rng(1201)
    )
    different = sample_connector_residual_randomization(
        config, np.random.default_rng(1202)
    )
    assert first == repeated
    assert first != different


def test_base_seed_defines_reproducible_but_varied_episode_stream():
    config = _load()

    def sequence(base_seed):
        samples = []
        rng = None
        for episode_index in range(2):
            reset_seed = reproducible_stream_reset_seed(
                base_seed, episode_index
            )
            if reset_seed is not None:
                domain_seed, _ = np.random.SeedSequence(
                    reset_seed
                ).spawn(2)
                rng = np.random.default_rng(domain_seed)
            samples.append(
                sample_connector_residual_randomization(config, rng)
            )
        return tuple(samples)

    first_run = sequence(1204)
    repeated_run = sequence(1204)
    assert first_run == repeated_run
    assert first_run[0] != first_run[1]
    assert reproducible_stream_reset_seed(1204, 0) == 1204
    assert reproducible_stream_reset_seed(1204, 1) is None


@pytest.mark.parametrize("episode_index", (-1, True, 0.5))
def test_episode_stream_rejects_invalid_index(episode_index):
    with pytest.raises((TypeError, ValueError), match="episode_index"):
        reproducible_stream_reset_seed(1204, episode_index)


def test_every_sample_stays_inside_conservative_v1_bounds():
    config = _load()
    seen_delays = set()
    for seed in range(200):
        sample = sample_connector_residual_randomization(
            config, np.random.default_rng(seed)
        )
        assert np.max(np.abs(sample.clamp_nominal_offsets_rad)) <= 0.01
        assert 0.95 <= sample.hand_kp_scale <= 1.05
        assert 0.95 <= sample.hand_kd_scale <= 1.05
        assert np.max(np.abs(sample.finger_torque_bias_nm)) <= 0.005
        assert sample.finger_torque_noise_std_nm == (0.002,) * 3
        assert sample.action_delay_policy_steps in (0, 1)
        seen_delays.add(sample.action_delay_policy_steps)
    assert seen_delays == {0, 1}


def test_episode_nominal_changes_but_v0_action_contract_does_not():
    distribution = _load()
    base = load_connector_residual_config(TASK_PATH)
    sample = sample_connector_residual_randomization(
        distribution, np.random.default_rng(41)
    )
    episode_config = randomized_residual_config(base, sample)
    expected = np.asarray(base.clamp_nominal_positions_rad) + np.asarray(
        sample.clamp_nominal_offsets_rad
    )
    assert np.allclose(
        episode_config.clamp_nominal_positions_rad, expected
    )
    assert episode_config.interface_version == base.interface_version
    assert episode_config.clamp_position_residual_limits_rad == (
        base.clamp_position_residual_limits_rad
    )


def test_torque_observation_noise_is_seeded_bounded_and_not_in_place():
    distribution = _load()
    sample = sample_connector_residual_randomization(
        distribution, np.random.default_rng(8)
    )
    raw = np.asarray([0.10, -0.20, 0.30], dtype=np.float64)
    original = raw.copy()
    first = randomized_finger_torque_sample(
        raw, sample, np.random.default_rng(99)
    )
    repeated = randomized_finger_torque_sample(
        raw, sample, np.random.default_rng(99)
    )
    assert np.array_equal(raw, original)
    assert np.array_equal(first, repeated)
    error_after_bias = first - raw - np.asarray(
        sample.finger_torque_bias_nm
    )
    assert np.all(np.abs(error_after_bias) <= 0.006 + 1.0e-12)


@pytest.mark.parametrize(
    "mutator,message",
    (
        (
            lambda document: document["contract"].update(
                {"observation_size": 25}
            ),
            "observation size",
        ),
        (
            lambda document: document["randomization"][
                "clamp_nominal_offset"
            ].update({"upper_rad": [0.011, 0.01, 0.01]}),
            "clamp offsets",
        ),
        (
            lambda document: document["randomization"][
                "hand_pd_scale"
            ].update({"kp": [0.94, 1.05]}),
            "kp scale",
        ),
        (
            lambda document: document["safety"].update(
                {"signal_source": "noisy_observation"}
            ),
            "raw_physics",
        ),
        (
            lambda document: document["randomization"][
                "action_delay_policy_steps"
            ].update({"choices": [0, 2]}),
            "exactly",
        ),
    ),
)
def test_invalid_or_unsafe_configuration_is_rejected(
    tmp_path, mutator, message
):
    document = yaml.safe_load(
        RANDOMIZATION_PATH.read_text(encoding="utf-8")
    )
    mutator(document)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_connector_residual_randomization_config(path)


def test_disabled_distribution_is_exactly_nominal():
    config = replace(_load(), enabled=False)
    sample = sample_connector_residual_randomization(
        config, np.random.default_rng(13)
    )
    assert sample.clamp_nominal_offsets_rad == (0.0, 0.0, 0.0)
    assert sample.hand_kp_scale == 1.0
    assert sample.hand_kd_scale == 1.0
    assert sample.finger_torque_bias_nm == (0.0, 0.0, 0.0)
    assert sample.finger_torque_noise_std_nm == (0.0, 0.0, 0.0)
    assert sample.action_delay_policy_steps == 0
