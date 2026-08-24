"""Regressions for descriptor-specific preshapes in the reused V2 path."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import fast_filter_predictions
from kcg_connector.grasp.carts_v2.models import (
    joint_positions_for_phases,
    load_v2_inputs,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_grasp_v2.yaml"
OBJECT_A = "current_d38999_26kj61sn_public_spec"


def test_candidate_spread_joint_survives_closure_and_full_sweep(monkeypatch) -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
    base = next(seed for seed in generate_raw_candidates(inputs) if seed.source_sample_index == 50)
    spread_index = inputs.hand_model.independent_joint_names.index("f1j1")
    maximum_increment = (
        inputs.config.section("dynamic")["finger_maximum_speed_rad_s"]
        * inputs.config.section("dynamic")["physics_dt_s"]
    )
    for spread in (0.699, 0.701):
        preshape = list(base.pregrasp_joint_positions_rad)
        preshape[spread_index] = spread
        seed = replace(base, candidate_id=f"spread_{spread}", pregrasp_joint_positions_rad=tuple(preshape))
        observed_closure: list[float] = []
        with monkeypatch.context() as patch:
            original = inputs.hand_model.pad_transforms

            def record_pad_fk(joints, **kwargs):
                observed_closure.append(float(np.asarray(joints)[spread_index]))
                return original(joints, **kwargs)

            patch.setattr(inputs.hand_model, "pad_transforms", record_pad_fk)
            prediction = SequentialClosurePredictor(inputs).predict(seed)
        assert prediction.status == "CLOSURE_SURVIVE"
        assert observed_closure and np.allclose(observed_closure, spread)

        observed_sweep: list[float] = []
        with monkeypatch.context() as patch:
            original = inputs.hand_model.forward_kinematics

            def record_sweep_fk(joints, **kwargs):
                observed_sweep.append(float(np.asarray(joints)[spread_index]))
                return original(joints, **kwargs)

            patch.setattr(inputs.hand_model, "forward_kinematics", record_sweep_fk)
            result = fast_filter_predictions(inputs, (prediction,))[0]
        assert observed_sweep and np.allclose(observed_sweep, spread)
        assert result.maximum_joint_increment_rad <= maximum_increment + 1.0e-12


def test_reference_joint_vector_validation_fails_closed() -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
    seed = generate_raw_candidates(inputs)[0]
    valid = list(seed.pregrasp_joint_positions_rad)
    assert np.array_equal(
        joint_positions_for_phases(inputs, seed.pregrasp_closure_phases),
        joint_positions_for_phases(
            inputs,
            seed.pregrasp_closure_phases,
            reference_joint_positions_rad=valid,
        ),
    )
    for reference in (valid[:-1], [np.nan, *valid[1:]], [1.571, *valid[1:]]):
        with pytest.raises(ValueError):
            joint_positions_for_phases(
                inputs,
                seed.pregrasp_closure_phases,
                reference_joint_positions_rad=reference,
            )
