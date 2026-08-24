"""The learned-pose contact predictor must not skip real controller steps."""

import json
from pathlib import Path

import numpy as np

from kcg_connector.grasp.carts_v2.closure_predictor import closure_phase_samples
from kcg_connector.grasp.carts_v2.models import (
    joint_positions_for_phases,
    load_v2_inputs,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_graspgenx_route1.yaml"
DESCRIPTORS = ROOT / "artifacts/carts_v2/graspgenx/descriptors/descriptor_manifest.json"
OBJECT_A = "current_d38999_26kj61sn_public_spec"


def test_graspgenx_contact_samples_follow_real_joint_increment() -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
    descriptor = json.loads(DESCRIPTORS.read_text())["descriptors"][2]
    names = inputs.hand_model.independent_joint_names
    reference = tuple(
        float(descriptor["open_joint_positions_rad"][name]) for name in names
    )
    phases = (0.0, 0.0, 0.0)
    samples = closure_phase_samples(
        inputs, phases, 0, float(descriptor["maximum_closure_phase"]), reference
    )
    states = [
        joint_positions_for_phases(
            inputs, (float(phase), 0.0, 0.0),
            reference_joint_positions_rad=reference,
        )
        for phase in np.concatenate(([0.0], samples))
    ]
    maximum_increment = (
        inputs.config.section("dynamic")["finger_maximum_speed_rad_s"]
        * inputs.config.section("dynamic")["physics_dt_s"]
    )
    assert len(samples) == 644
    assert max(
        np.max(np.abs(right - left)) for left, right in zip(states, states[1:])
    ) <= maximum_increment + 1.0e-12
