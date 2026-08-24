from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2 import fast_filter


ROOT = Path(__file__).resolve().parents[4]
CONTROLLER = ROOT / "src/kcg_connector/isaac/carts_v2/controller.py"


def _controller():
    spec = importlib.util.spec_from_file_location("carts_v2_controller_test", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graspgenx_approach_axis_offsets_pregrasp_away_from_object(monkeypatch) -> None:
    controller = _controller()
    targets = []

    def solve(_settings, **kwargs):
        targets.append(np.asarray(kwargs["target_world_from_hand_base"]))
        return tuple(np.zeros(7)), 0.0, 0.0, 0

    monkeypatch.setattr(controller, "solve_bounded_hand_base_ik", solve)
    inputs = type("Inputs", (), {
        "config": type("Config", (), {"section": lambda _self, name: (
            {"approach_clearance_height_m": 0.10}
            if name == "dynamic" else {"approach_path_sample_count": 3}
        )})(),
    })()
    target = np.eye(4)
    controller._solve_approach_waypoints(
        inputs, object(), {}, (0.0,), target, np.asarray((1.0, 0.0, 0.0))
    )
    assert np.allclose(targets[0][:3, 3], (-0.10, 0.0, 0.0))
    assert np.allclose(targets[-1][:3, 3], (0.0, 0.0, 0.0))


def _offline_case(direction_object):
    world_from_object = np.asarray(
        (
            (0.0, -1.0, 0.0, 0.52),
            (1.0, 0.0, 0.0, -0.21),
            (0.0, 0.0, 1.0, 0.20),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    object_from_hand = np.eye(4)
    object_from_hand[:3, 3] = (0.01, -0.02, 0.03)
    sections = {
        "fast_filter": {"approach_path_sample_count": 5},
        "dynamic": {
            "approach_clearance_height_m": 0.10,
            "finger_maximum_speed_rad_s": 0.18,
            "physics_dt_s": 1.0 / 120.0,
        },
        "closure_prediction": {
            "closing_order": (
                "finger_1_pad",
                "finger_2_pad",
                "finger_3_pad",
            )
        },
    }
    inputs = SimpleNamespace(
        config=SimpleNamespace(section=lambda name: sections[name]),
        frozen_world_from_object=world_from_object,
        hand_contract=SimpleNamespace(
            pads=tuple(
                SimpleNamespace(name=f"finger_{index}_pad")
                for index in range(1, 4)
            )
        ),
    )
    seed = SimpleNamespace(
        approach_direction_object=direction_object,
        object_from_hand_matrix=lambda: object_from_hand,
        pregrasp_joint_positions_rad=(0.0, 0.0, 0.0, 0.0),
        pregrasp_closure_phases=(0.0, 0.0, 0.0),
    )
    prediction = SimpleNamespace(
        seed=seed,
        final_closure_phases=(0.0, 0.0, 0.0),
    )
    return inputs, prediction, world_from_object @ object_from_hand


def test_offline_approach_states_match_controller_frame_and_sign(monkeypatch) -> None:
    inputs, prediction, target = _offline_case((1.0, 0.0, 0.0))
    monkeypatch.setattr(
        fast_filter,
        "joint_positions_for_phases",
        lambda *_args, **_kwargs: np.zeros(4),
    )

    states = fast_filter._sampled_hand_states(inputs, prediction)[:5]
    direction_world = inputs.frozen_world_from_object[:3, :3] @ np.asarray(
        prediction.seed.approach_direction_object
    )
    expected = []
    for fraction in np.linspace(1.0, 0.0, 5):
        pose = np.array(target, copy=True)
        pose[:3, 3] -= direction_world * 0.10 * float(fraction)
        expected.append(pose)

    assert [row[0] for row in states] == [
        "APPROACH_00",
        "APPROACH_01",
        "APPROACH_02",
        "APPROACH_03",
        "PREGRASP",
    ]
    assert np.allclose(np.asarray([row[1] for row in states]), expected)


def test_offline_approach_states_keep_legacy_world_z_fallback(monkeypatch) -> None:
    inputs, prediction, target = _offline_case(None)
    monkeypatch.setattr(
        fast_filter,
        "joint_positions_for_phases",
        lambda *_args, **_kwargs: np.zeros(4),
    )

    states = fast_filter._sampled_hand_states(inputs, prediction)[:5]
    expected = []
    for fraction in np.linspace(1.0, 0.0, 5):
        pose = np.array(target, copy=True)
        pose[2, 3] += 0.10 * float(fraction)
        expected.append(pose)

    assert np.allclose(np.asarray([row[1] for row in states]), expected)


@pytest.mark.parametrize(
    "direction",
    ((2.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (1.0, 0.0)),
)
def test_offline_approach_direction_fails_closed(direction) -> None:
    inputs, prediction, _target = _offline_case(direction)
    with pytest.raises(ValueError, match="finite unit vector"):
        fast_filter._sampled_hand_states(inputs, prediction)
