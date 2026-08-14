import numpy as np
import pytest

from kcg_connector.robot_model import (
    ACTIVE_HAND_JOINT_NAMES,
    ALL_HAND_JOINT_NAMES,
    ARM_JOINT_NAMES,
    expand_active_hand_positions,
    named_joint_target,
)


def test_active_hand_mapping_matches_mechanical_couplings():
    targets = expand_active_hand_positions([0.2, 0.4, 0.6, 0.8])

    assert targets["f3j1"] == targets["f1j1"] == 0.2
    assert targets["f1j3"] == targets["f1j2"] == 0.4
    assert targets["f2j2"] == targets["f2j1"] == 0.6
    assert targets["f3j3"] == targets["f3j2"] == 0.8


def test_named_target_uses_importer_dof_order():
    names = tuple(reversed(ARM_JOINT_NAMES + ALL_HAND_JOINT_NAMES))
    target = named_joint_target(
        names,
        np.arange(7, dtype=np.float64) / 10.0,
        [0.2, 0.4, 0.6, 0.8],
    )
    mapped = dict(zip(names, target))

    assert mapped["iiwa_joint_7"] == pytest.approx(0.6)
    assert mapped["f1j3"] == pytest.approx(0.4)
    assert mapped["f3j1"] == pytest.approx(0.2)


def test_named_target_rejects_missing_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="missing joints"):
        named_joint_target(ARM_JOINT_NAMES, [0.0] * 7, [0.0] * 4)
    with pytest.raises(ValueError, match="finite"):
        expand_active_hand_positions([0.0, float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="unique"):
        named_joint_target(
            ARM_JOINT_NAMES + ALL_HAND_JOINT_NAMES + ("f1j1",),
            [0.0] * 7,
            [0.0] * 4,
        )


def test_joint_name_contract_is_four_active_and_eight_modeled():
    assert ACTIVE_HAND_JOINT_NAMES == ("f1j1", "f1j2", "f2j1", "f3j2")
    assert len(ALL_HAND_JOINT_NAMES) == 8
