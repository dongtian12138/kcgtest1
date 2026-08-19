from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)


ROOT = Path(__file__).resolve().parents[3]
NOMINAL = ROOT / "src/kcg_connector/config/d38999_keyed_v2_tabletop_physical_grasp_v1.yaml"
CANDIDATE = ROOT / "src/kcg_connector/config/d38999_keyed_v2_tabletop_physical_grasp_compliant_s070_candidate_v1.yaml"


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_s070_candidate_is_one_variable_change_from_nominal():
    nominal = _read(NOMINAL)
    candidate = _read(CANDIDATE)

    assert nominal["sequential"]["consolidation_final_stiffness_scale"] == 1.0
    assert candidate["sequential"]["consolidation_final_stiffness_scale"] == pytest.approx(0.70)

    normalized = deepcopy(candidate)
    normalized["sequential"]["consolidation_final_stiffness_scale"] = 1.0
    assert normalized == nominal


def test_s070_candidate_loads_and_preserves_fail_closed_gates():
    loaded = load_physical_grasp_experiment_config(CANDIDATE)
    document = _read(CANDIDATE)

    assert loaded.sequential.consolidation_final_stiffness_scale == pytest.approx(0.70)
    assert document["lift"]["maximum_wrist_moment_nm"] == pytest.approx(0.30)
    assert document["lift"]["maximum_wrist_force_n"] == pytest.approx(8.0)
    assert document["lift"]["maximum_root_torque_delta_nm"] == pytest.approx(2.0)
    assert all(value is False for value in document["boundaries"].values())
    assert all(bounds[0] == bounds[-1] for bounds in document["randomization"].values())
