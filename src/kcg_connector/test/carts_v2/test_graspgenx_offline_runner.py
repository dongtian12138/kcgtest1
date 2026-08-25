"""Regressions for the file-bound GraspGenX offline runner."""

from types import SimpleNamespace

import numpy as np

from scripts.carts_v2.run_graspgenx_offline import _pregrasp_contact_key


def test_pregrasp_contact_cache_reuses_only_identical_pad_poses() -> None:
    transforms = {name: np.eye(4) for name in ("l1", "l2", "l3")}
    inputs = SimpleNamespace(
        hand_model=SimpleNamespace(forward_kinematics=lambda *_args, **_kwargs: transforms),
        task_grip_surfaces={f"p{i}": SimpleNamespace(link_name=f"l{i}")
                            for i in range(1, 4)},
    )
    seed = SimpleNamespace(
        pregrasp_joint_positions_rad=(0.0,) * 4,
        object_from_hand_matrix=lambda: np.eye(4),
        pregrasp_closure_phases=(0.0, 0.0, 0.0),
    )
    calls = []

    def query(name, transform):
        calls.append((name, transform.copy()))
        return SimpleNamespace(distance_m=np.asarray((0.01 * len(calls),))), None, None

    cache = {}
    first = _pregrasp_contact_key(inputs, SimpleNamespace(query_pad=query), seed, cache)
    second = _pregrasp_contact_key(inputs, SimpleNamespace(query_pad=query), seed, cache)
    assert len(calls) == 3
    assert first == second
    transforms["l1"][2, 3] = 0.001
    _pregrasp_contact_key(inputs, SimpleNamespace(query_pad=query), seed, cache)
    assert len(calls) == 4
