"""Regressions for the bounded proxy-only contact interval stage."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from kcg_connector.grasp.carts_v2 import contact_interval_solver as solver
from kcg_connector.grasp.carts_v2.models import CandidateSeed


def _seed(identifier: str, palm: float, axial: float) -> CandidateSeed:
    transform = np.eye(4); transform[0, 3] = axial
    return CandidateSeed(
        candidate_id=identifier, object_id="object", anchor_face_index=0,
        anchor_position_object_m=(0.0, 0.0, 0.0),
        object_from_hand=tuple(transform.ravel()),
        pregrasp_joint_positions_rad=(palm, 0.0, 0.0, 0.0),
        pregrasp_closure_phases=(0.1, 0.1, 0.1), source_sample_index=0,
        palm_configuration_rad=palm,
    )


def _row(family: str, index: int) -> dict:
    palm_count = 13 if family == "GLOBAL" else 7
    axial_count = 5 if family == "GLOBAL" else 4
    azimuth_count = 16 if family == "GLOBAL" else 8
    palm, axial = 0.1 * (index % palm_count), 0.1 * (index % axial_count)
    return {"candidate_id": f"{family}_{index:03d}", "family": family,
            "palm_key": palm, "axial_key": axial,
            "azimuth_key": index % azimuth_count,
            "maximum_positive_gap_m": 1.0e-3 + index * 1.0e-8,
            "gap_imbalance_m": 0.0, "hard_margin_m": 0.01,
            "table_margin_m": 0.02, "remaining_closure_phase": 0.2,
            "_seed": _seed(f"{family}_{index:03d}", palm, axial)}


def test_top120_enforces_family_caps_and_keeps_palm_axial_diversity() -> None:
    rows = [_row("GLOBAL", index) for index in range(90)]
    rows += [_row("DENSE_OPPOSITION", index) for index in range(85)]
    selected = solver._select_top120(rows)
    assert len(selected) == 120
    assert sum(row["family"] == "GLOBAL" for row in selected) == 60
    assert sum(row["family"] == "DENSE_OPPOSITION" for row in selected) == 60
    for family in ("GLOBAL", "DENSE_OPPOSITION"):
        source = [row for row in rows if row["family"] == family]
        chosen = [row for row in selected if row["family"] == family]
        assert {row["palm_key"] for row in chosen} == {row["palm_key"] for row in source}
        assert {row["axial_key"] for row in chosen} == {row["axial_key"] for row in source}
    assert max(sum(row["palm_key"] == palm for row in selected
                   if row["family"] == "GLOBAL")
               for palm in {row["palm_key"] for row in selected}) <= 6
    assert max(sum(row["palm_key"] == palm for row in selected
                   if row["family"] == "DENSE_OPPOSITION")
               for palm in {row["palm_key"] for row in selected}) <= 10


def test_top120_transfers_unused_family_slots_without_palm_monopoly() -> None:
    rows = [_row("GLOBAL", index) for index in range(50)]
    rows += [_row("DENSE_OPPOSITION", index) for index in range(100)]
    selected = solver._select_top120(rows)
    assert len(selected) == 120
    assert sum(row["family"] == "GLOBAL" for row in selected) == 50
    assert sum(row["family"] == "DENSE_OPPOSITION" for row in selected) == 70
    for palm in {row["palm_key"] for row in selected
                 if row["family"] == "DENSE_OPPOSITION"}:
        assert sum(row["palm_key"] == palm for row in selected
                   if row["family"] == "DENSE_OPPOSITION") <= 10


def test_sequential_solver_freezes_previous_fingers_at_expected_phase(monkeypatch) -> None:
    observed = []
    def fake_interval(_context, _seed, phases, finger):
        observed.append((finger, tuple(phases)))
        return {"proxy_expected_phase": 0.2 + 0.1 * finger,
                "proxy_q_expected_rad": 0.2, "proxy_q_safe_max_rad": 0.3}
    monkeypatch.setattr(solver, "_finger_interval", fake_interval)
    result = solver._sequential_intervals({}, _seed("seed", 0.0, 0.0))
    assert result["status"] == "PROXY_INTERVAL_SURVIVE"
    assert [row[0] for row in observed] == [0, 1, 2]
    assert np.allclose([row[1] for row in observed],
                       [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.2, 0.3, 0.1)])


def test_bisection_leaves_safe_state_within_one_control_increment() -> None:
    def state(phase):
        return {"phase": phase, "q_rad": phase, "safe": phase < 0.503}
    low, high = solver._bisect_pair(state(0.4), state(0.6), state,
                                    lambda row: row["safe"])
    assert low["safe"] is True and high["safe"] is False
    assert high["q_rad"] - low["q_rad"] <= 0.0015
    assert low["q_rad"] < 0.503 <= high["q_rad"]


def test_empty_first_finger_interval_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_finger_interval",
                        lambda *_args, **_kwargs: None)
    result = solver._sequential_intervals(SimpleNamespace(),
                                          _seed("empty", math.pi / 4.0, 0.5))
    assert result["status"] == "PROXY_INTERVAL_REJECT"
    assert result["reason"] == "NO_SAFE_PROXY_CONTACT_INTERVAL_FINGER_1"
    assert result["finger_intervals"] == []


def test_duplicate_specification_identity_fails_before_geometry() -> None:
    rows = [{"candidate_id": "duplicate", "status": "SEED_GEOMETRY_REJECT"}
            for _ in range(1488)]
    with np.testing.assert_raises_regex(ValueError, "1488 unique non-empty"):
        solver.solve_proxy_contact_intervals(SimpleNamespace(), (), rows)
