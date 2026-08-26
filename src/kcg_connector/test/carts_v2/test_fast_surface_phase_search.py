"""Focused regressions for cached feature-aware Surface-V2 ranking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2 import fast_surface_phase_search as search
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.opposition_seed_generator import (
    generate_opposition_anchors,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_surface_v2_fast6h.yaml"
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"


def _fake_seeds():
    rows = []
    source = 0
    phases = tuple((a / 10.0, b / 10.0, c / 10.0)
                   for a in range(3) for b in range(3) for c in range(3))
    for palm in range(45, 80, 5):
        for phase_index, phase in enumerate(phases):
            for azimuth in range(72):
                for axial in range(3):
                    rows.append(SimpleNamespace(
                        candidate_id=(f"q{palm:03d}_a{azimuth:03d}_z{axial}__"
                                      f"p{phase_index:02d}"),
                        palm_configuration_rad=math.radians(palm),
                        pregrasp_closure_phases=phase, source_sample_index=source))
                    source += 1
    return tuple(rows)


def test_public_search_registers_full_grid_and_caps_exact(monkeypatch) -> None:
    seeds = _fake_seeds()
    inputs = SimpleNamespace(
        task_grip_surfaces={"finger_1_pad": object()},
        face_roles=SimpleNamespace(method="TASK_AXIS_OUTER_ENVELOPE_THREE_ROLE_V2"),
        object_contract=SimpleNamespace(object_id=OBJECT_B),
    )
    monkeypatch.setattr(search, "generate_feature_opposition_grid",
                        lambda _inputs: (seeds, {"candidate_count": len(seeds),
                                                "axial_fractions": [0.25, 0.5, 0.75]}))
    monkeypatch.setattr(search, "_role_indexes", lambda _inputs: ({}, np.zeros(3)))
    monkeypatch.setattr(search, "_surface_representatives", lambda _surface: {})
    monkeypatch.setattr(search, "_geometry_support", lambda _inputs: ({}, ()))
    monkeypatch.setattr(
        search, "_kinematic_cache",
        lambda _inputs, seed, *_args: {"preshape": seed.pregrasp_closure_phases})

    def fake_score(_inputs, _indexes, _center, _cache, seed, *, patches):
        margin = 1.0e-3 - seed.source_sample_index * 1.0e-12
        return {"status": "FAST_SHORTLIST_ELIGIBLE",
                "three_contact_regions": True, "hard_margin_m": margin,
                "primary_fraction": 1.0, "effective_area_m2": float(patches),
                "table_clearance_proxy_m": 0.01, "closure_balance_phase": 0.0,
                "stop_phases": [0.4, 0.4, 0.4], "witness_counts": [3, 3, 3],
                "maximum_self_aabb_overlap_count": 0}

    monkeypatch.setattr(search, "_score", fake_score)
    shortlist, audit = search.search_feature_aware_opposition(inputs)
    assert audit["registered_candidate_count"] == 7 * 72 * 3 * 27 == 40824
    assert audit["fk_cache_count"] == 7 * 27 == 189
    assert audit["patch_ranked_count"] == 96
    assert len(shortlist) == audit["exact_shortlist_count"] == 24
    assert audit["patch_axial_counts"] == {"0": 32, "1": 32, "2": 32}
    assert audit["exact_axial_counts"] == {"0": 8, "1": 8, "2": 8}
    assert len({row.candidate_id for row in shortlist}) == 24
    assert "all_candidates" not in audit
    json.dumps(audit, allow_nan=False)


def test_axial_strata_override_single_layer_global_rank_without_reordering_layers() -> None:
    rows = []
    for axial in range(3):
        for rank in range(12):
            rows.append({"candidate_id": f"z{axial}_{rank:02d}", "axial_index": axial,
                         "three_contact_regions": True,
                         "hard_margin_m": 1.0 - axial - rank * 1.0e-3,
                         "primary_fraction": 1.0, "effective_area_m2": 1.0,
                         "table_clearance_proxy_m": 0.01,
                         "closure_balance_phase": 0.0})
    selected = search._stratified_rank(rows, 24)
    assert search._axial_counts(selected) == {"0": 8, "1": 8, "2": 8}
    assert {row["candidate_id"] for row in selected if row["axial_index"] == 2} == {
        f"z2_{rank:02d}" for rank in range(8)}


def test_cached_task_surface_center_matches_direct_real_fk() -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_B)
    seed = generate_opposition_anchors(inputs, (math.radians(60.0),))[0][0]
    representatives = {name: search._surface_representatives(surface)
                       for name, surface in inputs.task_grip_surfaces.items()}
    boxes, pairs = search._geometry_support(inputs)
    cache = search._kinematic_cache(inputs, seed, representatives, boxes, pairs)
    surface = inputs.task_grip_surfaces["finger_1_pad"]
    transform = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad)[surface.link_name]
    expected = (representatives["finger_1_pad"]["center"] @ transform[:3, :3].T
                + transform[:3, 3])
    assert len(cache["paths"]) == 3
    assert all(len(path["phases"]) == 25 for path in cache["paths"])
    assert all(len(path["patch_areas"]) == 33 for path in cache["paths"])
    assert cache["paths"][0]["centers"][0] == pytest.approx(expected, abs=1.0e-12)


def test_five_nanometre_role_race_fails_closed_to_exact_boundary() -> None:
    assert search._boundary_status(True, 5.559e-9) == "UNRESOLVED_BOUNDARY"
    assert search._boundary_status(True, 2.0e-8) == "FAST_SHORTLIST_ELIGIBLE"
    assert search._boundary_status(True, -2.0e-8) == "FAST_HARD_FIRST_PROXY"
    assert search._boundary_status(False, 1.0) == "FAST_NO_THREE_CONTACT_REGIONS"


def test_center_table_proxy_receives_every_cached_closure_state(monkeypatch) -> None:
    observed = []
    path = {"centers": np.zeros((2, 3)), "center_normals": np.zeros((2, 3)),
            "center_motion": np.zeros((2, 3)), "phases": np.asarray((0.0, 0.1)),
            "support": np.zeros((2, 8, 3)), "self_overlap": 0}
    monkeypatch.setattr(search, "_contact_summary", lambda *_args, **_kwargs: {
        "found": True, "phase": 0.1, "hard_margin": 0.001,
        "primary_fraction": 1.0, "area": 0.0, "witness_count": 1})
    monkeypatch.setattr(search, "_table_proxy",
                        lambda _inputs, points: observed.append(len(points)) or 0.01)
    seed = SimpleNamespace(object_from_hand_matrix=lambda: np.eye(4),
                           candidate_id="seed")
    result = search._score(SimpleNamespace(), {}, np.zeros(3),
                           {"paths": (path, path, path), "preshape": (0.0,) * 3},
                           seed, patches=False)
    assert observed == [3 * 2 * 8]
    assert result["table_clearance_proxy_m"] == 0.01
