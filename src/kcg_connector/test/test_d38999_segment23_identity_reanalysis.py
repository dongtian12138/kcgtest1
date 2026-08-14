"""CPU-only tests for geometry-anchored Segment_23 identity recovery."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector import d38999_segment23_identity_reanalysis as identity


REPOSITORY = Path(__file__).resolve().parents[3]
ASSET = (
    REPOSITORY
    / "artifacts/kcg_connector/isaac/"
    "d38999_shell25j_61_pair_proxy_v1.usda"
)


def test_authored_asset_has_exact_cyclic_segment23_geometry():
    geometry = identity.parse_authored_tooth_geometry(ASSET)
    assert geometry["segment_count"] == 24
    assert set(geometry["centers_local_m"]) == {
        f"Segment_{index:02d}" for index in range(24)
    }
    target = geometry["centers_local_m"]["Segment_23"]
    assert math.hypot(*target[:2]) == pytest.approx(0.02175, abs=1.0e-12)
    assert math.degrees(math.atan2(target[1], target[0])) % 360.0 == (
        pytest.approx(345.0, abs=1.0e-9)
    )


def test_asset_parser_rejects_index_rotation_substitution(tmp_path):
    changed = tmp_path / "changed.usda"
    source = ASSET.read_text(encoding="utf-8")
    marker = 'def Cube "Segment_23"'
    prefix, block = source.split(marker, 1)
    block = block.replace(
        "float xformOp:rotateZ = 345",
        "float xformOp:rotateZ = 330",
        1,
    )
    changed.write_text(prefix + marker + block, encoding="utf-8")
    with pytest.raises(identity.EvidenceError, match="rotation differs"):
        identity.parse_authored_tooth_geometry(changed)


def test_mutual_nearest_recovers_geometry_without_trusting_hue_label():
    projected = {
        "Segment_22": np.asarray((-10.0, 0.0)),
        "Segment_23": np.asarray((0.0, 0.0)),
        "Segment_00": np.asarray((10.0, 0.0)),
    }
    observed = {
        # The actual captured alias: the geometry at Segment_23 was assigned
        # the neighbouring cyclic palette label by the RGB hue decoder.
        "Segment_22": np.asarray((1.0, 0.0)),
        "Segment_21": np.asarray((-9.0, 0.0)),
        "Segment_00": np.asarray((11.0, 0.0)),
    }
    result = identity.mutual_nearest_assignment(
        projected=projected,
        observed=observed,
    )
    assert result["candidate_hue_label"] == "Segment_22"
    assert result["projection_error_pitch_fraction"] == pytest.approx(0.1)
    assert result["projected_identity_margin_pitch_fraction"] == (
        pytest.approx(0.8)
    )


def test_mutual_nearest_rejects_ambiguous_or_wrong_geometry():
    projected = {
        "Segment_22": np.asarray((-10.0, 0.0)),
        "Segment_23": np.asarray((0.0, 0.0)),
        "Segment_00": np.asarray((10.0, 0.0)),
    }
    with pytest.raises(identity.EvidenceError, match="not mutual nearest"):
        identity.mutual_nearest_assignment(
            projected=projected,
            observed={"Segment_22": np.asarray((6.0, 0.0))},
        )
    with pytest.raises(identity.EvidenceError, match="one-third pitch"):
        identity.mutual_nearest_assignment(
            projected=projected,
            observed={"Segment_22": np.asarray((4.0, 0.0))},
        )


def test_claim_boundaries_are_explicit_in_source():
    source = Path(identity.__file__).read_text(encoding="utf-8")
    assert '"hue_gate_widened": False' in source
    assert '"render_jitter_absence_claim_authorized": False' in source
    assert "physics_zero_anomaly_does_not_prove_renderer_zero_jitter" in source
