"""Pure tests for the ghost-fingers occlusion evidence aggregator."""

from __future__ import annotations

import csv
import hashlib

import pytest

from kcg_connector import d38999_tooth_occlusion_evidence as evidence


def test_contact_trace_hash_is_ordered_and_excludes_other_columns(tmp_path):
    path = tmp_path / "summary.csv"
    fields = (
        "global_step",
        "phase",
        "phase_step",
        "segment_contact_records",
        "parent_px_m",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "global_step": "10",
                "phase": "motion",
                "phase_step": "1",
                "segment_contact_records": "3",
                "parent_px_m": "0.5",
            }
        )
    payload = (
        "global_step\x1fphase\x1fphase_step\x1f"
        "segment_contact_records\n10\x1fmotion\x1f1\x1f3\n"
    ).encode("utf-8")
    assert evidence.contact_trace_sha256(path) == hashlib.sha256(
        payload
    ).hexdigest()


def test_external_binding_rejects_path_hash_and_size_substitution(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    binding = {
        "path": str(path.resolve()),
        "sha256": evidence.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    evidence.validate_external_binding(binding, path, "value")
    bad = dict(binding, sha256="0" * 64)
    with pytest.raises(evidence.EvidenceError, match="size/SHA"):
        evidence.validate_external_binding(bad, path, "value")
    other = tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="path differs"):
        evidence.validate_external_binding(binding, other, "value")


def test_strict_blocker_accepts_only_visual_coverage_failures():
    assert "24/24" in evidence._strict_blocker(
        RuntimeError("ghost visual identity union is not 24/24")
    )
    with pytest.raises(evidence.EvidenceError, match="before visual"):
        evidence._strict_blocker(RuntimeError("physics trace differs"))


def test_cli_requires_fresh_ghost_and_output_paths(tmp_path):
    arguments = evidence._arguments(
        [
            "--ghost-root",
            str(tmp_path / "ghost"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert arguments.ghost_root == tmp_path / "ghost"
    assert arguments.output == tmp_path / "output"
