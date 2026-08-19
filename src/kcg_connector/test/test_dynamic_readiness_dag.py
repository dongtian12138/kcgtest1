from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.dynamic_readiness_dag import (
    DEFAULT_DEPENDENCIES,
    EXPECTED_TASK_KEYS,
    SCHEMA_VERSION,
    build_dynamic_readiness_dag,
    render_dot,
    render_markdown,
    topological_order,
    write_report_triplet,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORK_QUEUE = Path("artifacts/agent_control/WORK_QUEUE.yaml")
BLOCKER_LEDGER = Path("artifacts/agent_control/BLOCKER_LEDGER.jsonl")
GATE_LEDGER = Path("artifacts/agent_control/GATE_LEDGER.csv")
GENERATED_AT = "2026-08-18T00:41:00Z"


def _build(root: Path = REPOSITORY_ROOT, **kwargs):
    return build_dynamic_readiness_dag(
        repository_root=root,
        work_queue_path=WORK_QUEUE,
        blocker_ledger_path=BLOCKER_LEDGER,
        gate_ledger_path=GATE_LEDGER,
        generated_at_utc=GENERATED_AT,
        **kwargs,
    )


def _copy_fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "repository"
    queue = yaml.safe_load((REPOSITORY_ROOT / WORK_QUEUE).read_text(encoding="utf-8"))
    for relative in (WORK_QUEUE, BLOCKER_LEDGER, GATE_LEDGER):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    for group in queue["groups"].values():
        for task in group["tasks"].values():
            source = REPOSITORY_ROOT / task["evidence"]
            destination = root / task["evidence"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    return root, queue


def _write_queue(root: Path, queue: dict) -> None:
    (root / WORK_QUEUE).write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_current_queue_counts_and_no_dynamic_promotion():
    report = _build()
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["node_count"] == 41
    assert report["implementation_ready_count"] == 35
    assert report["parked_queue_node_count"] == 6
    assert report["status_counts"]["OFFLINE_PASS"] == 31
    assert report["status_counts"]["STATIC_PASS"] == 4
    assert report["formal_dynamic_pass_count"] == 0
    assert report["dynamic_execution_authorized_count"] == 0
    assert report["static_or_offline_promoted_to_dynamic_count"] == 0


def test_dependency_graph_is_complete_acyclic_and_topological():
    order = topological_order(DEFAULT_DEPENDENCIES)
    assert set(order) == set(EXPECTED_TASK_KEYS)
    assert len(order) == 41
    positions = {key: index for index, key in enumerate(order)}
    for child, parents in DEFAULT_DEPENDENCIES.items():
        assert all(positions[parent] < positions[child] for parent in parents)


def test_earliest_dynamic_gate_and_frozen_home_frontier():
    report = _build()
    assert report["current_frontier_state"] == "HOME"
    assert report["earliest_unmet_formal_dynamic_gate"] == {
        "task_key": "A2",
        "name": "正式名义插入及程序完整性",
        "direct_root_blocker": "A1",
        "classification": "PARKED_FINAL_FOR_THIS_WINDOW",
    }


def test_top_three_root_blockers_follow_mainline_order():
    report = _build()
    rows = report["top_three_priority_root_blockers"]
    assert [row["task_key"] for row in rows] == ["A1", "B1", "C8"]
    assert rows[0]["classification"] == "PARKED_FINAL_FOR_THIS_WINDOW"
    assert rows[1]["classification"] == "AUTHORITATIVE_GRASP_BOUNDARY_MISSING"
    assert rows[2]["classification"] == (
        "FOUNDATIONPOSE_RUNTIME_AND_CURRENT_MODEL_INPUTS_NOT_READY"
    )


def test_all_five_root_blockers_are_evidence_bound_with_descendants():
    report = _build()
    rows = report["root_blockers"]
    assert [row["task_key"] for row in rows] == ["A1", "B1", "C8", "E3", "E6"]
    assert all(row["source_path"] for row in rows)
    assert all(row["blocked_descendant_count"] > 0 for row in rows)


def test_every_node_preserves_raw_and_concrete_dependency():
    report = _build()
    assert [row["task_key"] for row in report["nodes"]] == list(EXPECTED_TASK_KEYS)
    assert all(row["raw_dynamic_dependency"] is not None for row in report["nodes"])
    a2 = next(row for row in report["nodes"] if row["task_key"] == "A2")
    assert a2["raw_dynamic_dependency"] == "A1_DERIVED_INIT_GATE"
    assert a2["dynamic_dependencies"] == ["A1"]
    assert a2["unresolved_dynamic_dependencies"] == ["A1"]


def test_cycle_and_unknown_dependency_fail_closed():
    cyclic = dict(DEFAULT_DEPENDENCIES)
    cyclic["A1"] = ("F3",)
    with pytest.raises(ValueError, match="cycle"):
        topological_order(cyclic)
    unknown = dict(DEFAULT_DEPENDENCIES)
    unknown["A2"] = ("UNKNOWN",)
    with pytest.raises(ValueError, match="unknown dependency"):
        topological_order(unknown)


def test_queue_inventory_mutation_fails_closed(tmp_path):
    root, queue = _copy_fixture(tmp_path)
    del queue["groups"]["B"]["tasks"]["B2"]
    _write_queue(root, queue)
    with pytest.raises(ValueError, match="inventory/order"):
        _build(root)


def test_queue_evidence_status_mismatch_fails_closed(tmp_path):
    root, queue = _copy_fixture(tmp_path)
    queue["groups"]["B"]["tasks"]["B5"]["status"] = "STATIC_PASS"
    _write_queue(root, queue)
    with pytest.raises(ValueError, match="queue/evidence status mismatch"):
        _build(root)


def test_non_dynamic_authorization_claim_fails_closed(tmp_path):
    root, queue = _copy_fixture(tmp_path)
    queue["groups"]["C"]["tasks"]["C1"]["execution_authorized"] = True
    _write_queue(root, queue)
    with pytest.raises(ValueError, match="non-dynamic task claims"):
        _build(root)


def test_changed_root_blocker_classification_fails_closed(tmp_path):
    root, _ = _copy_fixture(tmp_path)
    path = root / "artifacts/agent_control/tasks/EIGHT-HOUR-B1-GRASP-REGION-COM/TASK_RESULT.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["classification"] = "CHANGED"
    path.write_text(json.dumps(document), encoding="utf-8")
    ledger_path = root / BLOCKER_LEDGER
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    for row in rows:
        if row.get("task_id") == "B1":
            row["classification"] = "CHANGED"
    ledger_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="classification is absent or changed"):
        _build(root)


def test_renderers_and_immutable_triplet(tmp_path):
    report = _build()
    markdown = render_markdown(report)
    dot = render_dot(report)
    assert markdown.startswith("# 动态就绪依赖图\n\n| 节点 |")
    assert "STATIC_PASS/OFFLINE_PASS 仅表示实现证据" in markdown
    assert '"A1" -> "A2";' in dot
    paths = [tmp_path / "report.json", tmp_path / "report.dot", tmp_path / "report.md"]
    write_report_triplet(report, *paths)
    assert json.loads(paths[0].read_text())["formal_dynamic_pass_count"] == 0
    with pytest.raises(FileExistsError, match="immutable"):
        write_report_triplet(report, *paths)


def test_builder_has_no_runtime_or_acceptance_side_effect_claims():
    report = _build()
    assert report["simulation_started_by_builder"] is False
    assert report["robot_commands_emitted_by_builder"] == 0
    assert report["assembly_success_claimed"] is False
    assert report["control_authorized"] is False
    assert report["formal_r12_generated"] is False
    assert report["hardware_authorized"] is False
    assert all(len(item["sha256"]) == 64 for item in report["source_manifest"].values())
