"""Deterministic integrity manifest over an explicit assembly evidence graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import yaml

from .assembly_evidence_report import EXPECTED_TASK_KEYS


SCHEMA_VERSION = "kcg_eight_hour_evidence_integrity_manifest_v1"
TASK_ID = "EIGHT-HOUR-G1-EVIDENCE-INTEGRITY-MANIFEST"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTROL_PATHS = (
    "AGENTS.md",
    "PLANS.md",
    "artifacts/agent_control/PROJECT_CHARTER_CN.md",
    "artifacts/agent_control/AUTONOMY_POLICY_CN.md",
    "artifacts/agent_control/WORK_QUEUE.yaml",
    "artifacts/agent_control/MASTER_STATE.json",
    "artifacts/agent_control/TASK_GRAPH.yaml",
    "artifacts/agent_control/CURRENT_TASK.md",
    "artifacts/agent_control/CURRENT_STATUS_CN.md",
    "artifacts/agent_control/EIGHT_HOUR_PROGRESS_CN.md",
    "artifacts/agent_control/BLOCKER_LEDGER.jsonl",
    "artifacts/agent_control/DECISION_LOG.jsonl",
    "artifacts/agent_control/GATE_LEDGER.csv",
    "artifacts/agent_control/STATUS_HISTORY.jsonl",
)
ADDITIONAL_PATHS = (
    "artifacts/agent_control/multilayer/HIGH_DETAIL_REFERENCE_MANIFEST.json",
    "artifacts/agent_control/multilayer/HIGH_DETAIL_BLOCKED_CONCLUSION_CN.md",
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/BUILD_RESULT.json",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json",
)
REFERENCE_KEYS = {
    "path",
    "evidence",
    "blocking_evidence",
    "review_bundle",
    "review_request",
    "task_result",
    "report",
    "mapping",
    "static_gates",
    "dynamic_result",
    "post_fix_report",
    "formal_analysis",
    "comparison",
    "contact_stats",
    "hard_penetrations",
    "raw_report",
    "h2_analysis",
    "build_result_path",
    "high_detail_reference_manifest",
    "high_detail_blocked_conclusion",
    "work_queue",
    "blocker_ledger",
    "progress_report",
}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".csv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty repository path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} absolute path is forbidden")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root")
    return path


def _strict_json(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, parse_constant=reject)


def parse_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = _strict_json(path.read_text(encoding="utf-8"))
        return {"kind": "JSON", "record_count": 1, "value": value}
    if suffix == ".jsonl":
        values = [
            _strict_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {"kind": "JSONL", "record_count": len(values), "value": values}
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {"kind": "YAML", "record_count": 1, "value": value}
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"CSV header is missing or duplicated: {path}")
            rows = list(reader)
        return {"kind": "CSV", "record_count": len(rows), "value": rows}
    raise ValueError(f"unsupported structured suffix: {path}")


def _looks_like_repository_path(value: Any) -> bool:
    return isinstance(value, str) and (
        value in {"AGENTS.md", "PLANS.md"}
        or value.startswith(("artifacts/", "src/", "tools/"))
    )


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mappings(item)


def validate_declared_hash_pairs(
    root: Path,
    document: Any,
) -> tuple[list[dict[str, str]], list[str]]:
    verified: list[dict[str, str]] = []
    unverified_code_paths: list[str] = []
    for mapping in _walk_mappings(document):
        for key, raw_digest in mapping.items():
            if key == "sha256":
                path_value = mapping.get("path")
            elif isinstance(key, str) and key.endswith("_sha256"):
                path_value = mapping.get(key[:-7])
            else:
                continue
            if isinstance(path_value, str) and Path(path_value).is_absolute():
                raise ValueError("declared hash path absolute path is forbidden")
            if not _looks_like_repository_path(path_value):
                continue
            if not isinstance(raw_digest, str) or SHA256_PATTERN.fullmatch(raw_digest) is None:
                raise ValueError(f"invalid declared SHA-256 for {path_value}")
            path = _resolve_inside(root, path_value, "declared hash path")
            if not path.is_file():
                raise ValueError(f"declared hash path is missing: {path_value}")
            actual = _sha256(path)
            if actual != raw_digest:
                raise ValueError(f"declared hash drift: {path_value}")
            verified.append(
                {"path": str(path.relative_to(root)), "sha256": actual}
            )
        rows = mapping.get("code_files")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("code_files row must be a mapping")
            path_value = row.get("path")
            if not _looks_like_repository_path(path_value):
                raise ValueError("code_files path is invalid")
            if row.get("sha256") is None:
                if not isinstance(row.get("validation_status"), str):
                    raise ValueError("unhashed code_files row lacks validation_status")
                path = _resolve_inside(root, path_value, "unverified code path")
                if not path.is_file():
                    raise ValueError(f"unverified code path is missing: {path_value}")
                unverified_code_paths.append(str(path.relative_to(root)))
    unique = {
        (row["path"], row["sha256"]): row for row in verified
    }
    return [unique[key] for key in sorted(unique)], sorted(set(unverified_code_paths))


def _reference_values(document: Any) -> list[str]:
    values: list[str] = []
    if isinstance(document, Mapping):
        for key, item in document.items():
            if key == "final_outputs":
                continue
            if key in REFERENCE_KEYS:
                candidates = item if isinstance(item, list) else [item]
                for candidate in candidates:
                    if _looks_like_repository_path(candidate):
                        values.append(candidate)
                    elif isinstance(candidate, Mapping):
                        path_value = candidate.get("path")
                        if _looks_like_repository_path(path_value):
                            values.append(path_value)
            values.extend(_reference_values(item))
    elif isinstance(document, list):
        for item in document:
            values.extend(_reference_values(item))
    return values


def _queue_evidence_paths(queue: Mapping[str, Any]) -> list[str]:
    groups = queue.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("work queue lacks groups")
    keys: list[str] = []
    paths: list[str] = []
    for group in groups.values():
        tasks = group.get("tasks") if isinstance(group, Mapping) else None
        if not isinstance(tasks, Mapping):
            raise ValueError("work queue group lacks tasks")
        for key, task in tasks.items():
            keys.append(str(key))
            if not isinstance(task, Mapping) or not _looks_like_repository_path(task.get("evidence")):
                raise ValueError(f"queue task {key} lacks repository evidence")
            paths.append(str(task["evidence"]))
    if tuple(keys) != EXPECTED_TASK_KEYS:
        raise ValueError("queue task inventory/order differs from G1 contract")
    return paths


def build_integrity_manifest(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    generated_at_utc: str,
    output_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be an explicit UTC timestamp")
    excluded = {
        _resolve_inside(root, value, "manifest output").relative_to(root).as_posix()
        for value in output_paths
    }
    roles: dict[str, set[str]] = {}
    pending: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}
    verified_pairs: dict[tuple[str, str], dict[str, str]] = {}
    unverified_code_paths: set[str] = set()

    def add(value: str, role: str) -> None:
        path = _resolve_inside(root, value, role)
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            raise ValueError(f"manifest output became an input: {relative}")
        if not path.is_file():
            raise ValueError(f"manifest input is missing: {relative}")
        if relative not in roles:
            roles[relative] = set()
            pending.append(relative)
        roles[relative].add(role)

    for value in CONTROL_PATHS:
        add(value, "mutable_control_snapshot")
    for value in ADDITIONAL_PATHS:
        add(value, "authoritative_or_frozen_input")
    queue_relative = _resolve_inside(root, work_queue_path, "work queue").relative_to(root).as_posix()
    add(queue_relative, "work_queue")

    cursor = 0
    while cursor < len(pending):
        relative = pending[cursor]
        cursor += 1
        path = root / relative
        if path.suffix.lower() not in STRUCTURED_SUFFIXES:
            continue
        parsed_row = parse_structured_file(path)
        parsed[relative] = {
            "kind": parsed_row["kind"],
            "record_count": parsed_row["record_count"],
        }
        document = parsed_row["value"]
        pairs, unverified = validate_declared_hash_pairs(root, document)
        for pair in pairs:
            verified_pairs[(pair["path"], pair["sha256"])] = pair
            add(pair["path"], "declared_hash_target")
        for item in unverified:
            unverified_code_paths.add(item)
            add(item, "parked_unverified_code_path")
        for reference in _reference_values(document):
            add(reference, "explicit_document_reference")

    queue = parsed[queue_relative]
    if queue["kind"] != "YAML":
        raise ValueError("work queue must parse as YAML")
    queue_document = yaml.safe_load((root / queue_relative).read_text(encoding="utf-8"))
    queue_paths = _queue_evidence_paths(queue_document)
    for value in queue_paths:
        if value not in roles:
            raise ValueError(f"queue evidence was not collected: {value}")

    high_detail_path = "artifacts/agent_control/multilayer/HIGH_DETAIL_REFERENCE_MANIFEST.json"
    high_detail = _strict_json((root / high_detail_path).read_text(encoding="utf-8"))
    baseline = high_detail.get("authoritative_high_detail_baseline", {})
    rejected = high_detail.get("rejected_local_variant_retained_as_evidence", {})
    if (
        high_detail.get("status") != "FROZEN_READ_ONLY_REFERENCE"
        or baseline.get("read_only") is not True
        or rejected.get("read_only") is not True
        or baseline.get("sha256") != "5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995"
        or rejected.get("sha256") != "d41477ee18052662904212444b907607874a8c6c27399d3d344e44ee4fd18d67"
    ):
        raise ValueError("high-detail freeze boundary changed")

    entries = []
    structured_counts: dict[str, int] = {}
    total_bytes = 0
    for relative in sorted(roles):
        path = root / relative
        size = path.stat().st_size
        total_bytes += size
        parse_row = parsed.get(relative)
        if parse_row:
            structured_counts[parse_row["kind"]] = structured_counts.get(parse_row["kind"], 0) + 1
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": size,
                "roles": sorted(roles[relative]),
                "structured_kind": parse_row["kind"] if parse_row else None,
                "structured_record_count": parse_row["record_count"] if parse_row else None,
            }
        )
    if any(row["path"] in excluded for row in entries):
        raise ValueError("manifest self-inclusion detected")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "scope_kind": "EXPLICIT_REFERENCE_GRAPH_NO_REPOSITORY_WALK",
        "entry_count": len(entries),
        "total_size_bytes": total_bytes,
        "queue_task_count": len(EXPECTED_TASK_KEYS),
        "queue_evidence_count": len(queue_paths),
        "declared_hash_pair_count": len(verified_pairs),
        "declared_hash_pairs_validated": True,
        "parked_unverified_code_paths": sorted(unverified_code_paths),
        "structured_file_counts": dict(sorted(structured_counts.items())),
        "structured_parse_failure_count": 0,
        "high_detail_frozen_asset_count": 2,
        "high_detail_baseline_sha256": baseline["sha256"],
        "rejected_local_variant_sha256": rejected["sha256"],
        "excluded_output_paths": sorted(excluded),
        "entries": entries,
        "simulation_started": False,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_integrity_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# 八小时窗口证据完整性清单",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 文件数 | {manifest['entry_count']} |",
        f"| 总字节数 | {manifest['total_size_bytes']} |",
        f"| 队列节点/证据 | {manifest['queue_task_count']}/{manifest['queue_evidence_count']} |",
        f"| 已验证声明摘要对 | {manifest['declared_hash_pair_count']} |",
        f"| 结构化解析失败 | {manifest['structured_parse_failure_count']} |",
        f"| 冻结高精细资产 | {manifest['high_detail_frozen_asset_count']} |",
        "",
        "## 边界",
        "",
        "- 清单来自显式引用图，不进行全仓库遍历。",
        "- 控制文件摘要是生成时点快照，后续状态交接会产生合法变化。",
        "- PARKED 代码若无冻结摘要，只记录路径存在，不提升为已验证。",
        "- 本清单不启动仿真，也不声明装配或正式 R12 通过。",
        "",
    ]
    return "\n".join(lines)


def write_integrity_outputs(
    manifest: Mapping[str, Any],
    json_output: str | Path,
    markdown_output: str | Path,
) -> None:
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("G1 integrity outputs are immutable")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_integrity_markdown(manifest), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("integrity generation requires --run")
    return args


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    json_output = _resolve_inside(root, args.json_output, "JSON output")
    markdown_output = _resolve_inside(root, args.markdown_output, "Markdown output")
    if not json_output.is_relative_to(output_root) or not markdown_output.is_relative_to(output_root):
        raise PermissionError("G1 outputs must remain inside the G1 task directory")
    manifest = build_integrity_manifest(
        repository_root=root,
        work_queue_path=args.work_queue,
        generated_at_utc=args.generated_at_utc,
        output_paths=(args.json_output, args.markdown_output),
    )
    write_integrity_outputs(manifest, json_output, markdown_output)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "build_integrity_manifest",
    "parse_structured_file",
    "render_integrity_markdown",
    "validate_declared_hash_pairs",
    "write_integrity_outputs",
]
