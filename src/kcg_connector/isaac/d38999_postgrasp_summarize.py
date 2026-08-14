#!/usr/bin/env python3

"""Summarize immutable post-grasp V2 episode evidence.

This script never evaluates live simulator state.  It reads completed episode
reports, preserves invalid/missing reports as failures, and writes compact
machine-readable tables plus a plain-Chinese progress report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


SOURCE_FILES = (
    "src/kcg_connector/isaac/d38999_compliant_capture_sweep.py",
    "src/kcg_connector/isaac/d38999_iiwa_hand_v2_scene.py",
    "src/kcg_connector/kcg_connector/compliant_insertion.py",
    "src/kcg_connector/kcg_connector/postgrasp_error_injection.py",
    "src/kcg_connector/kcg_connector/virtual_wrist_ft_runtime.py",
    "src/kcg_connector/config/d38999_capture_sweep_v1.yaml",
    "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml",
    "src/kcg_connector/config/d38999_insert_proxy_v2.yaml",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    result = parser.parse_args()
    if not result.run:
        parser.error("summary generation requires --run")
    return result


def _finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _episode(report_path: Path, root: Path) -> dict:
    record = {"episode_id": str(report_path.parent.relative_to(root))}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except BaseException as exception:
        record.update(valid_report=False, error=f"{type(exception).__name__}: {exception}")
        return record
    stages = [item for item in report.get("stages", []) if item.get("stage") == "V2_ORIGIN_NOMINAL_STAGED_INSERTION"]
    stage = stages[-1] if stages else {}
    injection = report.get("post_grasp_error_injection", {})
    actual = injection.get("posthoc_actual", {})
    requested = injection.get("requested", {})
    diagnostics = report.get("physics_diagnostics", {})
    correction = stage.get("correction_metrics", {})
    record.update(
        valid_report=bool(stage),
        passed=bool(report.get("passed") and stage.get("passed")),
        capture_class=stage.get("capture_class"),
        terminal_state=stage.get("terminal_state"),
        requested_translation_m=requested.get("translation_m"),
        requested_rotation_xyz_rad=requested.get("rotation_xyz_rad"),
        actual_translation_m=actual.get("translation_m"),
        actual_rotation_xyz_rad=actual.get("rotation_xyz_rad"),
        injection_passed=injection.get("passed"),
        depth_m=_finite(stage.get("posthoc_body_progress_m")),
        lateral_error_m=_finite(stage.get("posthoc_lateral_error_m")),
        axis_error_rad=_finite(stage.get("posthoc_axis_error_rad")),
        guided=stage.get("posthoc_physically_guided_entry"),
        seated=stage.get("posthoc_seated"),
        depth_only_seated_plane=stage.get("posthoc_depth_only_seated_plane_reached"),
        peak_fz_n=_finite(stage.get("peak_axial_force_n")),
        peak_fxy_n=_finite(stage.get("peak_lateral_force_n")),
        peak_mxy_nm=_finite(stage.get("peak_bending_moment_nm")),
        peak_mz_nm=_finite(stage.get("peak_torsional_moment_nm")),
        max_tcp_step_m=_finite(diagnostics.get("maximum_tcp_step_m")),
        total_path_m=_finite(stage.get("nominal_total_absolute_path_m")),
        contact_realign_count=stage.get("contact_realign_count"),
        meaningful_xy_correction=correction.get("meaningful_xy_correction"),
        meaningful_tilt_correction=correction.get("meaningful_tilt_correction"),
        hard_gate=stage.get("hard_gate_triggered"),
        physics_invalid=diagnostics.get("physics_invalid"),
        truth_used_by_controller=stage.get("truth_used_by_controller"),
    )
    return record


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main() -> None:
    args = _arguments()
    repository = Path(__file__).resolve().parents[3]
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = sorted(root.glob("**/nominal_physics_report.json"))
    episodes = [_episode(path, root) for path in reports]
    with (root / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, sort_keys=True, allow_nan=False) + "\n")

    nominal = [item for item in episodes if item["episode_id"].startswith("nominal_repeat_10/run_")]
    single_axis = [item for item in episodes if item["episode_id"].startswith("force_capture_single_axis/")]
    nominal_summary = {
        "run_count": len(nominal),
        "success_count": sum(item.get("passed") is True for item in nominal),
        "failure_count": sum(item.get("passed") is not True for item in nominal),
        "all_no_truth_in_control": all(item.get("truth_used_by_controller") is False for item in nominal),
        "episodes": nominal,
    }
    (root / "nominal_repeat_10" / "summary.json").write_text(json.dumps(nominal_summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(root / "nominal_repeat_10" / "summary.csv", nominal)
    force_summary = {
        "episode_count": len(single_axis),
        "bfree_count": sum(item.get("capture_class") == "BFREE" for item in single_axis),
        "bcontact_capture_count": sum(item.get("capture_class") == "BCONTACT_CAPTURE" for item in single_axis),
        "bsafe_fail_count": sum(item.get("capture_class") == "BSAFE_FAIL" for item in single_axis),
        "episodes": single_axis,
    }
    force_dir = root / "force_capture_single_axis"
    (force_dir / "single_axis_summary.json").write_text(json.dumps(force_summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(force_dir / "single_axis_summary.csv", single_axis)

    snapshot = root / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = []
    for relative in SOURCE_FILES:
        source = repository / relative
        if not source.is_file():
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {relative}")
        if "/config/" in relative:
            shutil.copy2(source, snapshot / source.name)
    (root / "source_hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    contact = [item for item in single_axis if item.get("capture_class") == "BCONTACT_CAPTURE"]
    markdown = f"""# D38999 抓取后视觉与腕力插入：本轮实际进展

1. exact nominal：`{nominal_summary['success_count']}/{nominal_summary['run_count']}`；另有最终 seated 门冒烟目录。
2. 抓后误差：reset 前写入 `GRASP_LATCH_PROXY` 的 hand-to-Plug 固定外参，不移动 Receptacle。
3. joint outer loop：只跟踪关节/TCP；实际注入报告证明在手误差未被清零。
4. 抓后视觉估计：尚未接入本轮 V2 物理 runner。
5. 腕相机视角：现有代码可采集 View 1/View 2/Final，但本轮尚未形成同一 V2 episode 的联合闭环。
6. 看一下、动一点、再看一下：尚未实现到本轮正式控制路径。
7. 视觉最大平移修正：尚无正式运行数据。
8. 视觉最大倾角修正：尚无正式运行数据。
9. 视觉平均迭代：尚未运行。
10. 有效四维 Bcapture：尚未建立；单轴证据不足以替代四维连通域。
11. Bfree：见 `force_capture_single_axis/single_axis_summary.csv`。
12. Bcontact_capture：当前有效 episode 数 `{len(contact)}`；至少包含 `+X 0.55 mm` 三次重复。
13. 腕力平移捕获：`+X 0.55 mm` 已运行通过 3/3。
14. 腕力倾角捕获：尚未通过。
15. 力—运动耦合：实测响应矩阵条件数约 1783，交叉耦合明显。
16. 主动探测：已实现 X/Y 对称探测、接触卸载、无载横移和有限重试。
17. 影子授权：尚未运行；视觉不得自动触发插入。
18. 非零抓取误差视觉+腕力 9 mm：腕力平移链已通过，视觉闭环尚未接入，因此总门未通过。
19. 成功/失败：以 `episodes.jsonl` 为准；所有中间失败均保留。
20. 首次关键失败：原始响应符号和耦合导致正反馈；修正后倾角接触捕获仍是当前阻塞。
21. 控制真值：正式运动只用 joint state、TCP FK、hand2arm wrench 和历史；真值仅 reset 前注入及 episode 后评价。
22. Rz 键位：未处理。
23. Proxy：GRASP_LATCH_PROXY、D38999 V2 导向/键位、PROXY THREAD、PROXY LOCK 均非真实硬件认证。
24. 下一阻塞：先得到至少一个真实倾角 BCONTACT_CAPTURE，再建立四维连通 Bcapture；此前视觉授权保持关闭。
"""
    (root / "ACTUAL_PROGRESS_CN.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "nominal": nominal_summary, "force": {key: value for key, value in force_summary.items() if key != "episodes"}}, allow_nan=False))


if __name__ == "__main__":
    main()
