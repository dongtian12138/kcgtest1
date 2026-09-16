#!/usr/bin/env python3
"""Check or rerun the preserved 2026-09-16 assembly on this workstation.

The default only checks files and prints commands. --run starts a fresh
preflight, then one bounded simulation. This is not a portable GitHub bundle
and does not turn a process exit code into a physical success verdict.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/full_rotation_20260914/visual_integration"
REFERENCE = BASE / "visual_complete_all_reserves14"
MOTION_FLAGS = (
    "--visual-body-start", "--postgrasp-key-observation",
    "--body-assembly-transport", "--body-key-entry",
    "--body-support-test", "--body-nut-regrasp",
)


def read_json(path):
    return json.loads(path.read_text())


def replace_value(command, flag, value):
    if command.count(flag) != 1:
        raise ValueError(f"基线命令中 {flag} 的数量不正确")
    command[command.index(flag) + 1] = str(value)


def check_reference():
    invocation = read_json(REFERENCE / "run_invocation.json")
    recorded_root = Path(invocation["working_directory"])
    if ROOT != recorded_root:
        raise ValueError("此入口依赖原本机目录；跨目录/跨机器复现包尚未整理，不能悄悄使用旧目录资产")
    manifest = read_json(REFERENCE / "execution_source_snapshots/manifest.json")
    evaluation = read_json(REFERENCE / "evaluation.json")
    binding = evaluation["evidence_binding"]
    expected = {
        str(Path(row["source_path"]).relative_to(recorded_root)): row["sha256"]
        for row in manifest["files"]
    }
    expected.update(binding["scene_evidence_sha256"])
    command = read_json(BASE / "visual_complete_all_reserves14_command.json")
    expected[command[command.index("--config") + 1]] = binding["config_sha256"]
    expected["src/kcg_connector/config/carts_v2_isaac_runtime.json"] = binding["runtime_resources_sha256"]
    expected[command[command.index("--robot-asset") + 1]] = evaluation["pad_surface_identity_evidence"]["robot_asset_sha256"]
    changed = []
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    if changed:
        raise ValueError("以下文件缺失或与记录版本不同；不会覆盖你的修改：\n" + "\n".join(changed))
    if command[:2] != ["src/kcg_connector/isaac/run_isaac_python.sh",
                       "src/kcg_connector/isaac/run_body_assembly_with_video.py"]:
        raise ValueError("不是预期的本机 Isaac Sim 入口")
    runtime = Path(os.environ.get("ISAAC_ENV_PREFIX", str(ROOT.parent / "isaacsim/.conda-env")))
    if not os.access(runtime / "bin/python", os.X_OK):
        raise ValueError(f"找不到 Isaac Sim Python：{runtime / 'bin/python'}")
    print(f"已核对 {len(expected)} 个记录绑定文件，与基线一致。")
    print(f"Isaac Sim 环境：{runtime}")
    print("这不等于完整依赖包已可移植，也不保证每次物理结果完全相同。")
    return command, len(expected)


def build_commands(original, output, gui):
    motion = list(original)
    replace_value(motion, "--output-directory", output / "run")
    replace_value(motion, "--preflight-evaluation", output / "preflight/evaluation.json")
    preflight = list(motion)
    replace_value(preflight, "--mode", "preflight")
    replace_value(preflight, "--output-directory", output / "preflight")
    index = preflight.index("--preflight-evaluation")
    del preflight[index:index + 2]
    for flag in MOTION_FLAGS:
        preflight.remove(flag)
    if gui:
        preflight.append("--gui")
        motion.append("--gui")
    return preflight, motion


def execute(command, output, label, limit, reserve):
    env = os.environ.copy()
    env.pop("KCG_BOUNDED_EXPERIMENT", None)
    env.pop("KCG_EXPERIMENT_ACTION_DEADLINE", None)
    env.update(KCG_EXPERIMENT_WALL_LIMIT_S=str(limit),
               KCG_EXPERIMENT_CLOSEOUT_RESERVE_S=str(reserve))
    log = output / f"{label}.log"
    print(f"开始 {label}；日志：{log}", flush=True)
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            print("正在通知原有有界包装器中止并收尾，请等待日志保存。", flush=True)
            child.send_signal(signal.SIGINT)
            code = child.wait()
    print(f"{label} 退出码：{code}", flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="只核对并打印命令；默认行为")
    action.add_argument("--run", action="store_true", help="实际执行新预检和一次完整仿真")
    parser.add_argument("--gui", action="store_true", help="给预检和正式运行都启用 Isaac Sim 窗口")
    parser.add_argument("--output-root", type=Path, help="新的结果目录，必须尚不存在")
    args = parser.parse_args(argv)
    original, checked = check_reference()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_root or ROOT / "artifacts/reproductions" / f"current_hand_{stamp}"
    output = output.resolve()
    if output.exists():
        raise ValueError(f"结果目录已存在，拒绝覆盖：{output}")
    preflight, motion = build_commands(original, output, args.gui)
    print(f"新结果目录：{output}")
    for label, command, limit, reserve in (("预检", preflight, 300, 30),
                                          ("完整装配", motion, 18000, 1200)):
        print(f"\n{label}命令：\nKCG_EXPERIMENT_WALL_LIMIT_S={limit} "
              f"KCG_EXPERIMENT_CLOSEOUT_RESERVE_S={reserve} {shlex.join(command)}")
    if not args.run:
        print("\n仅检查完成，没有创建结果目录、启动物理实验或修改原运行记录。")
        return 0
    if args.gui and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("--gui 需要图形桌面环境；当前没有 DISPLAY/WAYLAND_DISPLAY")
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "reproduction_plan.json"
    plan = {"source_run": str(REFERENCE), "checked_bound_files": checked,
            "gui": args.gui, "gui_variant_newly_requested": args.gui,
            "preflight_command": preflight, "motion_command": motion,
            "simulation_only": True, "hardware_authorized": False,
            "physical_success_claimed": False, "status": "PREFLIGHT_RUNNING"}
    def save():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    save()
    code = execute(preflight, output, "preflight", 300, 30)
    result_file = output / "preflight/evaluation.json"
    result = read_json(result_file) if result_file.is_file() else {}
    checks = {key: result.get(key) is True for key in (
        "accepted_preflight_pass", "controller_preflight_pass",
        "engine_health_pass", "identity_hash_check_pass")}
    plan.update(preflight_returncode=code, preflight_checks=checks)
    if code != 0 or not all(checks.values()):
        plan["status"] = "PREFLIGHT_FAILED_NO_ASSEMBLY_STARTED"
        save()
        print("预检未通过，没有启动完整装配。请查看预检日志与原始评估。")
        return code if code > 0 else 2
    plan["status"] = "ASSEMBLY_RUNNING"
    save()
    code = execute(motion, output, "assembly", 18000, 1200)
    plan.update(status="PROCESS_ENDED_REQUIRES_PHYSICAL_REVIEW", returncode=code)
    save()
    release_file = output / "run/socket_transport/nut_terminal_release/nut_reindex_controller_result.json"
    release = read_json(release_file) if release_file.is_file() else {}
    if release.get("completed") is True and release.get("outer_abort_reason") is None:
        print("本轮有已完成的最终松手控制记录；实际到位和保持仍需原始物理数据核验。")
    else:
        print("本轮没有完成的最终松手控制记录，请先检查装配日志中的停止原因。")
    video = output / "run/video/assembly_four_view.mp4"
    if video.exists():
        print(f"本轮录制文件：{video}")
    else:
        print(f"本轮尚未生成录制文件，请查看：{output / 'assembly.log'}")
    print("退出码不能代替物理验收。需检查新一轮实际深度、真实松手及源面/键槽；不复用旧回合的通过结论。")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"本机复现入口停止：{error}", file=sys.stderr)
        raise SystemExit(2)
