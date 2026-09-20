#!/usr/bin/env python3
"""Check or replay this branch's assembly configuration with versioned assets and a fresh preflight.

--run starts simulation; --gui displays Isaac Sim. An exit code is not a physical verdict.
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
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reproducibility/four_camera_baseline_20260920"
ASSETS = ROOT / "reproducibility/assembly_20260916"
REFERENCE = ROOT / "reproducibility/high_global1_acceptance_20260919/evidence"
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
    try:
        subprocess.check_output(['git','rev-parse','--verify','HEAD'],cwd=ROOT,
                                text=True,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError) as error:
        raise ValueError('请按复现说明用 git clone 获取此分支，以保留源码版本和验收来源；没有启动物理') from error
    manifest = read_json(ASSETS / "runtime_assets.json")
    source = read_json(BASE / "portable_source_manifest.json")
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    expected.update({row['path']:row['sha256'] for row in read_json(BASE/'extra_runtime_inputs.json')['files']})
    expected.update(source["sha256"])
    changed = []
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    if changed:
        raise ValueError("缺失或与本版本不符的文件：\n" + "\n".join(changed))
    sys.path.insert(0, str(ROOT / "src/kcg_connector/isaac"))
    from te_runtime_paths import isaac_environment_prefix, sam6d_runtime, planner_python, ros_setup_bash
    sam_root, sam_python = sam6d_runtime(ROOT)
    executables = [isaac_environment_prefix() / "bin/python", sam_python, planner_python(ROOT)]
    for executable in executables:
        if not os.access(executable, os.X_OK):
            raise ValueError(f"缺少环境 Python：{executable}")
    for required in (ROOT / "install/setup.bash", ros_setup_bash()):
        if not required.is_file():
            raise ValueError(f"缺少 ROS 资源环境：{required}")
    sam = read_json(ASSETS / "sam6d.json")
    for row in sam["weights"]:
        path = sam_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"缺少视觉权重或大小不符：{path}；用 prepare_assembly_reproduction.py 校验/准备")
    print(f"已核对 {len(expected)} 个本版本源码与资产文件。")
    print(f"Isaac Python：{executables[0]}\n视觉 Python：{sam_python}\n规划 Python：{executables[2]}")
    print(f"视觉源码：{sam_root}；权重完整 SHA256 由准备脚本核验。")
    probe = subprocess.run([str(executables[0]), '-c',
        'import sys,importlib.metadata as m; '
        'sys.path[:0]=sys.argv[1:]; '
        'import _contact_copy_native,_contact_reports_native; '
        'from recording_compression import gzip_backend; gzip_backend("isal"); '
        'assert m.version("warp-lang")=="1.13.0"; '
        'print("原生记录模块、isal 与 Warp 1.13.0 已核对")',
        str(ROOT/'src/kcg_connector/isaac/carts_v2')], cwd=ROOT, text=True)
    if probe.returncode:
        raise ValueError('请按本分支复现说明安装记录依赖并编译两个原生模块')
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():
        raise ValueError('五视角录像需要 fonts-noto-cjk 中文字体')
    return read_json(BASE / "assembly_command.json"), len(expected)


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
               KCG_EXPERIMENT_CLOSEOUT_RESERVE_S=str(reserve), OPENBLAS_NUM_THREADS='1')
    log = output / f"{label}.log"
    print(f"开始 {label}；日志：{log}", flush=True)
    with log.open("x") as stream:
        started=time.monotonic()
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 stdout=stream, stderr=subprocess.STDOUT)
        process_path=output/('process.json' if label=='assembly' else 'preflight_process.json')
        process={'pid':child.pid,'argv':command,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'HIGH_GLOBAL1_FOUR_CAMERA_BASELINE_'+label.upper(),
            'simulation_only':True,'hardware_authorized':False}
        process_path.write_text(json.dumps(process,indent=2)+'\n')
        try:
            code = child.wait()
        except KeyboardInterrupt:
            print("正在通知原有有界包装器中止并收尾，请等待日志保存。", flush=True)
            run_directory=output/('run' if label=='assembly' else 'preflight')
            if run_directory.is_dir():
                (run_directory/'STOP_REQUEST').write_text('Requested by reproduction launcher\n')
            else:
                child.send_signal(signal.SIGINT)
            code = child.wait()
        process.update(exit_code=code,wall_s=time.monotonic()-started,
                       ended_utc=datetime.now(timezone.utc).isoformat())
        process_path.write_text(json.dumps(process,indent=2)+'\n')
    print(f"{label} 退出码：{code}", flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="只核对并打印命令；默认行为")
    action.add_argument("--run", action="store_true", help="实际执行新预检和一次完整仿真")
    action.add_argument("--preflight-only", action="store_true", help="只执行独立新预检")
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
                                          ("完整装配", motion, 21600, 300)):
        print(f"\n{label}命令：\nKCG_EXPERIMENT_WALL_LIMIT_S={limit} "
              f"KCG_EXPERIMENT_CLOSEOUT_RESERVE_S={reserve} {shlex.join(command)}")
    if not (args.run or args.preflight_only):
        print("\n仅检查完成，没有创建结果目录、启动物理实验或修改原运行记录。")
        return 0
    if args.gui and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("--gui 需要图形桌面环境；当前没有 DISPLAY/WAYLAND_DISPLAY")
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "reproduction_plan.json"
    configuration_id = read_json(BASE / "portable_source_manifest.json").get("configuration_id", "preserved-baseline")
    try:
        source_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    plan = {"source_run": str(REFERENCE), "configuration_id": configuration_id, "checked_bound_files": checked,
            "source_git_commit": source_commit,
            "source_manifest_sha256": hashlib.sha256((BASE / 'portable_source_manifest.json').read_bytes()).hexdigest(),
            "initial_pose_variation_requested": False,
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
    if args.preflight_only:
        plan["status"] = "PREFLIGHT_PASSED_ASSEMBLY_NOT_STARTED"
        save()
        return 0
    plan["status"] = "ASSEMBLY_RUNNING"
    save()
    code = execute(motion, output, "assembly", 21600, 300)
    plan.update(status="PROCESS_ENDED_REQUIRES_PHYSICAL_REVIEW", returncode=code)
    save()
    release_file = output / "run/socket_transport/nut_terminal_release/nut_reindex_controller_result.json"
    release = read_json(release_file) if release_file.is_file() else {}
    if release.get("completed") is True and release.get("outer_abort_reason") is None:
        print("本轮有已完成的最终松手控制记录；实际到位和保持仍需原始物理数据核验。")
    else:
        print("本轮没有完成的最终松手控制记录，请先检查装配日志中的停止原因。")
    video = output / "run/video/assembly_five_view.mp4"
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
        print(f"复现入口停止：{error}", file=sys.stderr)
        raise SystemExit(2)
