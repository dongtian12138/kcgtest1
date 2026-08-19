#!/usr/bin/env python3
"""Wrap the compliant insertion loop in a bounded C2 yaw branch retry."""
from pathlib import Path

path = Path("/home/noob/WorkPlace/kcgtest1/src/kcg_connector/isaac/d38999_compliant_capture_sweep.py")
text = path.read_text(encoding="utf-8")

loop_start_anchor = '        for _ in range(maximum_steps):\n            measured_tcp = last_sample["tcp"]'
loop_end_anchor = '            if measured_from_preinsert >= planned_target_progress:\n                nominal_terminal = "PLANNED_DEPTH_REACHED"\n                break\n'

si = text.find(loop_start_anchor)
if si < 0:
    raise SystemExit("loop start anchor not found")
ei = text.find(loop_end_anchor, si)
if ei < 0:
    raise SystemExit("loop end anchor not found")
ei_end = ei + len(loop_end_anchor)
loop_block = text[si:ei_end]

indented = "\n".join(("    " + line) if line.strip() else line for line in loop_block.split("\n"))

preamble = (
    "        insertion_attempt = 0\n"
    "        nominal_terminal = None\n"
    "        request_soft_backoff = False\n"
    "        c2_branch_retry = {\"attempted\": False, \"rotated_rad\": 0.0, \"reason\": None}\n"
    "        while True:\n"
)

retry_block = (
    "            if (\n"
    "                request_soft_backoff\n"
    "                and insertion_attempt == 0\n"
    "                and arguments.control_mode == \"compliant\"\n"
    "                and measured_from_preinsert < 0.5 * planned_target_progress\n"
    "            ):\n"
    "                backoff_speed = float(controller_config[\"recovery\"][\"backoff_speed_m_s\"])\n"
    "                backoff_distance = float(controller_config[\"recovery\"][\"backoff_distance_m\"])\n"
    "                backoff_steps = max(1, int(math.ceil(backoff_distance / (backoff_speed * dt))))\n"
    "                for _ in range(backoff_steps):\n"
    "                    controller_command_tcp[:3, 3] -= initial_axis * backoff_speed * dt\n"
    "                    command_arm = np.asarray(\n"
    "                        solve_fixed_q7_tcp_pose(\n"
    "                            tuple(float(value) for value in command_arm),\n"
    "                            tuple(float(value) for value in controller_command_tcp[:3, 3]),\n"
    "                            target_rotation=controller_command_tcp[:3, :3],\n"
    "                            maximum_iterations=8,\n"
    "                        )\n"
    "                    )\n"
    "                    last_sample = observe_loaded(\n"
    "                        \"mixed_grip_physical_insert_branch_retry_retract\",\n"
    "                        command_arm,\n"
    "                    )\n"
    "                axis = initial_axis / np.linalg.norm(initial_axis)\n"
    "                k = np.array(\n"
    "                    [\n"
    "                        [0.0, -axis[2], axis[1]],\n"
    "                        [axis[2], 0.0, -axis[0]],\n"
    "                        [-axis[1], axis[0], 0.0],\n"
    "                    ]\n"
    "                )\n"
    "                flip = np.eye(3) + 2.0 * (k @ k)\n"
    "                controller_command_tcp[:3, :3] = (\n"
    "                    flip @ controller_command_tcp[:3, :3]\n"
    "                )\n"
    "                controller_state = ControllerState(\n"
    "                    phase=InsertionState.GUARDED_APPROACH\n"
    "                )\n"
    "                previous_rigid_twist = np.zeros(6, dtype=np.float64)\n"
    "                insertion_attempt += 1\n"
    "                c2_branch_retry = {\n"
    "                    \"attempted\": True,\n"
    "                    \"rotated_rad\": float(math.pi),\n"
    "                    \"reason\": nominal_terminal,\n"
    "                }\n"
    "                nominal_terminal = None\n"
    "                request_soft_backoff = False\n"
    "                continue\n"
    "            break\n"
)

new_block = preamble + indented + "\n" + retry_block
text = text[:si] + new_block + text[ei_end:]
path.write_text(text, encoding="utf-8")
print("patched loop region")
