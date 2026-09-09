#!/usr/bin/env python3
"""Make a labelled review copy from one completed episode's camera recording."""

import argparse
import json
from pathlib import Path
import subprocess


EXACT = {
    "ft_free_space_tare": "传感器基线", "settle": "预抓准备",
    "preshape_at_home": "预抓准备", "approach_above": "接近插头",
    "wait_above_settled": "接近插头", "approach_descent": "下降至预抓位姿",
    "pregrasp_hold": "预抓位姿", "tare": "关节力矩基线",
    "prelift_effort_check": "抬升前抓持确认",
    "lift": "抬升插头", "hold": "抓持保持",
    "key_probe_body_carry_observation": "搬运至观察位置",
    "key_probe_body_side_observation": "侧面观察插座键槽",
    "key_probe_body_transport": "搬运至插座",
    "key_probe_body_precontact": "接近键槽",
    "key_probe_contact": "低力导键进给",
    "key_probe_body_support_unload": "承载确认后卸力",
    "key_probe_body_support_open": "张指",
    "key_probe_body_support_hold": "脱手保持",
    "key_probe_nut_open": "打开至螺母预抓角",
    "key_probe_nut_open_hold": "开手保持",
    "key_probe_nut_transfer": "移至螺母预抓位置",
    "key_probe_nut_transfer_hold": "螺母预抓位置保持",
    "key_probe_nut_tare": "换抓前力矩基线",
    "key_probe_nut_contact": "指腹闭合夹持螺母",
    "key_probe_nut_grip_hold": "螺母抓持保持",
    "key_probe_nut_rotation_axial_settle": "旋拧前轴向力调整",
    "key_probe_nut_rotation_turn": "机械臂旋拧螺母",
    "key_probe_nut_rotation_hold": "旋拧后保持",
}


def phase_label(phase):
    if phase in EXACT:
        return EXACT[phase]
    for prefix, label in (("key_probe_body_align", "视觉对键"),
                          ("parallel_contact", "三指闭合抓持本体"),
                          ("finger_", "本体抓持"), ("preload", "本体抓持预载")):
        if phase.startswith(prefix):
            return label
    raise ValueError(f"an explicit stage caption is required for {phase}")


def ass_time(seconds):
    value = round(seconds * 100)
    return f"{value//360000}:{value//6000%60:02d}:{value//100%60:02d}.{value%100:02d}"


def label(directory):
    summary = json.loads((directory / "assembly_four_view_video.json").read_text())
    frames = [json.loads(line) for line in (directory / "assembly_four_view_frames.jsonl").read_text().splitlines()]
    fps = summary["fps"]
    if len(frames) != summary["frame_count"] or [row["frame"] for row in frames] != list(range(len(frames))):
        raise ValueError("the frame index does not cover the complete encoded recording")
    captions = [phase_label(row["phase"]) for row in frames]
    output = directory / "assembly_four_view_CN.mp4"
    subtitle = directory / "assembly_four_view_CN.ass"
    if output.exists() or subtitle.exists():
        raise FileExistsError("refusing to overwrite a labelled video")
    lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
             "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
             "Style: Default,Noto Sans CJK SC,28,&H00FFFFFF,&H00FFFFFF,&H90000000,&H90000000,0,0,0,0,100,100,0,0,3,2,0,7,0,0,0,1", "",
             "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    def add(start, end, x, y, text):
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{{\\pos({x},{y})}}{text}")
    duration = len(frames) / fps
    for x, y, text in ((24, 20, "机械臂与装配区域"), (1456, 10, "全局相机"),
                       (1456, 370, "掌心相机"), (1456, 730, "手腕相机")):
        add(0, duration, x, y, text)
    start = 0
    for index in range(1, len(frames)+1):
        if (index == len(frames) or captions[index] != captions[start]
                or round(frames[index]["world_time_before_render_s"], 1)
                != round(frames[start]["world_time_before_render_s"], 1)):
            timestamp = frames[start]["world_time_before_render_s"]
            add(start/fps, index/fps, 24, 1012, f"{captions[start]}  |  仿真时间 {timestamp:.1f} s")
            start = index
    subtitle.write_text("\n".join(lines)+"\n", encoding="utf-8")
    # Work in the directory so the filter contains only a fixed, safe basename.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", "assembly_four_view.mp4",
                    "-vf", "ass=assembly_four_view_CN.ass", "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output.name],
                   cwd=directory, check=True)
    (directory / "labelled_review_copy.json").write_text(json.dumps({
        "original_video": summary["path"], "labelled_copy": str(output),
        "phase_source": "same-episode frame index", "timestamp_source": "recorded physics clock before rendering",
        "original_video_modified": False, "frames_reordered_or_interpolated": False,
        "captions_name_controller_actions_not_success_verdicts": True,
    }, ensure_ascii=False, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    label(parser.parse_args().directory.resolve())
