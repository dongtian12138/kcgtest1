"""Clarify historical visual readings on the same closed five-view recording.

Only the existing header is covered. Camera imagery and original recording
remain preserved; captions use recorded online readings, never object truth.
"""
import argparse,json,subprocess
from pathlib import Path


def phase_label(phase):
    exact={
        'ft_free_space_tare':'传感器归零',
        'initial_rgbd_settle':'初始相机观测准备',
        'initial_visual_planning_latency_hold':'初始视觉与规划等待',
        'settle':'初始保持',
        'preshape_at_home':'手指预张开',
        'approach_above':'接近插头上方',
        'wait_above_settled':'接近后保持',
        'approach_descent':'下探至抓取位置',
        'pregrasp_hold':'抓取前保持',
        'tare':'抓取前归零',
        'parallel_contact_approach':'建立指尖接触',
        'preload':'建立夹持预载',
        'prelift_effort_check':'抬升前夹持检查',
        'lift':'抬升插头',
        'hold':'抬升后保持',
        'key_probe_body_to_fixed_key_view':'搬运到固定相机2观测位',
        'key_probe_observation_latency_hold':'第一次键图像处理等待',
        'key_probe_wrist_latency_hold':'腕部槽图像处理等待',
        'key_probe_body_coarse_axial_alignment':'依据视觉角差进行绕轴粗调',
        'key_probe_refined_observation_latency_hold':'第二次键图像处理等待',
        'key_probe_body_socket_centering':'精调并短距离移到插座上方',
        'key_probe_body_align_1':'插入前精对准',
        'key_probe_body_align_2':'插入前精对准',
        'key_probe_body_precontact':'到达预接触位置',
        'key_probe_contact':'低力插入键槽',
        'key_probe_body_support_unload':'卸载本体夹持',
        'key_probe_body_support_open':'张手释放本体',
        'key_probe_body_support_hold':'观察入槽后的本体支撑',
        'key_probe_nut_transfer':'移到螺母抓取位置',
        'key_probe_nut_tare':'螺母夹持前归零',
        'key_probe_nut_contact':'建立螺母夹持',
        'key_probe_nut_grip_hold':'检查螺母夹持',
        'key_probe_nut_rotation_turn':'旋拧螺母',
        'key_probe_nut_index_unload':'卸载螺母夹持',
        'key_probe_nut_index_open':'张手准备换抓',
        'nut_index_free_rotate':'张手回转以便再次抓取',
        'nut_index_free_open_hold':'张手保持观测',
        'nut_index_free_final_hold':'回转后保持',
    }
    if phase in exact:return exact[phase]
    if 'palm' in phase:return '等待掌心位置和轴线反馈'
    if 'planning' in phase or 'preparation' in phase:return '规划与几何计算后的物理保持'
    if 'nut' in phase:return '螺母操作准备与保持'
    return '受保护的阶段保持'


def stamp(seconds):
    centiseconds=round(seconds*100)
    h,r=divmod(centiseconds,360000);m,r=divmod(r,6000);s,cs=divmod(r,100)
    return f'{h}:{m:02d}:{s:02d}.{cs:02d}'


def label(run,output=None):
    run=Path(run).resolve();directory=run/'video'
    metadata=json.loads((directory/'assembly_five_view_video.json').read_text())
    if metadata.get('ffmpeg_return_code')!=0:raise ValueError('A closed original video is required')
    rows=[json.loads(line) for line in (directory/'assembly_five_view_frames.jsonl').read_text().splitlines()]
    if len(rows)!=metadata['frame_count'] or any(r['frame']!=i for i,r in enumerate(rows)):
        raise ValueError('Video frames and online caption records must match exactly')
    destination=Path(output).resolve() if output else directory/'assembly_five_view_CN.mp4'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    subtitle=destination.with_suffix('.ass')
    header='''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Header,Noto Sans CJK SC,23,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Reading,Noto Sans CJK SC,23,&H00C8EB78,&H00C8EB78,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines=[header];fps=metadata['fps']
    for index,row in enumerate(rows):
        status=row.get('online_visual_and_encoder_status',{})
        q=status.get('measured_joint7_deg')
        title=f"主视角（仅录像）  仿真时间 {row['simulation_time_s']:.1f} 秒  ｜ {phase_label(row['phase'])}"
        if q is not None:title+=f'  ｜ 第七关节 {q:.2f}°'
        count=status.get('observation_count',0)
        reading=f'已采用键位图像：{count}/2'
        if 'angle_deg' in status:
            name='第一次视觉粗转指令' if status['stage']=='coarse' else '第二次视觉更新后的修正角'
            reading+=f"   {name} {status['angle_deg']:+.3f}°（历史读数，键图 t={status['sample_time_s']:.2f} 秒）"
        elif count:reading+=f"   键图 t={status['sample_time_s']:.2f} 秒，等待角差计算"
        else:reading+='   尚未采用键角测量'
        for style,y,text in [('Header',3,title),('Reading',35,reading)]:
            lines.append(f'Dialogue: 0,{stamp(index/fps)},{stamp((index+1)/fps)},{style},,0,0,0,,{{\\pos(14,{y})}}{text}\n')
    subtitle.write_text(''.join(lines),encoding='utf-8')
    command=['ffmpeg','-hide_banner','-loglevel','warning','-n','-i',str(directory/'assembly_five_view.mp4'),
        '-vf',f'drawbox=x=0:y=0:w=1440:h=68:color=0x0c1218:t=fill,ass={subtitle.name}',
        '-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    subprocess.run(command,cwd=destination.parent,check=True)
    manifest={'source':str(directory/'assembly_five_view.mp4'),'output':str(destination),
        'caption_source':str(directory/'assembly_five_view_frames.jsonl'),
        'frame_count':len(rows),'fps':fps,'edited_region':'EXISTING_MAIN_VIEW_HEADER_ONLY_0_0_1440_68',
        'original_video_preserved':True,'caption_values_from_object_truth':False,
        'historical_key_readings_explicitly_labeled':True}
    destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output');a=p.parse_args();label(a.run,a.output)
