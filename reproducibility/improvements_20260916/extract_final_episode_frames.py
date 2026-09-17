#!/usr/bin/env python3
"""Extract actual final-turn/release frames; never synthesize a seated scene."""
import argparse
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args();run=args.run.resolve()
archive=run/'truth_samples.msgpack.gz';index=json.loads(Path(str(archive)+'.index.json').read_text())
if not (run/'motion_timing.json').is_file() or index['blocks'][-1]['end']!=archive.stat().st_size:
    raise ValueError('Only an ended episode with a sealed archive may be reviewed')
release=json.loads((run/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json').read_text())
if not release.get('completed'):raise ValueError('This episode did not complete final release')
frames=[json.loads(line) for line in (run/'video/assembly_four_view_frames.jsonl').read_text().splitlines()]
turn=next(row for row in reversed(frames) if row['phase'].startswith('key_probe_nut_rotation_'))
hold=[row for row in frames if release['opening_last_step']<=row['step']<release['support_hold_last_step'] and row['phase']=='nut_index_free_open_hold']
if not hold:raise ValueError('No actual final held-release frame')
chosen=[('rotation_end_before_release',turn),('released_hold',hold[0]),('final_released',hold[-1])]
output=run/'postrun_evidence';output.mkdir(exist_ok=True)
video=run/'video/assembly_four_view.mp4';records=[]
for label,row in chosen:
    if row.get('render_error') is not None or row['world_time_before_render_s']!=row['world_time_after_render_s']:
        raise ValueError('Selected frame is not a clean observation without physics advancement')
    image=output/(label+'.png')
    if image.exists():raise FileExistsError(image)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f"select=eq(n\\,{row['frame']})",'-vsync','0','-frames:v','1',str(image)],check=True)
    records.append({'label':label,'image':str(image),**row})
manifest={'scope':'ACTUAL_EPISODE_VIDEO_FRAME_EXTRACTION','source_video':str(video),
          'image_or_model_geometry_edited':False,'selected_actual_frames':records,
          'visibility_accepted':False,'actual_image_review_pending':True,
          'note':'This is an extraction record, not an automatic red-band or assembly acceptance decision.'}
(output/'actual_final_frame_selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'images':[r['image'] for r in records],'visibility_review_pending':True},indent=2))
