#!/usr/bin/env python3
"""Review a new ended episode; saved baseline PASS files are never copied."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CHECKS=[
 ('body','src/kcg_connector/isaac/evaluate_visual_body_grasp.py',[],'source_nail_body_review.json','accepted'),
 ('high_global1','reproducibility/two_key_20260919/review_high_global1.py',[],'high_global1_initial_review.json','accepted'),
 ('nut','src/kcg_connector/isaac/evaluate_source_nut_pad.py',['--geometry-plan',str(ROOT/'artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json')],'source_nut_pad_review.json','accepted'),
 ('two_key','reproducibility/two_key_20260919/review_two_key.py',[],'two_key_alignment_review.json','passed'),
 ('cameras','reproducibility/four_camera_20260918/audit_camera_records.py',[],'four_camera_contract_review.json','passed'),
 ('transport','reproducibility/four_camera_20260918/review_transport.py',[],'four_camera_transport_posthoc.json',None),
 ('support','reproducibility/four_camera_20260918/review_native_body_support.py',[],'native_body_support_review.json','accepted'),
 ('keys','src/kcg_connector/isaac/evaluate_source_key_containment.py',[],'source_key_containment_review.json','accepted'),
 ('release','reproducibility/four_camera_20260918/review_terminal_release.py',[],'three_second_release_review.json','accepted'),
 ('band','reproducibility/improvements_20260916/postreview/review_source_band_occlusion.py',[],'source_band_radial_occlusion_review.json','all_sampled_radial_views_blocked_by_nut')]

def read(path):return json.loads(path.read_text())

def identity(run):
 p=run/'truth_samples.msgpack.gz';s=p.stat()
 return {'run':str(run),'process_sha256':hashlib.sha256((run.parent/'process.json').read_bytes()).hexdigest(),
         'archive_size':s.st_size,'archive_mtime_ns':s.st_mtime_ns}

def execute(run, name, script, extra=()):
 env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ['src/kcg_connector','src/kcg_connector/isaac','src/kcg_connector/isaac/carts_v2'])
 cmd=[str(ROOT/'src/kcg_connector/isaac/run_isaac_python.sh'),'-c',
      'import gc,runpy,sys;gc.disable();sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")',
      str(ROOT/script),str(run),*extra]
 with (run/'postrun_evidence'/(name+'.log')).open('w') as f:
  subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' finished',flush=True)

def frames(run, terminal):
 out=run/'postrun_evidence';data=[json.loads(x) for x in (run/'video/assembly_five_view_frames.jsonl').read_text().splitlines()]
 turns=[x for x in data if x['phase']=='key_probe_nut_rotation_turn']
 held=[x for x in data if terminal['opening_last_step']<=x['step']<terminal['support_hold_last_step'] and x['phase']=='nut_index_free_open_hold']
 if not turns or not held:raise ValueError('Missing original final-rotation or released video frames')
 selected=[]
 for label,row in [('rotation_end_before_release',turns[-1]),('released_hold_midpoint',held[len(held)//2]),('final',held[-1])]:
  image=out/(label+'.png')
  if not image.exists():
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(run/'video/assembly_five_view.mp4'),
       '-vf',f"select=eq(n\\,{row['frame']})",'-frames:v','1',str(image)],check=True)
  selected.append({**row,'label':label,'image':str(image)})
 return selected

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
 p.add_argument('--confirm-band-hidden',action='store_true',help='已亲自查看提取的三帧，并确认主视角/全局2未露出红色指示带；仅在完成数值审查后使用')
 a=p.parse_args();run=a.run.resolve();process=read(run.parent/'process.json')
 if 'exit_code' not in process:raise ValueError('Only an ended episode may be reviewed')
 terminal=read(run/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json')
 if terminal.get('completed') is not True:raise ValueError('Terminal release did not complete; diagnose the actual failure first')
 out=run/'postrun_evidence';out.mkdir(exist_ok=True);binding=identity(run)
 if not a.confirm_band_hidden:
  def chain(job):
   for name,script,extra,_,_ in job:execute(run,name,script,extra)
  jobs=[[CHECKS[0],CHECKS[1]]]+[[x] for x in CHECKS[2:]]
  with ThreadPoolExecutor(max_workers=3) as pool:
   for future in as_completed([pool.submit(chain,j) for j in jobs]):future.result()
  results={name:(read(run/result).get(key) is True if key else 'DIAGNOSTIC_ONLY') for name,_,_,result,key in CHECKS}
  (out/'numerical_review_binding.json').write_text(json.dumps({'binding':binding,'results':results},indent=2)+'\n')
  print(json.dumps(results,indent=2))
  if any(v is False for v in results.values()):raise ValueError('At least one numerical condition failed; no success is claimed')
  selected=frames(run,terminal);(out/'visibility_review_inputs.json').write_text(json.dumps(selected,indent=2)+'\n')
  print('数值分项完成；请查看 postrun_evidence/ 中三张实际帧。确认红带未露出后，再以 --confirm-band-hidden 完成影像记录和全程审核。')
  return
 prior=read(out/'numerical_review_binding.json')
 if prior['binding']!=binding or any(v is False for v in prior['results'].values()):raise ValueError('Numerical review is missing, failed, or belongs to changed data')
 selected=read(out/'visibility_review_inputs.json');band=read(run/'source_band_radial_occlusion_review.json')
 if not band['all_sampled_radial_views_blocked_by_nut']:raise ValueError('Source indicator band is not covered')
 visual={'scope':'USER_INSPECTED_ORIGINAL_SAME_EPISODE_FRAMES','accepted':True,'online_control_used':False,
         'reviewed_actual_frames':selected,'actual_image_observation':'User explicitly confirmed no exposed red indicator band in main/Global2 views of all three extracted original frames.',
         'geometric_cross_check':{'path':str(run/'source_band_radial_occlusion_review.json'),'sample_count':band['sample_count'],'covered_by_nut_sample_count':band['covered_by_nut_sample_count']}}
 (run/'final_mating_visibility_review.json').write_text(json.dumps(visual,indent=2)+'\n')
 execute(run,'whole','src/kcg_connector/isaac/evaluate_visual_assembly_v1.py')
 whole=read(run/'whole_assembly_review.json');passed=whole.get('complete_visual_assembly_verified') is True and read(run/'three_second_release_review.json')['accepted'] is True
 print('整体验收：'+('VERIFIED' if passed else 'REVIEW_REQUIRED'))
 if not passed:raise SystemExit(2)

if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
  print('审查未完成：'+str(error),file=sys.stderr);raise SystemExit(2)
