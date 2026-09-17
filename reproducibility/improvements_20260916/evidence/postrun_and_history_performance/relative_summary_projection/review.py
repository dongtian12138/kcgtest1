"""Compare only an ended episode's existing relative-motion summary reader."""
import ast,gc,json,math,sys,time,hashlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import iter_truth_fields
from carts_v2.sample_store import GzipSampleStore
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat01_restart01/run'
assert (RUN/'motion_timing.json').is_file()
archive=RUN/'truth_samples.msgpack.gz'
index=json.loads(Path(str(archive)+'.index.json').read_text())
assert index['blocks'][-1]['end']==archive.stat().st_size
metadata=json.loads((RUN/'trace_metadata.json').read_text())
reference=metadata['split_plug_relative_motion']
source=ROOT/'src/kcg_connector/isaac/carts_v2/run_grasp_lift.py'
names={'_quaternion_wxyz_rotation','_split_plug_relative_motion_summary'}
body=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(body)==2
module=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]));ns={'np':np,'math':math};exec(compile(module,str(source),'exec'),ns)
scene={'relative_motion_model':'BODY_PLUS_UNLIMITED_COAXIAL_COUPLING_NUT_REVOLUTE','part_prim_paths':reference['part_order'],'joint_rotational_resistance':reference['rotational_resistance']}
fields=('object_part_positions_m','object_part_orientations_wxyz')
class ProjectedRows:
 def __init__(self,first,last):self.first,self.last=first,last
 def __len__(self):return self.last-self.first+1
 def __iter__(self):
  yield from iter_truth_fields(RUN,fields,self.first,self.last)

def measure(samples,label):
 enabled=gc.isenabled();gc.disable();start=time.perf_counter();cpu=time.process_time()
 try:result=ns['_split_plug_relative_motion_summary']({'scene':scene,'auditor':SimpleNamespace(samples=samples)})
 finally:
  elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu
  if enabled:gc.enable()
 print(label,elapsed,'seconds',flush=True)
 return result,{'wall_s':elapsed,'process_cpu_s':cpu_elapsed,'sample_count':len(samples)}

candidate,full_timing=measure(ProjectedRows(0,index['sample_count']-1),'full_pose_only')
(OUT/'full_candidate_summary.json').write_text(json.dumps(candidate,indent=2)+'\n')
assert candidate==reference,'Full projected summary differs from preserved real-episode result'
# Reuse the original store reader in read-only state for a bounded paired window.
store=GzipSampleStore.__new__(GzipSampleStore);store.name=str(archive);store.codec='msgpack';store.format=index['format'];store.block_size=index['block_size'];store.cache_blocks=1;store._count=index['sample_count'];store._blocks=index['blocks'];store._cache=OrderedDict();store._live=[];store.closed=True
first=index['sample_count']-1024;last=index['sample_count']-1
original,old_time=measure(store[first:last+1],'dense1024_original_store')
projected,new_time=measure(ProjectedRows(first,last),'dense1024_projected')
assert original==projected,'Matched window summaries differ'
report={'scope':'POSTRUN_READER_ONLY_ORIGINAL_SUMMARY_FORMULAS_UNMODIFIED','episode':str(RUN),'source_commit_of_episode':metadata.get('source_git_commit','25a0d6f68771080fe60a7dde948cdaa4e8d5927f'),'summary_functions_ast_sha256':hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),'full_summary_all_fields_exactly_equal':candidate==reference,'full_projected_timing':full_timing,'paired_dense_window_inclusive':[first,last],'paired_summaries_exactly_equal':original==projected,'paired_original_timing':old_time,'paired_projected_timing':new_time,'scene_inputs_only':['relative_motion_model','part_prim_paths','joint_rotational_resistance'],'measured_summary_values_used_as_inputs':False,'production_source_modified':False,'physics_executed':False,'raw_archive_or_original_reports_modified':False,'reference_full_summary_separate_timing_unavailable':True,'timing_limits':'One sequential pair during another process execution; not a claimed whole-episode acceleration.'}
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
