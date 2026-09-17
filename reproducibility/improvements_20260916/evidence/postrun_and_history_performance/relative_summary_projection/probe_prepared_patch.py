"""Check the prepared reader branch and old callers without starting Isaac."""
import ast,hashlib,json,math,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from carts_v2.sample_store import GzipSampleStore
OUT=Path(__file__).resolve().parent
patch=json.loads((OUT/'prepared_reader_change.json').read_text())
path=ROOT/patch['file'];source=path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest()==patch['prepared_against_sha256']
assert source.count(patch['before'])==1
candidate=source.replace(patch['before'],patch['after'])
names={'_quaternion_wxyz_rotation','_split_plug_relative_motion_summary'}
def load(text):
 body=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(body)==2
 ns={'np':np,'math':math,'Path':Path};exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
 return ns['_split_plug_relative_motion_summary']
original=load(source);prepared=load(candidate)
rows=[{'step':i,'phase':'synthetic_reader_fixture','object_part_positions_m':[[.1,.2,.3],[.1+i*.0001,.2,.3+i*.0002]],'object_part_orientations_wxyz':[[1.,0.,0.,0.],[math.cos(i*.03),0.,0.,math.sin(i*.03)]],'contacts':{'unused_numbers':[1.,2.,3.]}} for i in range(4)]
scene={'relative_motion_model':'BODY_PLUS_UNLIMITED_COAXIAL_COUPLING_NUT_REVOLUTE','part_prim_paths':['fixture_body','fixture_nut'],'joint_rotational_resistance':{}}
def runtime(samples,archive=None):return {'scene':scene,'auditor':SimpleNamespace(samples=samples),'truth_stream':archive}
expected=original(runtime(rows));cases=[]
assert prepared(runtime(rows))==expected;cases.append('ordinary_list_caller_without_output_directory')
with tempfile.TemporaryDirectory(dir=OUT,prefix='reader_guard_') as tmp:
 folder=Path(tmp);store=GzipSampleStore(folder/'truth_samples.msgpack.gz',block_size=2,cache_blocks=1,codec='msgpack')
 for row in rows:store.append(row)
 try:
  assert prepared(runtime(store,store))==expected;cases.append('unclosed_fixture_retains_original_reader')
  store.close()
  assert prepared(runtime(store,store))==expected;cases.append('closed_fixture_without_motion_marker_retains_original_reader')
  (folder/'motion_timing.json').write_text('{}\n')
  assert prepared(runtime(store,store))==expected;cases.append('closed_fixture_with_marker_uses_sealed_pose_projection')
 finally:store.close()
report={'scope':'PREPARED_POSTRUN_PATCH_AST_AND_EXISTING_READER_WITH_SYNTHETIC_ARCHIVE','all_case_summaries_exactly_equal':True,'cases':cases,'production_modified':False,'physics_executed':False,'existing_episode_files_modified':False}
(OUT/'prepared_patch_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
