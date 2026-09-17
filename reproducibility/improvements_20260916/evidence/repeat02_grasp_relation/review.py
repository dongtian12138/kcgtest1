"""Use the established pose decomposition on three ended repeat02 strokes."""
import ast,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import iter_truth_fields
source=(ROOT/'artifacts/control_review/grasp_relation_decomposition/review.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='review')
function=ast.unparse(node).replace('def review(run):','def review(run, stage):')
function=function.replace("run / 'socket_transport/nut_rotation_continued_02'","run / 'socket_transport' / stage")
function=function.replace('row = without_cyclic_gc(read_truth_sample, run, step)',"row = next(iter_truth_fields(run, ('object_part_positions_m', 'native_robot_link_pose_audit', 'arm_control'), step, step))")
function=function.replace("next((r for r in controls if abs(r['commanded_rotation_deg']) >= 20))","min(controls, key=lambda r: abs(abs(r['commanded_rotation_deg']) - min(20., abs(controls[-1]['commanded_rotation_deg'])/2)))")
assert 'without_cyclic_gc' not in function and 'socket_transport/nut_rotation_continued_02' not in function
namespace=globals().copy();exec(compile(function,str(Path(__file__).resolve()),'exec'),namespace)
run=ROOT/'artifacts/full_validation/contact_last_gc128_repeat02/run';results=[]
for stage in ['nut_rotation_continued_01','nut_rotation_continued_02','nut_rotation_continued_04']:
 result=namespace['review'](run,stage);result['stage']=stage;result['scope']='ENDED_REPEAT02_STAGE_POSE_DECOMPOSITION';results.append(result)
(Path(__file__).parent/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
for result in results:print(result['stage'],json.dumps(max(result['rows'],key=lambda r:r['reported_xy_error_mm']),indent=2))
