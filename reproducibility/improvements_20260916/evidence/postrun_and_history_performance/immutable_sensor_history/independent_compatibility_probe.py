"""Small read-only type/serialization checks on sealed robot records; no physics."""
import ast
from collections import Counter
import copy
import gzip
import io
import itertools
import json
from pathlib import Path
import sys
import ijson
import msgpack
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat01_restart01/run'
assert (RUN/'motion_timing.json').is_file()
sys.path.insert(0,str(ROOT/'src/kcg_connector/isaac'))
from carts_v2.fast_json import dumps, dump_array

def function(path,name,ns):
    tree=ast.parse(path.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    return ns[name]

freeze=function(Path(__file__).with_name('review.py'),'own_immutable_arrays',{})
ready=function(ROOT/'src/kcg_connector/isaac/te_foundationpose_handoff_runtime.py','_json_ready',{'np':np,'Path':Path})
phases=Counter();samples=[];first_step=last_step=None
with gzip.open(RUN/'wrist_ft_samples.json.gz','rb') as stream:
    for i,row in enumerate(itertools.islice(ijson.items(stream,'item',use_float=True),32768)):
        phases[row['phase']]+=1
        if first_step is None:first_step=row['step']
        last_step=row['step']
        if i in (0,32767):samples.append(copy.deepcopy(row))
for relative in ('socket_transport/nut_rotation_continued_04/joint_ft_samples.json.gz',
                 'socket_transport/nut_terminal_release/joint_ft_samples.json.gz'):
    with gzip.open(RUN/relative,'rb') as stream:
        samples.append(next(ijson.items(stream,'item',use_float=True)))

checks=[]
for original in samples:
    candidate=freeze(original)
    before=msgpack.packb(candidate,use_bin_type=True)
    assert before==msgpack.packb(original,use_bin_type=True)
    assert json.dumps(ready(candidate),separators=(',',':'))==json.dumps(ready(original),separators=(',',':'))
    assert dumps(candidate)==dumps(original)
    a,b=io.StringIO(),io.StringIO()
    dump_array(a,[original]);dump_array(b,[candidate]);assert a.getvalue()==b.getvalue()
    for key in ('active_positions_rad','active_targets_rad','active_efforts_nm',
                'handbase_rotation_world_row_major','handbase_position_world_m','hand2arm_raw_wrench'):
        old=np.asarray(original[key]);new=np.asarray(candidate[key])
        assert old.dtype==new.dtype and np.array_equal(old,new)
        assert isinstance(candidate[key],tuple)
    hand=np.eye(4);hand[:3,:3]=candidate['handbase_rotation_world_row_major'];hand[:3,3]=candidate['handbase_position_world_m']
    raw=candidate['hand2arm_raw_wrench']
    assert np.array_equal(hand[:3,:3]@raw[:3],np.asarray(original['handbase_rotation_world_row_major'])@original['hand2arm_raw_wrench'][:3])
    mutable=np.asarray(candidate['active_targets_rad'],dtype=float)
    mutable[0]+=1.
    original['active_positions_rad'][0]+=1.
    assert msgpack.packb(candidate,use_bin_type=True)==before
    checks.append({'step':candidate['step'],'phase':candidate['phase'],
        'msgpack_json_ready_fast_json_and_dump_array_equal':True,
        'numpy_shapes_dtypes_values_and_raw_tuple_matmul_equal':True,
        'source_list_mutation_and_mutable_numpy_work_array_do_not_change_history':True})

special={'array':[None,True,-0.0,float('inf'),float('-inf'),float('nan')]}
assert dumps(special)==dumps(freeze(special))
assert msgpack.packb(special,use_bin_type=True)==msgpack.packb(freeze(special),use_bin_type=True)
result={'scope':'SEALED_ROBOT_HISTORY_CONSUMER_TYPE_CHECK_ONLY',
    'source_run':str(RUN),'benchmark_prefix':{'rows':sum(phases.values()),'first_step':first_step,'last_step':last_step,'phase_counts':dict(phases)},
    'four_representative_record_checks':checks,'synthetic_special_values_preserved':True,
    'controller_executed':False,'physics_executed':False,'production_modified':False,
    'limitations':['This exercises current numeric/serialization consumption patterns, not the full live controller or all exceptional sensor states.',
                   'The benchmark prefix is read only for phase coverage; its GC timings are not rerun here.',
                   'No repeat02 file is read.']}
with Path(__file__).with_suffix('.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
