"""Installed Float3 contract and guard-domain probes; no scene or physics."""
import ast
import copy
import ctypes
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
KIT=Path('/home/noob/WorkPlace/isaacsim/.conda-env/lib/python3.12/site-packages/isaacsim/kit')
ctypes.CDLL(str(KIT/'libcarb.so'),mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0,str(KIT/'kernel/py'))
import carb

tree=ast.parse((HERE/'review.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('original','values','streamed')]
ns={'np':np,'math':math,'itertools':itertools}
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(HERE/'review.py'),'exec'),ns)

def answer(method,headers):
    try:return {'accepted':method(headers)}
    except Exception as error:return {'exception':type(error).__name__}

point={'position_m':(1.,2.,3.),'normal':(0.,0.,1.),'impulse_n_s':(0.,0.,.1),'separation_m':-1e-6}
cases={
    'valid_two_points':[copy.deepcopy(point),copy.deepcopy(point)],
    'empty_report':[],
    'ragged_second_position_len2':[copy.deepcopy(point),{**point,'position_m':(1.,2.)}],
    'uniform_position_len2':[{**point,'position_m':(1.,2.)}],
    'numeric_string_component':[{**point,'position_m':('1.0',2.,3.)}],
    'none_component':[{**point,'position_m':(None,2.,3.)}],
    'nan_then_ragged':[dict(point,position_m=(float('nan'),2.,3.)),dict(point,position_m=(1.,2.))],
}
out={}
for name,contacts in cases.items():
    headers=[{'contacts':contacts}] if contacts else []
    out[name]={label:answer(ns[label],headers) for label in ('original','streamed')}
assert out['ragged_second_position_len2']['original']=={'exception':'ValueError'}
assert out['ragged_second_position_len2']['streamed']=={'accepted':True}

native=[]
for value in (0.,-0.,float('nan'),float('inf'),-float('inf'),3e38):
    v=carb.Float3([value,1.,2.]);owned=tuple(v[:3])
    assert len(v)==3 and len(owned)==3 and all(type(x) is float for x in owned)
    headers=[{'contacts':[dict(point,position_m=owned)]}]
    old=answer(ns['original'],headers);new=answer(ns['streamed'],headers)
    assert old==new
    v[0]=5.
    assert repr(owned[0])==repr(value) or math.isclose(owned[0],value,rel_tol=1e-6)
    native.append({'requested_first_component':repr(value),'native_components':[repr(x) for x in owned],
                   'length':len(v),'slice_length':len(owned),'element_types':[type(x).__name__ for x in owned],
                   'both_guards':old,'copied_tuple_retained_after_source_mutation':True})

length_inputs=[]
for seq in ([1.,2.],[1.,2.,3.,4.]):
    try:
        v=carb.Float3(seq)
        result={'constructed_length':len(v),'slice_length':len(v[:3])}
        assert len(v)==len(v[:3])==3
    except Exception as error:result={'exception':type(error).__name__}
    length_inputs.append({'supplied_length':len(seq),**result})

result={'scope':'PURE_GUARD_AND_INSTALLED_CARB_FLOAT3_CONTRACT_NO_SIMULATION',
    'native_float3_cases':native,'native_wrong_length_construction':length_inputs,
    'arbitrary_python_container_counterexamples':out,
    'native_contact_data_contract_source':'omni.physx110.1.13 _physx.pyi:442-496',
    'float3_storage_contract_source':str(KIT/'dev/include/carb/Types.h')+':369',
    'simulation_app_created':False,'physics_executed':False,'raw_run_files_read':False,
    'production_modified':False,'prepared_files_modified':False,
    'interpretation':'Guard equivalence holds for ten built-in-float scalars from the installed Native ContactData/unchanged decoder. It is not universal malformed-container equivalence.'}
with Path(__file__).with_suffix('.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
