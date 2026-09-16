"""Small postrun benchmark of unchanged contact output with path-role reuse.

No SimulationApp, native step, production edit, decimation, or field reduction.
Only six indexed blocks (about30MB compressed) are read from sealed full14.
"""
import copy
import gc
import inspect
import json
import math
from pathlib import Path
import statistics
import textwrap
import time

import msgpack
import carts_v2.evaluate_run as evaluation
from trace_metadata import read_truth_sample, without_cyclic_gc

ROOT = Path(__file__).resolve().parents[2]
import argparse, ast
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
RUN = args.run.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
STEPS = [0, 35000, 182263, 228525, 259955, 279681]
baseline_path = ROOT / 'reproducibility/assembly_20260916/execution_sources/src/kcg_connector/isaac/carts_v2/evaluate_run.py'
baseline_text = baseline_path.read_text()
cls = next(n for n in ast.parse(baseline_text).body if isinstance(n, ast.ClassDef) and n.name == 'TruthAuditRecorder')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_contact_counts')
original_source = textwrap.dedent('\n'.join(baseline_text.splitlines()[fn.lineno-1:fn.end_lineno]))
namespace = dict(evaluation.__dict__)
exec(compile(original_source, str(baseline_path), 'exec'), namespace)
original = namespace['_contact_counts']
candidate = evaluation.TruthAuditRecorder._contact_counts
candidate_source = textwrap.dedent(inspect.getsource(candidate))


def packed(value):
    return msgpack.packb(value, use_bin_type=True)


def recorder(roots, sensor_count, native=True):
    r = evaluation.TruthAuditRecorder.__new__(evaluation.TruthAuditRecorder)
    r.contact_audit_mode = 'native-report' if native else 'full'
    r.roots = dict(roots)
    r.tensor_contact_sensor_paths = tuple(str(i) for i in range(sensor_count))
    return r


def prepare(r, raw, *, events=None, batches=None):
    polled = [{**h, 'paths':tuple(h['paths'])} for h in raw['poll_headers']]
    r._physics_step_reports = [polled] if batches is None else batches
    r._event_headers = ([(tuple(h['paths']),h['records']) for h in raw['event_headers']]
                        if events is None else events)


def call(fn, r, raw):
    prepare(r, raw)
    return fn(r)


def finite_and_branch_checks(roots):
    p = {'position_m':[.1,.2,.3], 'normal':[0.,0.,1.],
         'impulse_n_s':[0.,0.,.2], 'separation_m':-.00001}
    root = roots['object']; robot = roots['robot']; table = roots['table']; fixture = roots['fixture']
    paths = [(robot+'/handbase_link/f1Link3',root,robot+'/finger',root+'/shape'),
             (robot+'/handbase_link/f2Link2',table,robot+'/finger2',table+'/shape'),
             (robot+'/palm',root,robot+'/palm/shape',root+'/shape'),
             (robot+'/arm',fixture,robot+'/arm/shape',fixture+'/shape'),
             (robot+'/arm','/Other',robot+'/arm/shape','/Other/shape'),
             (root,table,root+'/shape',table+'/shape')]
    raw = {'poll_headers':[{'paths':list(paths[i]),'records':1,'contact_data_offset':i,'contacts':[copy.deepcopy(p)]} for i in range(len(paths))],
           'event_headers':[{'paths':list(v),'records':1} for v in paths]}
    # Preserve zero-impulse points, repeated headers and event/poll disagreements.
    raw['poll_headers'][0]['contacts'][0]['impulse_n_s'] = [0.,0.,0.]
    raw['event_headers'][1]['records'] = 3
    raw['event_headers'].append({'paths':list(paths[2]),'records':0})
    a, b = recorder(roots,19), recorder(roots,19)
    assert packed(call(original,a,raw)) == packed(call(candidate,b,raw))
    a.roots['object'] = '/ChangedObject'; b.roots['object'] = '/ChangedObject'
    assert packed(call(original,a,raw)) == packed(call(candidate,b,raw))
    assert len(b._contact_role_cache[1]) <= len(raw['poll_headers'])+len(raw['event_headers'])
    a, b = recorder(roots,19), recorder(roots,19)
    empty = {'poll_headers':[], 'event_headers':[]}
    assert packed(call(original,a,empty)) == packed(call(candidate,b,empty))
    errors = []
    for field, size in [('position_m',3),('normal',3),('impulse_n_s',3),('separation_m',1)]:
        for index in range(size):
            for bad in (float('nan'),float('inf'),-float('inf')):
                altered = copy.deepcopy(raw)
                if size==1: altered['poll_headers'][0]['contacts'][0][field] = bad
                else: altered['poll_headers'][0]['contacts'][0][field][index] = bad
                messages=[]
                for fn,r in ((original,a),(candidate,b)):
                    try: call(fn,r,altered)
                    except RuntimeError as e: messages.append(str(e))
                assert messages == ['native contact report is not finite']*2
                errors.append((field,index,str(bad)))
    # Both versions still reject missing callback evidence before producing a row.
    for fn,r in ((original,a),(candidate,b)):
        prepare(r,empty,batches=[])
        try: fn(r)
        except RuntimeError as e:
            assert str(e)=='native physics-step callback did not provide a contact report'
        else: raise AssertionError('Missing callback did not stop')
    # The untouched full-channel branch must preserve tensor/friction outputs.
    a,b=recorder(roots,19,False),recorder(roots,19,False)
    tensor=[{'sensor_index':0,'paths':[root,table],'records':1,'contacts':[{'normal_impulse_n_s':.3}]}]
    friction=[{'sensor_index':0,'filter_index':1,'paths':[root,table],'records':1,'contacts':[{'force_n':[.1,.2,.3]}]}]
    for r in (a,b):
        r._tensor_contact_rows=lambda:tensor
        r._tensor_friction_rows=lambda:friction
    assert packed(call(original,a,raw)) == packed(call(candidate,b,raw))
    return {'finite_field_failure_cases':len(errors),'all_failure_messages_unchanged':True,
            'empty_callback_data_accepted_equally':True,'missing_callback_still_stops':True,
            'changed_roots_invalidate_cache':True,'event_poll_disagreement_and_duplicates_preserved':True,
            'full_tensor_and_friction_branch_synthetic_equivalence':True}


def main():
    metadata=json.loads((RUN/'trace_metadata.json').read_text())
    roots=metadata['audit_roots']
    archive_index=json.loads((RUN/'truth_samples.msgpack.gz.index.json').read_text())
    blocks=[archive_index['blocks'][s//archive_index['block_size']] for s in STEPS]
    result={'scope':'OFFLINE_CONTACT_COUNTS_PATH_ROLE_CACHE_ONLY','source_run':str(RUN),
            'simulation_app_created':False,'physics_executed':False,'production_source_changed':False,
            'raw_fields_sampling_and_stop_conditions_changed':False,
            'indexed_compressed_bytes_read':sum(b['end']-b['offset'] for b in blocks),
            'full_simulation_speedup_claimed':False,'gc_enabled_during_timing':gc.isenabled(),
            'equivalence_checks':finite_and_branch_checks(roots),'cases':[]}
    # Keep the candidate alive across changing frames: only previous-frame keys survive.
    a=b=None
    for step in STEPS:
        sample=without_cyclic_gc(read_truth_sample,RUN,step);raw=sample['contacts']
        if a is None:
            a,b=recorder(roots,raw['tensor_contact_sensor_count']),recorder(roots,raw['tensor_contact_sensor_count'])
        t=time.perf_counter();old=call(original,a,raw);old_cold=time.perf_counter()-t
        t=time.perf_counter();new=call(candidate,b,raw);new_cold=time.perf_counter()-t
        expected=packed(raw)
        assert packed(old)==packed(new)==expected, f'Complete archived contacts differ at{step}'
        timings={'original':[],'candidate':[]}
        for repetition in range(10):
            order=[('original',original,a),('candidate',candidate,b)]
            if repetition%2:order.reverse()
            for name,fn,r in order:
                gc.collect();prepare(r,raw)
                start=time.perf_counter();output=fn(r);elapsed=time.perf_counter()-start
                timings[name].append(elapsed)
                assert packed(output)==expected
                del output
        medians={name:statistics.median(values) for name,values in timings.items()}
        points=[p for h in raw['poll_headers'] for p in h['contacts']]
        record={'step':step,'phase':sample['phase'],'headers':len(raw['poll_headers']),
            'points':len(points),'zero_impulse_points':sum(all(v==0 for v in p['impulse_n_s']) for p in points),
            'negative_separation_points':sum(p['separation_m']<0 for p in points),
            'complete_contacts_msgpack_bytes_identical_to_archive':True,
            'first_call_s':{'original':old_cold,'candidate':new_cold},
            'median_s':medians,'timings_s':timings,
            'candidate_to_original_ratio':medians['candidate']/medians['original'],
            'retained_role_cache_entries':len(b._contact_role_cache[1])}
        result['cases'].append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='timings_s'}),flush=True)
    timing=json.loads((RUN/'run_timing.json').read_text())
    total=timing['total_before_shutdown_s'];counts=timing['truth_capture']['contact_counts_s']
    result['scope_of_possible_benefit']={
        'full14_contact_counts_s':counts,'full14_total_before_shutdown_s':total,
        'maximum_total_fraction_even_if_all_contact_counts_cost_vanished':counts/total,
        'contact_callback_s_not_optimized':timing['truth_capture']['contact_callback_s'],
        'archive_append_s_not_optimized':timing['truth_capture']['archive_append_s'],
        'timing_excludes_native_report_acquisition_physics_decoding_archive_and_control':True,
        'sample_specific_ratios_are_not_whole_run_weighted_speedup':True}
    (OUT/'contact_role_cache_equivalence_and_timing.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
