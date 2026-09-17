"""Check a streamed finiteness guard through the original contact-count method."""
import ast,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import msgpack,numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import read_truth_sample,without_cyclic_gc
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat01_restart01/run'
metadata=json.loads((RUN/'trace_metadata.json').read_text())
source_path=ROOT/'src/kcg_connector/isaac/carts_v2/evaluate_run.py';source=source_path.read_text()
before="""            values=np.asarray([[*c['position_m'],*c['normal'],*c['impulse_n_s'],c['separation_m']]
                for row in report_rows for c in row['contacts']],dtype=float)
            if not np.isfinite(values).all():raise RuntimeError('native contact report is not finite')"""
after="""            from itertools import chain
            values = chain.from_iterable(
                vector for row in report_rows for c in row['contacts']
                for vector in (c['position_m'], c['normal'], c['impulse_n_s'],
                               (c['separation_m'],)))
            if not all(map(math.isfinite, values)):
                raise RuntimeError('native contact report is not finite')"""
assert source.count(before)==1
candidate=source.replace(before,after)
def load(text):
 node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='_contact_counts')
 ns={'np':np,'math':math,'TERMINAL_LINK_NAMES':('f1Link3','f2Link2','f3Link3'),'_below':lambda path,root:path==root or path.startswith(root+'/')}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source_path),'exec'),ns)
 return ns['_contact_counts']
old,new=load(source),load(candidate)
def recorder(raw):
 contacts=raw['contacts'];headers=[]
 for row in contacts['poll_headers']:
  points=[{**c,**{k:tuple(c[k]) for k in ('position_m','normal','impulse_n_s')}} for c in row['contacts']]
  headers.append({**row,'paths':tuple(row['paths']),'contacts':points})
 events=[(tuple(r['paths']),r['records']) for r in contacts['event_headers']]
 assert contacts['physics_step_callback_count']==1
 return SimpleNamespace(roots=metadata['audit_roots'],contact_audit_mode='native-report',tensor_contact_sensor_paths=[None]*contacts['tensor_contact_sensor_count'],_event_headers=events,_physics_step_reports=[headers])
rows=[]
for step in (10000,35000,230008,294559):
 raw=without_cyclic_gc(read_truth_sample,RUN,step)
 original=without_cyclic_gc(old,recorder(raw));prepared=without_cyclic_gc(new,recorder(raw))
 a=msgpack.packb(original,use_bin_type=True);b=msgpack.packb(prepared,use_bin_type=True);saved=msgpack.packb(raw['contacts'],use_bin_type=True)
 assert a==b and a==saved,step
 rows.append({'step':step,'phase':raw['phase'],'all_contact_fields_and_encoded_bytes_equal_to_original_saved_record':True,'contact_record_bytes':len(a),'sha256':hashlib.sha256(a).hexdigest()})
patch={'file':str(source_path.relative_to(ROOT)),'before':before,'after':after,'prepared_against_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'applied':False,'all_native_points_and_existing_invalid_value_rejection_retained':True}
(OUT/'prepared_streaming_guard_change.json').write_text(json.dumps(patch,indent=2)+'\n')
report={'scope':'PREPARED_GUARD_FULL_CONTACT_METHOD_REPLAY_OF_FOUR_SEALED_REAL_FRAMES','rows':rows,'production_modified':False,'physics_executed':False,'raw_archive_or_reports_modified':False}
(OUT/'prepared_patch_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
