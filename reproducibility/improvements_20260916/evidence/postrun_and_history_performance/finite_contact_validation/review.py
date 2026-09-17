"""Offline equivalent finiteness checks on one sealed dense contact report."""
import gc,itertools,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import read_truth_sample,without_cyclic_gc
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat01_restart01/run'
index=json.loads((RUN/'truth_samples.msgpack.gz.index.json').read_text())
assert (RUN/'motion_timing.json').exists() and index['blocks'][-1]['end']==(RUN/'truth_samples.msgpack.gz').stat().st_size
raw=without_cyclic_gc(read_truth_sample,RUN,index['sample_count']-1)
headers=raw['contacts']['poll_headers']
for row in headers:
 for c in row['contacts']:
  for k in ('position_m','normal','impulse_n_s'):c[k]=tuple(c[k])
def original(h):
 values=np.asarray([[*c['position_m'],*c['normal'],*c['impulse_n_s'],c['separation_m']] for row in h for c in row['contacts']],dtype=float)
 return bool(np.isfinite(values).all())
def values(h):
 return itertools.chain.from_iterable(v for row in h for c in row['contacts'] for v in (c['position_m'],c['normal'],c['impulse_n_s'],(c['separation_m'],)))
def streamed(h):return all(map(math.isfinite,values(h)))
def fromiter(h):return bool(np.isfinite(np.fromiter(values(h),dtype=float)).all())
checks={'original_array':original,'streamed_isfinite':streamed,'fromiter_array':fromiter}
validation=[]
for bad in (float('nan'),float('inf'),-float('inf'),1.e308,-1.e308):
 for component in range(10):
  a=[0.]*10;a[component]=bad
  fixture=[{'contacts':[{'position_m':tuple(a[:3]),'normal':tuple(a[3:6]),'impulse_n_s':tuple(a[6:9]),'separation_m':a[9]}]}]
  answers={k:f(fixture) for k,f in checks.items()};assert len(set(answers.values()))==1
  validation.append({'value':str(bad),'component':component,'accepted':answers['original_array']})
assert all(f([]) for f in checks.values()) and all(f(headers) for f in checks.values())
times={k:[] for k in checks};gc.collect(2);before=gc.isenabled();gc.disable()
try:
 for repetition in range(30):
  order=list(checks) if repetition%2==0 else list(checks)[::-1]
  for name in order:
   t=time.perf_counter();assert checks[name](headers);times[name].append(time.perf_counter()-t)
finally:
 if before:gc.enable()
report={'scope':'OFFLINE_ONE_REAL_DENSE_REPORT_PLUS_NONFINITE_AND_EXTREME_FINITE_FIELDS','run':str(RUN),'sample_step':raw['step'],'contact_points':sum(len(r['contacts']) for r in headers),'balanced_order_repetitions':30,'median_s':{k:statistics.median(v) for k,v in times.items()},'all_tested_boolean_results_identical':True,'invalid_and_extreme_values_tested_at_all_ten_components':validation,'production_modified':False,'raw_archive_modified':False,'physics_executed':False,'not_a_whole_episode_timing':True}
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='invalid_and_extreme_values_tested_at_all_ten_components'},indent=2))
