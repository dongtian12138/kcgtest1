"""Evidence remains numeric and complete; runtime deadlines are real time."""
from pathlib import Path
import json,math,os,subprocess,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from carts_v2.fast_json import dumps,loads,dump_array


def test_fast_codec_preserves_floats_and_failed_nonfinite_observations(tmp_path):
    values=[-0.,1e-16,1.2345678901234567,float('nan'),float('inf'),-float('inf')]
    original={'p':np.asarray(values),'path':Path('/tmp/原始数据'),'contact':[{'impulse':values[:3]}]}
    encoded=dumps(original);decoded=loads(encoded)
    assert decoded['path']=='/tmp/原始数据'
    assert [float(x).hex() for x in decoded['p'][:3]]==[x.hex() for x in values[:3]]
    assert math.isnan(decoded['p'][3]) and decoded['p'][4]==float('inf')
    assert 'NaN' in encoded and 'Infinity' in encoded
    import io
    f=io.StringIO();dump_array(f,[{'sample':i,'x':np.float64(i/3)} for i in range(50)])
    assert json.loads(f.getvalue())==[{'sample':i,'x':i/3} for i in range(50)]


def test_contact_path_cache_does_not_change_raw_contact_values():
    from types import SimpleNamespace as N
    from carts_v2.evaluate_run import TruthAuditRecorder
    r=TruthAuditRecorder.__new__(TruthAuditRecorder);seen=[]
    r.path_decoder=lambda x:seen.append(x) or '/body/'+str(x)
    h=N(actor0=1,actor1=2,collider0=3,collider1=4,contact_data_offset=0,num_contact_data=1)
    p=[1.,2.,3.];c=N(position=p,normal=[0.,0.,1.],impulse=[0.,0.,.003],separation=-1e-6)
    first=r._decode_full_report([h],[c]);second=r._decode_full_report([h],[c])
    assert first==second and seen==[1,2,3,4]
    p[0]=9.
    assert first[0]['contacts'][0]['position_m']==[1.,2.,3.]


def test_real_time_guard_terminates_a_stuck_child_before_long_experiment(tmp_path):
    guard=Path(__file__).resolve().parents[1]/'isaac/bounded_experiment.py'
    env=os.environ.copy();env.update(KCG_EXPERIMENT_WALL_LIMIT_S='.3',KCG_EXPERIMENT_CLOSEOUT_RESERVE_S='.1')
    started=time.monotonic()
    result=subprocess.run([sys.executable,str(guard),sys.executable,'-c','import time; time.sleep(10)'],env=env,capture_output=True,timeout=3)
    assert result.returncode==124
    assert time.monotonic()-started<2.
    assert b'EXPERIMENT_WALL_BUDGET_REACHED' in result.stdout


def test_source_snapshot_uses_the_sealed_block_index(tmp_path):
    from carts_v2.sample_store import GzipSampleStore
    from trace_metadata import read_truth_sample
    store=GzipSampleStore(tmp_path/'truth_samples.jsonl.gz',block_size=4)
    for i in range(11):store.append({'step':i,'value':i/3,'contact':[i,i+1]})
    store.close()
    assert read_truth_sample(tmp_path,9)=={'step':9,'value':3.,'contact':[9,10]}
