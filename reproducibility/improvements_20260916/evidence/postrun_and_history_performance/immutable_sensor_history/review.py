"""Offline lifetime cost of saved sensor arrays; never used by the live controller."""
import gc,gzip,hashlib,itertools,json,resource,statistics,subprocess,sys,time
from pathlib import Path
import ijson,msgpack
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat01_restart01/run'
assert (RUN/'motion_timing.json').is_file()
SOURCE=RUN/'wrist_ft_samples.json.gz'
COUNT=32768

def own_immutable_arrays(value):
 if isinstance(value,list):return tuple(own_immutable_arrays(v) for v in value)
 if isinstance(value,dict):return {k:own_immutable_arrays(v) for k,v in value.items()}
 return value

def child(mode):
 rows=[];expected=hashlib.sha256();actual=hashlib.sha256();conversion=0.;started=time.perf_counter();before=gc.isenabled();gc.disable()
 try:
  with gzip.open(SOURCE,'rb') as stream:
   for source_row in itertools.islice(ijson.items(stream,'item',use_float=True),COUNT):
    original_bytes=msgpack.packb(source_row,use_bin_type=True);expected.update(original_bytes)
    t=time.perf_counter();row=source_row if mode=='lists' else own_immutable_arrays(source_row);conversion+=time.perf_counter()-t
    encoded=msgpack.packb(row,use_bin_type=True);assert encoded==original_bytes;actual.update(encoded);rows.append(row)
  del row,source_row,encoded,original_bytes
  assert len(rows)==COUNT
  ingestion=time.perf_counter()-started
  collection=[]
  for _ in range(6):
   t=time.perf_counter();collected=gc.collect(2);elapsed=time.perf_counter()-t
   collection.append({'wall_s':elapsed,'collected':collected,'tracked_sensor_row_dicts':sum(gc.is_tracked(r) for r in rows)})
  result={'mode':mode,'sensor_rows':COUNT,'scope':'OFFLINE_SAVED_ROBOT_SIGNALS_ONLY','input':str(SOURCE),'ingestion_parse_encode_and_check_wall_s':ingestion,'array_conversion_s':conversion,'gen2_collections':collection,'steady_collection_median_s':statistics.median(x['wall_s'] for x in collection[2:]),'all_row_encoded_values_exactly_equal':expected.hexdigest()==actual.hexdigest(),'logical_rows_sha256':actual.hexdigest(),'maximum_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'production_modified':False,'physics_executed':False,'controller_mutation_compatibility_not_yet_verified':True}
  (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:
  if before:gc.enable()

if __name__=='__main__':
 if len(sys.argv)>1:child(sys.argv[1])
 else:
  for mode in ('lists','tuples'):
   with (OUT/(mode+'.log')).open('w') as log:subprocess.run([sys.executable,str(Path(__file__).resolve()),mode],stdout=log,stderr=subprocess.STDOUT,check=True)
  a=json.loads((OUT/'lists.json').read_text());b=json.loads((OUT/'tuples.json').read_text());assert a['logical_rows_sha256']==b['logical_rows_sha256']
  result={'scope':'BOUNDED_OFFLINE_32768_REAL_SENSOR_RECORDS','original':a,'candidate':b,'collection_cost_reduction_fraction':1-b['steady_collection_median_s']/a['steady_collection_median_s'],'not_a_whole_episode_speed_estimate':True,'not_adopted_into_live_physics':True}
  (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
