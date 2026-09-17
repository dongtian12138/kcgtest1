"""Bounded sealed-run key review: source binding plus only three witness frames."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]/'run'
sys.path.insert(0,str(ROOT/'src/kcg_connector/isaac'))
from trace_metadata import iter_truth_fields

def load(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()

report=load(RUN/'source_key_containment_review.json')
index=load(RUN/'truth_samples.msgpack.gz.index.json')
metadata=load(RUN/'trace_metadata.json')
entry=load(RUN/'socket_transport/key_entry/key_entry_controller_result.json')
installation=load(RUN/'frozen_model_installation.json')
plan=load(RUN.parent/'reproduction_plan.json')
assert (RUN/'motion_timing.json').is_file()
assert index['blocks'][-1]['end']==(RUN/'truth_samples.msgpack.gz').stat().st_size
assert report['first_reviewed_step']==entry['contact_first_step']==146350
assert report['final']['step']==index['sample_count']-1==294559
assert report['numerical_sidewall_review_tolerance_m']==2e-6

source=Path(report['source_model'])
source_rel=str(source.relative_to(ROOT))
source_hash=sha(source)
original=Path('/home/noob/WorkPlace/kcgtest1')/source_rel
assert source_hash==metadata['evidence_binding']['scene_evidence_sha256'][source_rel]
assert source_hash==sha(original)
assert str(source)==installation['model_path']
dimensions_path=ROOT/'artifacts/kcg_connector/key_antirotation_20260910/key_backlash_dimensions.json'
if not dimensions_path.exists():dimensions_path=ROOT/'reproducibility/assembly_20260916/key_backlash_dimensions.json'
dimensions=load(dimensions_path)
original_dimensions=Path('/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/key_antirotation_20260910/key_backlash_dimensions.json')
assert dimensions==load(original_dimensions)
manifest_path=ROOT/'reproducibility/assembly_20260916/portable_source_manifest.json'
manifest=load(manifest_path)
assert sha(manifest_path)==plan['source_manifest_sha256']
evaluator_rel='src/kcg_connector/isaac/evaluate_source_key_containment.py'
assert sha(ROOT/evaluator_rel)==manifest['sha256'][evaluator_rel]
dimensions_rel=str(dimensions_path.relative_to(ROOT))
assert sha(dimensions_path)==manifest['sha256'][dimensions_rel]
old_review=load(Path('/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/source_key_containment_review.json'))
assert old_review['numerical_sidewall_review_tolerance_m']==report['numerical_sidewall_review_tolerance_m']

stage=Usd.Stage.Open(str(source))
vertices=[]
for k in range(5):
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(f'/World/TE_J35FreeSplitPlug/Body/SourceGuideKey_{k:03d}'))
    local=np.asarray(mesh.GetPointsAttr().Get())
    transform=np.asarray(UsdGeom.Xformable(mesh).GetLocalTransformation())
    vertices.append((np.c_[local,np.ones(len(local))]@transform)[:,:3])
socket=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][
    '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']),float).T
rows=[]
fields={'step','phase','object_part_positions_m','object_part_orientations_wxyz'}
for row in iter_truth_fields(RUN,fields,first_step=230007,last_step=230009):
    rotation=socket[:3,:3].T@Rotation.from_quat(np.roll(row['object_part_orientations_wxyz'][0],-1)).as_matrix()
    position=socket[:3,:3].T@(np.asarray(row['object_part_positions_m'][0])-socket[:3,3])
    gaps=[];inside_counts=[]
    for points,key in zip(vertices,dimensions['keys']):
        world=points@rotation.T+position
        inside=world[world[:,2]<0];inside_counts.append(len(inside))
        values=np.asarray([inside*1000@np.asarray(plane['normal'])-plane['d_mm'] for plane in key['planes']])*.001
        gaps.append(float(values.min()) if len(inside) else None)
    rows.append({'step':row['step'],'phase':row['phase'],
                 'body_position_m':row['object_part_positions_m'][0],
                 'body_orientation_wxyz':row['object_part_orientations_wxyz'][0],
                 'inside_vertex_counts':inside_counts,'minimum_sidewall_gaps_m':gaps})
assert [x['step'] for x in rows]==[230007,230008,230009]
witness=report['minimum_gap_witnesses'][2]
assert witness['step']==230008 and rows[1]['phase']==witness['phase']
assert rows[1]['minimum_sidewall_gaps_m'][2]==witness['gap_m']==report['minimum_sidewall_gaps_m'][2]

summary={}
for name in ['source_nail_body_review.json','source_nut_pad_review.json','physical_key_entry_result.json',
             'final_mating_visibility_review.json','whole_assembly_review.json','evaluation.json']:
    p=RUN/name
    if not p.exists():summary[name]={'present':False};continue
    value=load(p)
    summary[name]={'present':True,**{k:value[k] for k in ('scope','status','accepted','complete_visual_assembly_verified','mechanical_conditions','visual_stage_conditions','additional_review_conditions') if k in value}}
result={
    'scope':'INDEPENDENT_BOUND_SOURCE_AND_THREE_FRAME_WORST_WITNESS_REVIEW_NOT_FULL_RESCAN',
    'run':str(RUN),'full_archive_sample_count':index['sample_count'],
    'archive_index_end_equals_file_bytes':index['blocks'][-1]['end'],
    'source_model':str(source),'source_model_sha256':source_hash,
    'source_matches_run_evidence_binding_and_original_full14_model':True,
    'dimensions_path':str(dimensions_path),'dimensions_sha256':sha(dimensions_path),
    'dimensions_equal_original_json':True,'run_manifest_and_bound_evaluator_unchanged':True,
    'first_contact_step_matches_report':entry['contact_first_step'],
    'last_report_step_matches_sealed_last_sample':report['final']['step'],
    'numerical_tolerance_m_unchanged_from_full14':report['numerical_sidewall_review_tolerance_m'],
    'reported_minimum_sidewall_gaps_um':[v*1e6 for v in report['minimum_sidewall_gaps_m']],
    'reported_evaluated_samples_by_key':report['evaluated_sample_count_by_key'],
    'worst_key_number_one_based':3,'witness_recomputed_exactly_equal':True,
    'headroom_to_original_2um_band_um':(2e-6+report['minimum_sidewall_gaps_m'][2])*1e6,
    'sampled_rows':rows,'available_summary_snapshot':summary,
    'limitations':[
        'Only230007..230009 were read here; no25GB full archive rescan or compressed stream hash.',
        'The original evaluator traverses the indexed interval146350..294559 contiguously but applies its original key_probe_/nut_index_ phase filter and only tests vertices behind the socket mouth. Counts are not294560 checks per key.',
        'Accepted means within the original2um postrun numerical band; the small negative gaps are preserved, not declared zero penetration.',
        'The final key report omits a run-path or archive-digest field; this audit binds it by source hashes, same-episode entry/final indices and exact witness recomputation.',
        'Summary fields are read from existing reports; their entire scans and actual video viewing are not repeated in this bounded audit.',
    ],
}
with Path(__file__).with_suffix('.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
