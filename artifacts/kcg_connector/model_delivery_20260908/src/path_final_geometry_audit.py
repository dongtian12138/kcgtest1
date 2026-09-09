"""Offline final collision-region review using the actual exported candidate."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from pxr import Usd,UsdGeom
import fcl
import argparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='full_path_01')
parser.add_argument('--output-prefix',default='path_final_geometry')
parser.add_argument('--source-geometry-model',type=Path,
                    help='Evaluate unchanged original metal surfaces when the candidate partitions its collision mesh.')
args=parser.parse_args()
RUN=ROOT/'runs'/args.run
actual_stage=Usd.Stage.Open(str(RUN/'connector_candidate.usdc'))
stage=Usd.Stage.Open(str(args.source_geometry_model)) if args.source_geometry_model else actual_stage
body_path='/World/TE_J35FreeSplitPlug/Body'
nut_path='/World/TE_J35FreeSplitPlug/CouplingNut'
socket_path='/World/TEVisualHandoff/FixedReceptaclePose'
paths={'core':body_path+'/SocketRigidCoreCollision','nut':nut_path+'/SourceCadCollision',
       'socket':socket_path+'/OfficialVisual/Geometry'}
actual_socket_world=np.asarray(UsdGeom.Xformable(actual_stage.GetPrimAtPath(socket_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).T
def usdT(path):
    return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).T
def mesh_in_frame(path,frame):
    m=UsdGeom.Mesh.Get(stage,path); v=np.asarray(m.GetPointsAttr().Get(),float)
    counts=np.asarray(m.GetFaceVertexCountsAttr().Get(),int)
    assert np.all(counts==3)
    f=np.asarray(m.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
    local=np.linalg.inv(usdT(frame))@usdT(path)
    v=v@local[:3,:3].T+local[:3,3]
    return v,f
geometry={k:mesh_in_frame(path,body_path if k=='core' else nut_path if k=='nut' else socket_path) for k,path in paths.items()}
models={}
for name,(v,f) in geometry.items():
    m=fcl.BVHModel();m.beginModel(len(f),len(v));m.addSubModel(v,f);m.endModel();models[name]=m
socket_obj=fcl.CollisionObject(models['socket'])
last=None;near=None;err=float('inf')
for line in (RUN/'samples.jsonl').open():
    row=json.loads(line);last=row
    e=abs(row['body_depth_m']-.0144)
    if row['phase']=='rotation' and e<err:near=row;err=e
source_pin=json.loads((ROOT.parent/'isaac/te_full_assembly_20260905/source_socket_blind_bore_faces_v1.json').read_text())
centers=np.asarray(source_pin['source_plug_pin_centers_m'])
socket_centers=np.asarray([x['center_m'][:2] for x in source_pin['cap_faces']])
core_v,core_f=geometry['core'];ct=core_v[core_f]
vertex_pin_dist=cKDTree(centers).query(core_v[:,:2])[0]
key_slab=(core_v[:,2]>=-.007646)&(core_v[:,2]<=-.000761)
key_outer=key_slab&(np.linalg.norm(core_v[:,:2],axis=1)>.0183)
key_faces=np.all(key_slab[core_f],axis=1)&np.any(key_outer[core_f],axis=1)
stop_faces=np.all(abs(ct[:,:,2])<2e-9,axis=1)
pin_faces=(np.all(vertex_pin_dist[core_f]<.000374,axis=1)&np.all(ct[:,:,2]>=-.015001,axis=1)
           &np.all(ct[:,:,2]<=-.00946149,axis=1))
labels=np.full(len(core_f),'other_rigid_core',dtype=object)
labels[key_faces]='original_keys';labels[pin_faces]='metal_pin_or_retained_stem';labels[stop_faces]='body_front_shell_stop_z0'
tipmask=(abs(core_v[:,2]+.0094615)<3e-9)&(vertex_pin_dist<.0003722)
stopv=np.unique(core_f[stop_faces]);tipv=core_v[tipmask]
floor=np.asarray(geometry['socket'][0]);floor_z=float(np.median(floor[abs(floor[:,2]+.00541655)<3e-9,2]))
results=[];raw={}
for name,row in [('comparison_near_14p4mm',near),('final_after_free_hold',last)]:
    records=[]; transforms={}
    for i,part in enumerate(('core','nut')):
        q=np.asarray(row['quaternions_wxyz'][i]); T=np.eye(4)
        T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();T[:3,3]=row['positions_world_m'][i]
        T=np.linalg.inv(actual_socket_world)@T;transforms[part]=T
        obj=fcl.CollisionObject(models[part],fcl.Transform(T[:3,:3],T[:3,3]))
        request=fcl.CollisionRequest(num_max_contacts=20000,enable_contact=True)
        result=fcl.CollisionResult();n=fcl.collide(obj,socket_obj,request,result)
        ids=np.array([[c.b1,c.b2] for c in result.contacts],int).reshape(-1,2)
        points=np.array([c.pos for c in result.contacts],float).reshape(-1,3)
        pd=np.array([c.penetration_depth for c in result.contacts],float)
        groups={}
        for label in sorted(set(labels[ids[:,0]]) if part=='core' and len(ids) else {'nut_thread_or_nut_surface'} if len(ids) else {}):
            sel=(labels[ids[:,0]]==label) if part=='core' else np.ones(len(ids),bool)
            p=points[sel]; sf=geometry['socket'][1][ids[sel,1]]; own=geometry[part][1][ids[sel,0]]
            sv=geometry['socket'][0][sf].reshape(-1,3);ov=geometry[part][0][own].reshape(-1,3)
            groups[label]={'triangle_pair_records':int(sel.sum()),'unique_own_faces':int(len(np.unique(ids[sel,0]))),
                           'unique_socket_faces':int(len(np.unique(ids[sel,1]))),
                           'contact_point_socket_frame_bounds_m':[p.min(0).tolist(),p.max(0).tolist()],
                           'own_local_vertex_bounds_m':[ov.min(0).tolist(),ov.max(0).tolist()],
                           'socket_face_vertex_bounds_m':[sv.min(0).tolist(),sv.max(0).tolist()],
                           'fcl_reported_triangle_penetration_range_m':[float(pd[sel].min()),float(pd[sel].max())]}
        raw[name+'_'+part+'_triangle_pairs']=ids;raw[name+'_'+part+'_contact_points_socket_m']=points
        raw[name+'_'+part+'_triangle_reported_penetration_m']=pd
        records.append({'part':part,'triangle_intersection':bool(result.is_collision),'returned_pairs':len(ids),
                        'capacity':20000,'pair_buffer_saturated':len(ids)>=20000,'groups':groups})
    T=transforms['core'];tips=tipv@T[:3,:3].T+T[:3,3];stops=core_v[stopv]@T[:3,:3].T+T[:3,3]
    tip_d,tip_i=cKDTree(socket_centers).query(tips[:,:2]);
    # The actual deep Socket face at -14.605 mm is the rigid shoulder paired
    # with the source Body's z=0 shell annulus. These are axial plane gaps,
    # not triangle penetration-depth estimates.
    axial_stop_gaps=stops[:,2]+.014605
    results.append({'name':name,'source_step':row['step'],'time_s':row['time_s'],'phase':row['phase'],
                    'body_depth_m':row['body_depth_m'],'nut_angle_deg':row['nut_angle_deg'],
                    'body_axis_tilt_deg':float(np.rad2deg(np.arccos(np.clip(-T[2,2],-1,1)))),
                    'fcl_pairs':records,
                    'pin_tip_geometry':{'sample_count':len(tips),'associated_cavities':int(len(np.unique(tip_i))),
                        'actual_floor_z_m':floor_z,'pin_tip_to_floor_axial_gap_range_m':[float((tips[:,2]-floor_z).min()),float((tips[:,2]-floor_z).max())],
                        'maximum_tip_radius_from_associated_socket_center_m':float(tip_d.max()),
                        'minimum_radial_clearance_to_nominal_straight_bore_m':float(.00046355-tip_d.max())},
                    'shell_stop_axial_geometry':{'source_body_stop_plane_z_m':0.,'socket_stop_plane_z_m':-.014605,
                        'original_core_stop_face_count':int(stop_faces.sum()),
                        'axial_plane_gap_range_m':[float(axial_stop_gaps.min()),float(axial_stop_gaps.max())],
                        'source_stop_radius_range_m':[float(np.linalg.norm(core_v[stopv,:2],axis=1).min()),float(np.linalg.norm(core_v[stopv,:2],axis=1).max())]}})
report={'scope':'OFFLINE_EXACT_AUTHORED_TRIANGLE_GEOMETRY_AT_RECORDED_POSES_NOT_NATIVE_SDF_OR_FORCE_REPORT',
        'candidate':str(RUN/'connector_candidate.usdc'),'source_samples':str(RUN/'samples.jsonl'),
        'source_geometry_model':str(args.source_geometry_model) if args.source_geometry_model else None,
        'geometry_role':'Original metal geometry is transformed by actual candidate poses; partitioned candidate collision faces are not a substitute for checking original pins.',
        'geometry':{k:{'path':paths[k],'vertices':len(v),'faces':len(f),'local_bounds_m':[v.min(0).tolist(),v.max(0).tolist()]} for k,(v,f) in geometry.items()},
        'full_original_grasp_envelope_and_soft_seal_not_in_rigid_check':True,
        'original_key_faces_classified_not_deleted':int(key_faces.sum()),'results':results,
        'limits':['FCL mesh contact records describe intersecting triangle pairs, not reliable whole-solid minimum translation depth.',
                  'Axial pin-floor and shell-stop plane distances use actual transformed source vertices and actual authored floor; they are separate geometric measurements.',
                  'Triangle intersection retains existing original-key CAD overlap; it cannot be silently treated as zero.',
                  'No per-shape native contact force or SDF signed distance is inferred from these static FCL results.',
                  'The closest recorded 14.4 mm state is a trajectory comparison, not a teleported model state.']}
(ROOT/'references'/f'{args.output_prefix}_audit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(ROOT/'references'/f'{args.output_prefix}_triangle_pairs.npz',**raw)
print(json.dumps(report,indent=2))
