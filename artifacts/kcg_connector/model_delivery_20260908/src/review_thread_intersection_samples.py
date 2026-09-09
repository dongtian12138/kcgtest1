"""Bounded offline check of the largest FCL triangle-overlap reports."""
from pathlib import Path
import json
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from scipy.spatial.transform import Rotation
import argparse
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=root/'runs/full_path_maximal_cpu_01')
parser.add_argument('--pairs',type=Path,default=root/'references/takeover_full_path_maximal_cpu_01_triangle_pairs.npz')
args=parser.parse_args();run=args.run
stage=Usd.Stage.Open(str(run/'connector_candidate.usdc'))
rows_path=run/'samples.jsonl'
last=None
for line in rows_path.open():last=json.loads(line)
socket_root='/World/TEVisualHandoff/FixedReceptaclePose'
nut_root='/World/TE_J35FreeSplitPlug/CouplingNut'
def transform(path):return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
mesh_data={}
for name,path,frame in [('socket',socket_root+'/OfficialVisual/Geometry',socket_root),('nut',nut_root+'/SourceCadCollision',nut_root)]:
 m=UsdGeom.Mesh.Get(stage,path);v=np.asarray(m.GetPointsAttr().Get(),float);f=np.asarray(m.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3);t=np.linalg.inv(transform(frame))@transform(path);v=v@t[:3,:3].T+t[:3,3];mesh_data[name]=(v,f)
a=np.load(args.pairs)
ids=a['final_after_free_hold_nut_triangle_pairs'];depth=a['final_after_free_hold_nut_triangle_reported_penetration_m'];order=np.argsort(depth)[-20:]
v,f=mesh_data['nut'];tri=v[f[ids[order,0]]];samples=np.concatenate([tri.reshape(-1,3),tri.mean(1),(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2])
q=np.asarray(last['quaternions_wxyz'][1]);t=np.eye(4);t[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();t[:3,3]=last['positions_world_m'][1];t=np.linalg.inv(transform(socket_root))@t;samples=samples@t[:3,:3].T+t[:3,3]
sv,sf=mesh_data['socket'];mesh=trimesh.Trimesh(sv*1000,sf,process=False)
signed=trimesh.proximity.signed_distance(mesh,samples*1000)/1000
report={'scope':'EXACT_SOURCE_MESH_SIGNED_DISTANCE_AT_SAMPLED_POINTS_NOT_GLOBAL_PENETRATION_BOUND','run':str(run),'step':last['step'],'fcl_max_triangle_overlap_report_m':float(depth.max()),'sampled_triangle_count':20,'sample_count':len(samples),'maximum_sampled_inside_socket_distance_m':float(max(0,signed.max())),'signed_distance_range_m':[float(signed.min()),float(signed.max())],'inside_positive':True,'socket_watertight':bool(mesh.is_watertight),'caveat':'Samples only; FCL triangle penetration is not whole-solid penetration. Source model was not changed.'}
(run/'thread_source_distance_check.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(run/'thread_source_distance_check.npz',points_socket_m=samples,signed_distance_m=signed,fcl_selected_pairs=ids[order],fcl_selected_overlap_m=depth[order]);print(json.dumps(report,indent=2))
