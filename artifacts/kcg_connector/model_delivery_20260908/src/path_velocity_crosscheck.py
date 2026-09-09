"""Offline compare independent native velocity views and pose increments."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/full_path_02'
rows=[]
keys=['step','time_s','phase','positions_world_m','quaternions_wxyz','native_linear_velocity_m_s',
      'native_angular_velocity_rad_s','native_link_velocity_crosscheck','nut_angle_deg','body_depth_m']
for line in (RUN/'samples.jsonl').open():
    try:r=json.loads(line)
    except json.JSONDecodeError:break
    rows.append({k:r[k] for k in keys})
stage=Usd.Stage.Open(str(RUN/'connector_candidate.usdc'))
paths=['/World/TE_J35FreeSplitPlug/Body','/World/TE_J35FreeSplitPlug/CouplingNut']
com=np.array([UsdPhysics.MassAPI(stage.GetPrimAtPath(p)).GetCenterOfMassAttr().Get() for p in paths],float)
t=np.array([r['time_s'] for r in rows]);step=np.array([r['step'] for r in rows]);phase=np.array([r['phase'] for r in rows])
pos=np.array([r['positions_world_m'] for r in rows]);q=np.array([r['quaternions_wxyz'] for r in rows])
v=np.array([r['native_linear_velocity_m_s'] for r in rows]);w=np.array([r['native_angular_velocity_rad_s'] for r in rows])
cross=[r['native_link_velocity_crosscheck'] for r in rows]
vp=np.array([r['link_com_linear_velocity_world_m_s'] for r in cross]);wp=np.array([r['link_angular_velocity_world_rad_s'] for r in cross])
pp=np.array([r['link_positions_world_m'] for r in cross]);qp=np.array([r['link_quaternions_xyzw'] for r in cross])
jointq=np.array([r['native_dof_positions_rad'][0] for r in cross]);jointv=np.array([r['native_dof_velocities_rad_s'][0] for r in cross])
Rs=[Rotation.from_quat(q[:,i,[1,2,3,0]]) for i in range(2)]
cp=pos+np.stack([rr.apply(np.broadcast_to(com[i],(len(rows),3))) for i,rr in enumerate(Rs)],axis=1)
dt=np.diff(t);fdv=np.diff(cp,axis=0)/dt[:,None,None]
fdw=np.stack([(rr[1:]*rr[:-1].inv()).as_rotvec()/dt[:,None] for rr in Rs],axis=1)
fdj=np.diff(jointq)/dt
per_phase=[]
for p in dict.fromkeys(phase):
    mask=(phase[1:]==p)&(phase[:-1]==p)
    ii=np.flatnonzero(phase==p);a,b=ii[0],ii[-1]
    trapezoid=np.sum((v[a:b]+v[a+1:b+1])*.5*dt[a:b,None,None],axis=0)
    qtrap=float(np.sum((jointv[a:b]+jointv[a+1:b+1])*.5*dt[a:b]))
    per_phase.append({'phase':p,'duration_s':float(t[b]-t[a]),'sample_count':len(ii),
        'COM_actual_net_displacement_m':(cp[b]-cp[a]).tolist(),
        'COM_reported_velocity_trapezoid_integral_m':trapezoid.tolist(),
        'COM_integral_displacement_difference_m':(trapezoid-(cp[b]-cp[a])).tolist(),
        'native_linear_minus_COM_pose_difference_rms_m_s':np.sqrt(np.mean((v[1:][mask]-fdv[mask])**2,axis=0)).tolist(),
        'native_angular_minus_orientation_difference_rms_rad_s':np.sqrt(np.mean((w[1:][mask]-fdw[mask])**2,axis=0)).tolist(),
        'joint_actual_angle_change_rad':float(jointq[b]-jointq[a]),'joint_reported_velocity_integral_rad':qtrap,
        'joint_velocity_minus_angle_difference_rms_rad_s':float(np.sqrt(np.mean((jointv[1:][mask]-fdj[mask])**2)))})
selected=[]
for target in [.4,5,11.5,25,45,50,55,60,62.5]:
    i=max(1,int(abs(t-target).argmin()))
    selected.append({'step':int(step[i]),'time_s':float(t[i]),'phase':str(phase[i]),
        'linear_reported_m_s':v[i].tolist(),'COM_pose_difference_m_s':fdv[i-1].tolist(),
        'angular_reported_rad_s':w[i].tolist(),'orientation_difference_rad_s':fdw[i-1].tolist(),
        'joint_reported_rad_s':float(jointv[i]),'joint_angle_difference_rad_s':float(fdj[i-1])})
report={'scope':'READ_ONLY_RECORDED_NATIVE_API_COMPARISON_NO_PHYSICS_OR_CONTROL_MUTATION',
        'source_samples':str(RUN/'samples.jsonl'),'sample_count':len(rows),'last_step':int(step[-1]),'last_time_s':float(t[-1]),
        'run_outcome':'ENDED_BY_RENDER_AUDIT_AT_STEP_15000; THESE ARE THE RECORDED PRE_RENDER_SAMPLES',
        'two_native_interface_differences':{
            'linear_velocity_max_absolute_m_s':float(abs(v-vp).max()),
            'angular_velocity_max_absolute_rad_s':float(abs(w-wp).max()),
            'link_position_max_absolute_m':float(abs(pos-pp).max()),
            'quaternion_xyzw_max_absolute_after_order_conversion':float(abs(q[:,:,[1,2,3,0]]-qp).max())},
        'com_local_m':com.tolist(),'pose_differencing':'x_COM = actor_position + R_actor * local_COM; angular = log(R_next * inverse(R_prev))/dt in world frame',
        'per_phase':per_phase,'selected_samples':selected,
        'official_sources':[
          {'url':'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html',
           'supports':'RigidBodyView may wrap articulation links. Both velocity getters report world velocities with linear component at COM.'},
          {'url':'https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#solver-iterations',
           'supports':'Position and velocity iterations use different bias terms. Reported carried velocity need not equal transform difference over the simulation step.'},
          {'url':'https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf',
           'supports':'TGS last-substep joint velocity may differ substantially from full-timestep joint velocity, especially with high stiffness.'}],
        'conclusion':'Both native interfaces agree exactly in recorded data; the mismatch remains after correct COM and world-frame conversion. This excludes a discrepancy unique to the RigidPrim view in this run. It is compatible with documented constrained-solver position/velocity semantics, not sufficient on its own to diagnose model geometry failure or a solver bug.',
        'limits':['Not a bound on acceptable solver error magnitude','Not a proof of hardware fidelity or contact energy passivity',
                  'Do not replace native velocity by pose finite differences in force/momentum balance or feed this posthoc measurement to online control',
                  'Pose increments establish actual path; reported velocities characterize the solver-carried state. Keep both with names and definitions.',
                  'Rendering abort is separate and is not disproven by these pre-render API matches']}
(ROOT/'references/path_velocity_crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(ROOT/'references/path_velocity_crosscheck_arrays.npz',step=step,time_s=t,phase=phase,
    positions=pos,COM_positions=cp,native_linear=v,native_angular=w,COM_pose_fd=fdv,orientation_fd=fdw,
    joint_position=jointq,native_joint_velocity=jointv,joint_angle_fd=fdj)
print(json.dumps({'sample_count':len(rows),'differences':report['two_native_interface_differences'],
                  'rotation':next(x for x in per_phase if x['phase']=='rotation'),'last_sample':selected[-1]},indent=2))
