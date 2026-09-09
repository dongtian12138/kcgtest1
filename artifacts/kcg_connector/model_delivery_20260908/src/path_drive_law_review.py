"""Read-only D6 terminal-state law and whole-connector moment comparison."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/full_path_01'
stage=Usd.Stage.Open(str(RUN/'test_apparatus_before_physics.usdc'))
guide=UsdPhysics.Joint.Get(stage,'/World/DeclaredModelVerificationGuide')
q=guide.GetLocalRot0Attr().Get();frame=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
axis=frame[:,2];origin=np.asarray(guide.GetLocalPos0Attr().Get(),float)
drive=UsdPhysics.DriveAPI(guide.GetPrim(),'rotZ')
K=float(drive.GetStiffnessAttr().Get())*180/np.pi;D=float(drive.GetDampingAttr().Get())*180/np.pi
cap=float(drive.GetMaxForceAttr().Get())
inertias=[];masses=[];coms=[]
for path in ['/World/TE_J35FreeSplitPlug/Body','/World/TE_J35FreeSplitPlug/CouplingNut']:
    m=UsdPhysics.MassAPI(stage.GetPrimAtPath(path));q=m.GetPrincipalAxesAttr().Get()
    pr=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
    inertias.append(pr@np.diag(m.GetDiagonalInertiaAttr().Get())@pr.T)
    masses.append(m.GetMassAttr().Get());coms.append(m.GetCenterOfMassAttr().Get())
inertias=np.asarray(inertias);masses=np.asarray(masses);coms=np.asarray(coms)
results=[];last=None
for line in (RUN/'samples.jsonl').open():
    r=json.loads(line)
    if not 15160<=r['step']<=15400:continue
    R=Rotation.from_quat(np.asarray(r['quaternions_wxyz'])[:,[1,2,3,0]]).as_matrix()
    p=np.asarray(r['positions_world_m']);v=np.asarray(r['native_linear_velocity_m_s']);w=np.asarray(r['native_angular_velocity_rad_s'])
    c=p+np.einsum('nij,nj->ni',R,coms);Iw=R@inertias@np.transpose(R,(0,2,1))
    H=(np.einsum('nij,nj->ni',Iw,w)+np.cross(c-origin,masses[:,None]*v)).sum(0)
    wc=np.asarray(r['normal_wrenches_n_nm'])+np.asarray(r['friction_wrenches_n_nm'])
    contactM=(wc[:,3:]+np.cross(p-origin,wc[:,:3])).sum(0)
    gravityM=np.cross(c-origin,masses[:,None]*np.array([0.,0.,-9.81])).sum(0)
    target=Rotation.from_euler('z',r['command_deg'],degrees=True).as_matrix()
    delta=target.T@(frame.T@R[1]);error=-float(delta[1,0])
    spring=K*error;damping=-D*float(w[1]@axis)
    if last is not None:
        hdot=float((H-last[0])@axis/(r['time_s']-last[1]))
        inferred=hdot-float((contactM+gravityM)@axis)
        results.append({'step':r['step'],'time_s':r['time_s'],
            'exact_pure_rotZ_geometric_error':error,'terminal_spring_law_nm':spring,
            'terminal_damping_law_nm':damping,'terminal_PD_with_cap_nm':float(np.clip(spring+damping,-cap,cap)),
            'recorded_spring_angle_estimate_nm':r['drive_spring_only_estimate_nm'],
            'nut_socket_contact_axis_torque_nm':float(wc[1,3:]@axis),
            'whole_contact_axis_moment_about_fixed_guide_origin_nm':float(contactM@axis),
            'whole_gravity_axis_moment_nm':float(gravityM@axis),
            'whole_native_angular_momentum_change_per_step_nm':hdot,
            'inferred_average_rotary_actuator_torque_nm_if_other_guide_axial_moment_zero':inferred})
    last=(H,r['time_s'])
summary={'scope':'OFFLINE_CONSTRAINT_LAW_REVIEW_NOT_NATIVE_ACTUATOR_READBACK',
    'source_run':str(RUN),'usd_stiffness_per_degree':float(drive.GetStiffnessAttr().Get()),
    'usd_damping_per_degree_per_second':float(drive.GetDampingAttr().Get()),'sdk_stiffness_nm_rad':K,
    'sdk_damping_nm_s_rad':D,'cap_nm':cap,
    'source_code_result':{'rotZ_path':'D6 swing / swing2; rotX and rotY locked, so SLERP inactive',
        'geometric_error':'-element[1,0] of R_target.T @ R_relative; pure Z = sin(targetAngle-actualAngle)',
        'half_angle_factor_in_this_drive':False,
        'usd_to_sdk':'multiply angular stiffness and damping by 180/pi; no factor 1/2',
        'tgs_erp_half_factor_applies_to_spring':False,
        'snapshot_formula_limit':'Terminal-state PD evaluated using reported pose/velocity is not the mean applied impulse divided by dt; internal substep states and constraint impulses were not recorded.'},
    'selected_samples':[x for x in results if x['step'] in [15272,15284,15296]],
    'decision':'Do not divide stiffness by two or change force parameters. Keep measured contact torque separate from terminal-state PD and finite cap. An exact applied drive torque requires native drive constraint impulse or substep solve data, not present in these logs.',
    'angular_momentum_inference_limits':['Assumes external guide constraints have negligible projection of moment along the free rotation axis. Their reactions are not measured here.',
        'Original internal Body-Nut forces/friction cancel in whole-connector angular momentum balance.',
        'Recorded world COM velocity and inertia were used; pose finite differences were not substituted for native velocity.',
        'Source checked in official PhysX 5.6.1 and current main; not claimed to be a binary symbol-level proof of every local proprietary patch.']}
(ROOT/'references/path_drive_law_review.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'references/path_drive_law_window.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(summary,indent=2))
