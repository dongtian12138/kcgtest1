"""Image/CAD-only prototype; no simulator state or controller mutation."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def prepare_template(v,faces):
    tri=v[faces]
    keep=(tri[:,:,2].max(axis=1)>-.022)&(tri[:,:,2].min(axis=1)<-.0175)&(np.linalg.norm(tri[:,:,:2],axis=2).max(axis=1)>.0205)
    tri=tri[keep];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    rng=np.random.default_rng(20260915);ids=rng.choice(len(tri),500000,p=area/area.sum());a=np.sqrt(rng.random(len(ids)));b=rng.random(len(ids))
    sample=(1-a[:,None])*tri[ids,0]+(a*(1-b))[:,None]*tri[ids,1]+(a*b)[:,None]*tri[ids,2]
    radius=np.linalg.norm(sample[:,:2],axis=1);sample=sample[(radius>.0205)&(radius<.0245)&(sample[:,2]>-.022)&(sample[:,2]<-.0175)]
    return cKDTree(sample)

def extract_points(observation,depth,intrinsics,world_from_socket,offset_m=.0005):
    B=np.asarray(observation['world_from_plug_five_dof']);C=np.asarray(observation['world_from_camera_cv']);S=np.asarray(world_from_socket).reshape(4,4);K=np.asarray(intrinsics)
    z=B[:3,2];x=-S[:3,0];x=x-z*(z@x);x/=np.linalg.norm(x);R=np.column_stack((x,np.cross(z,x),z));t=B[:3,3]+offset_m*z
    yy,xx=np.indices(depth.shape);pc=np.stack(((xx+.5-K[0,2])*depth/K[0,0],(yy+.5-K[1,2])*depth/K[1,1],depth),-1)
    points=(pc@C[:3,:3].T+C[:3,3]-t)@R;rad=np.linalg.norm(points[...,:2],axis=2)
    mask=np.isfinite(depth)&(depth>0)&(rad>.0208)&(rad<.0242)&(points[...,2]>-.025)&(points[...,2]<-.018)
    return points[mask],R,t

def fit_phase(tree,points,prior_deg,half_width_deg=8.):
    if len(points)<500:raise ValueError('Insufficient Nu exterior depth points')
    sampled=points[::max(1,len(points)//1800)]
    def score(deg):
        t=np.radians(deg);c,s=np.cos(t),np.sin(t);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        distance=tree.query(sampled@R,workers=1)[0]
        return {'yaw_deg':float(deg),'clipped_rms_m':float(np.sqrt(np.mean(np.minimum(distance,.0005)**2))),
                'median_distance_m':float(np.median(distance)),'p90_distance_m':float(np.percentile(distance,90)),
                'fraction_below0p1mm':float(np.mean(distance<.0001))}
    lower=.25*np.ceil((prior_deg-half_width_deg)/.25)
    upper=.25*np.floor((prior_deg+half_width_deg)/.25)
    coarse=[score(t) for t in np.arange(lower,upper+.001,.25)]
    best=min(coarse,key=lambda r:r['clipped_rms_m'])
    fine=[score(t) for t in np.arange(best['yaw_deg']-.3,best['yaw_deg']+.3001,.025)]
    best=min(fine,key=lambda r:r['clipped_rms_m'])
    alternatives=[score(best['yaw_deg']+delta) for delta in (-3,3)]
    return {'best':best,'selected_points':len(points),'fit_points':len(sampled),'coarse_score_curve':coarse,
            'fine_score_curve':fine,'three_degree_alternatives':alternatives,'prior_deg':float(prior_deg),
            'search_half_width_deg':half_width_deg,'online_object_truth_used':False}


def _source_template(repository, settings, cache):
    import hashlib
    from pxr import Usd, UsdGeom
    model=(Path(repository)/settings['source_model']).resolve()
    key=str(model)
    if cache.get('template_key')!=key:
        digest=hashlib.sha256(model.read_bytes()).hexdigest()
        if settings.get('source_model_sha256') and digest!=settings['source_model_sha256']:
            raise ValueError('Nu phase source model differs from its declared geometry')
        stage=Usd.Stage.Open(str(model));root='/World/TE_J35FreeSplitPlug/CouplingNut'
        prim=stage.GetPrimAtPath(root+'/RepresentativeThreadVisual');mesh=UsdGeom.Mesh(prim)
        if not prim or str(UsdGeom.Imageable(prim).ComputeVisibility())=='invisible':
            raise ValueError('The declared visible Nu source surface is unavailable')
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts==3):raise ValueError('Nu phase template requires source triangles')
        vertices=np.asarray(mesh.GetPointsAttr().Get(),float)
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        transforms=UsdGeom.XformCache();parent=np.asarray(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T
        local=np.linalg.inv(parent)@np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        vertices=vertices@local[:3,:3].T+local[:3,3]
        cache.update(template_key=key,template_sha256=digest,tree=prepare_template(vertices,faces))
    return cache['tree']


def estimate_for_grasp(repository, observation, world_from_socket, actual_hand,
                       cumulative_command_deg, settings, cache):
    from scipy.spatial.transform import Rotation
    from te_body_socket_observation import hand_camera_mount
    if not observation.get('position_and_axis_measured'):
        raise ValueError('Current image-derived Body position and axis are required')
    if observation.get('rgbd_directory'):
        rgbd=Path(observation['rgbd_directory'])
        frame=rgbd.parent
    elif observation.get('tracking_seed_mask'):
        frame=Path(observation['tracking_seed_mask']).parent
        rgbd=frame/'rgbd'
    else:
        raise ValueError('The current RGB-D frame path is required')
    depth_file=rgbd/'depth_m.npy'
    if not depth_file.is_file():raise ValueError('The current RGB-D frame is unavailable')
    if not 0.<float(settings.get('search_half_width_deg',8.))<22.5:
        raise ValueError('Nu phase search must remain within one local exterior phase interval')
    K=np.asarray(hand_camera_mount(repository,'palm')['intrinsics_3x3'])
    points,reference,origin=extract_points(observation,np.load(depth_file),K,world_from_socket,
        offset_m=float(settings.get('captured_nut_offset_m',.0005)))
    target_yaw=float(settings['reference_target_hand_nut_yaw_deg'])
    hand_x=reference.T@np.asarray(actual_hand)[:3,0]
    prior=np.degrees(np.arctan2(hand_x[1],hand_x[0]))-target_yaw
    prior+=90.*round((float(cumulative_command_deg)-prior)/90.)
    fit=fit_phase(_source_template(repository,settings,cache),points,prior,
        half_width_deg=float(settings.get('search_half_width_deg',8.)))
    best=fit['best'];ratio=min(r['clipped_rms_m'] for r in fit['three_degree_alternatives'])/max(best['clipped_rms_m'],1e-12)
    quality={'enough_exterior_points':len(points)>=500,
        'within_search_window':abs(best['yaw_deg']-prior)<float(settings.get('search_half_width_deg',8.))-.4,
        'source_surface_fit':best['clipped_rms_m']<=float(settings.get('maximum_fit_rms_m',.0001)),
        'three_degree_alternative_distinguishable':ratio>=float(settings.get('minimum_three_degree_score_ratio',1.2))}
    nut_rotation=reference@Rotation.from_euler('z',best['yaw_deg'],degrees=True).as_matrix()
    return {'scope':'CURRENT_RGBD_NU_EXTERIOR_LOCAL_PHASE_ESTIMATE','source_frame':str(frame.resolve()),
        'capture_physics_time_s':observation['capture_physics_time_s'],
        'source_model_sha256':cache['template_sha256'],'phase_fit':fit,'quality_conditions':quality,
        'quality_passed':bool(all(quality.values())),'alternative_score_ratio':float(ratio),
        'estimated_world_from_nut_rotation':nut_rotation.tolist(),'reference_origin_world_m':origin.tolist(),
        'reference_target_hand_nut_yaw_deg':target_yaw,
        'global_revolution_count_measured':False,'exterior90degree_equivalence_used':True,
        'object_or_contact_truth_used':False}


def grasp_target_from_phase(base_target,canonical,phase,maximum_increment_deg=5.):
    from scipy.spatial.transform import Rotation
    if not phase['quality_passed']:raise ValueError('Current Nu phase image fit is not distinguishable')
    if not np.isfinite(maximum_increment_deg) or not 0.<maximum_increment_deg<=5.:
        raise ValueError('Visual grasp yaw correction is bounded by five degrees')
    base=np.asarray(base_target)@np.linalg.inv(np.asarray(canonical))
    N=np.asarray(phase['estimated_world_from_nut_rotation']);angle=np.radians(phase['reference_target_hand_nut_yaw_deg'])
    wanted=N@np.array([np.cos(angle),np.sin(angle),0.])
    alpha=np.degrees(np.arctan2(canonical[1,0],canonical[0,0]))
    increment=(np.degrees(np.arctan2(base[:3,1]@wanted,base[:3,0]@wanted))-alpha+45.)%90.-45.
    if not np.isfinite(increment) or abs(increment)>maximum_increment_deg:
        raise ValueError('Current visual Nu phase needs more than the bounded grasp correction')
    frame=base.copy();frame[:3,:3]=base[:3,:3]@Rotation.from_euler('z',increment,degrees=True).as_matrix()
    return frame@canonical,float(increment)


def grasp_phase_error_deg(actual_hand,phase):
    relative=np.asarray(phase['estimated_world_from_nut_rotation']).T@np.asarray(actual_hand)[:3,:3]
    angle=np.degrees(np.arctan2(relative[1,0],relative[0,0]))
    return float((angle-phase['reference_target_hand_nut_yaw_deg']+45.)%90.-45.)
