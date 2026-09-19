"""The four declared perception cameras, separate from evidence video views."""
from pathlib import Path
import copy
import numpy as np
import yaml


def configuration(repository, runtime):
    assembly=runtime.get('body_assembly_control_config')
    if not assembly:return None
    path=Path(assembly)
    if not path.is_absolute():path=Path(repository)/path
    vision=yaml.safe_load(path.read_text()).get('perception',{})
    rig=vision.get('four_camera_rig')
    if not rig:return None
    source=Path(repository)/rig
    data=yaml.safe_load(source.read_text())
    if (data.get('schema_version')!='kcg_four_camera_assembly_v1'
            or data.get('simulation_only') is not True
            or data.get('global_camera_poses_may_change_during_episode') is not False
            or data.get('maximum_key_observation_events') not in (1,2)):
        raise ValueError('Four-camera mode requires a fixed rig and one or two declared key observations')
    if data['maximum_key_observation_events']==2:
        if data.get('key_observation_policy')!='COARSE_AXIAL_TURN_THEN_REOBSERVE':
            raise ValueError('Two observations require explicit coarse-turn/reobservation semantics')
        if not 0<float(data.get('maximum_refinement_rotation_deg',0))<=1.:
            raise ValueError('Refinement must remain within the declared one-degree small-motion bound')
    return data


def camera_spec(repository, rig, role, hand=None):
    """Global poses use installation values; hand poses use encoder FK only."""
    from te_foundationpose_handoff_runtime import _camera_cv_pose_from_eye_target
    root=Path(repository)
    if role=='global_1':
        spec=copy.deepcopy(yaml.safe_load((root/rig[role]['source_config']).read_text())['camera'])
        pose=_camera_cv_pose_from_eye_target(spec['eye_world_m'],spec['target_world_m'])
    elif role=='global_2':
        spec=copy.deepcopy(rig[role])
        pose=_camera_cv_pose_from_eye_target(spec['eye_world_m'],spec['target_world_m'])
    elif role in ('palm','wrist'):
        from te_body_socket_observation import hand_camera_mount
        if hand is None:raise ValueError('Hand-mounted camera requires current encoder hand pose')
        mount=hand_camera_mount(root,role)
        if mount['source_config']!=str(root/rig['mount_source_config']):
            raise ValueError('Hand camera mount differs from the declared rig')
        spec={k:mount[k] for k in ('resolution_px','focal_length_mm','horizontal_aperture_mm','clipping_range_m')}
        spec.update(prim_path=rig[role]['prim_path'],mount=mount)
        pose=np.asarray(hand,float).reshape(4,4)@np.asarray(mount['hand_from_camera_cv'],float)
    else:raise ValueError('Perception camera must be one of the four declared roles')
    return spec,np.asarray(pose,float).reshape(4,4)


def intrinsics(spec):
    width,height=map(int,spec['resolution_px'])
    focal=width*float(spec['focal_length_mm'])/float(spec['horizontal_aperture_mm'])
    return np.array([[focal,0.,width/2],[0.,focal,height/2],[0.,0.,1.]])
