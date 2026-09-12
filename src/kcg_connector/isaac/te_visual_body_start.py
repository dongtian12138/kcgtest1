"""Current RGB-D axis localization for the first Body grasp of an episode.

Reuse the existing image-only estimator. Its cylinder-axis observation is
not a measured key angle: the initial grasp selects an axial frame, and a
new key observation is still required after pickup before insertion.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_nominal_scene_scope(repository,reference_scene,current_scene,current_config):
    """Compare the physical setup used by nominal preflight, not later tasks."""
    repository=Path(repository);current_path=Path(current_config)
    if not current_path.is_absolute():current_path=repository/current_path
    current_document=yaml.safe_load(current_path.read_text())
    candidates=[]
    for relative,sha in reference_scene.items():
        path=repository/relative
        if path.suffix not in ('.yaml','.yml'):continue
        if not path.is_file() or digest(path)!=sha:
            raise ValueError('The preserved preflight configuration changed: '+str(path))
        document=yaml.safe_load(path.read_text())
        if document.get('schema_version')==current_document['schema_version']:
            candidates.append((relative,document))
    if len(candidates)!=1:raise ValueError('A unique preserved assembly scene configuration is required')
    source_path,source=candidates[0]
    keys=('collision','physics_numerics','passive_joint_solver','grounding_band_contact_model','validated_connector')
    old={k:source[k] for k in keys};new={k:current_document[k] for k in keys}
    if old!=new:raise ValueError('Nominal preflight physical scene differs from this candidate')
    left=dict(reference_scene);right=dict(current_scene)
    left.pop(source_path);right.pop(str(current_path.resolve().relative_to(repository.resolve())))
    signature=hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest()
    left['ASSEMBLY_PHYSICAL_CONFIGURATION']=signature;right['ASSEMBLY_PHYSICAL_CONFIGURATION']=signature
    return left,right,{'reference_configuration':source_path,'current_configuration':str(current_path),
        'physical_fields_compared':list(keys),'physical_parameters_match':True,
        'later_controller_settings_are_not_preflight_results':True}


def body_grasp_frame(axis_pose):
    if axis_pose.get("status") != "OBSERVED_AXIS_POSITION_YAW_FREE":
        raise ValueError("The current image did not resolve the supported plug axis")
    position=np.asarray(axis_pose["position_xyz_m"],float)
    z=np.asarray(axis_pose["outward_axis_world"],float)
    if position.shape!=(3,) or z.shape!=(3,) or not np.isfinite(np.r_[position,z]).all() or np.linalg.norm(z)<1e-9:
        raise ValueError("Finite measured plug axis and position are required")
    z=z/np.linalg.norm(z)
    hint=np.array([1.,0.,0.]) if abs(z[0])<.9 else np.array([0.,1.,0.])
    x=hint-z*float(z@hint);x/=np.linalg.norm(x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack((x,np.cross(z,x),z));pose[:3,3]=position
    return pose


def observe_tabletop_body(repository, runtime, output):
    from pxr import Gf,UsdGeom
    import omni.usd
    import omni.replicator.core as rep
    from te_foundationpose_handoff_runtime import (
        _author_camera,_camera_cv_pose_from_eye_target,_capture_rgbd,_close_rgbd_resources)
    from kcg_connector.te_rgbd_observability import EndpointWorkspace,evaluate_te_rgbd_observability
    from kcg_connector.te_rgbd_pose_provider import run_te_rgbd_pose_provider

    repository=Path(repository);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    world=runtime["world"];stage=omni.usd.get_context().get_stage()
    camera_path=repository/'src/kcg_connector/config/te_rgbd_camera_near_side_v1.yaml'
    camera=yaml.safe_load(camera_path.read_text())["camera"]
    source=json.loads((repository/'src/kcg_connector/config/te_same_reset_rgbd_observe_v1.json').read_text())["frozen_sources"]
    template_path=repository/source["provider_input_template"]
    template=json.loads(template_path.read_text())
    static_path=repository/source["static_background_depth"]
    if digest(template_path)!=source["provider_input_template_sha256"] or digest(static_path)!=source["static_background_depth_sha256"]:
        raise ValueError("The preserved camera/background/provider source changed")
    pose=_camera_cv_pose_from_eye_target(camera["eye_world_m"],camera["target_world_m"])
    expected=np.asarray(template["camera_calibration"]["world_from_camera_cv_row_major"]).reshape(4,4)
    if not np.allclose(pose,expected,rtol=0,atol=1e-12):raise ValueError("Initial RGB-D extrinsics differ from calibrated view")
    K=np.asarray(template["camera_calibration"]["intrinsics_3x3"])
    world.pause();before=float(world.current_time)
    _author_camera(stage,camera["prim_path"],pose,resolution=tuple(camera["resolution_px"]),
        focal_length_mm=camera["focal_length_mm"],horizontal_aperture_mm=camera["horizontal_aperture_mm"],
        clipping_range_m=tuple(camera["clipping_range_m"]),Gf=Gf,UsdGeom=UsdGeom)
    resources={};started=datetime.now(timezone.utc)
    try:
        world.render()
        capture=_capture_rgbd(rep=rep,resources=resources,camera_path=camera["prim_path"],
            resolution=tuple(camera["resolution_px"]),output_dir=output/'observation',warmup_frames=5,rt_subframes=4)
    finally:
        _close_rgbd_resources(resources)
    if abs(float(world.current_time)-before)>1e-9:raise RuntimeError("Initial perception capture advanced physics")
    from PIL import Image
    rgb=np.asarray(Image.open(output/'observation/rgb.png'));depth=np.load(output/'observation/depth_m.npy')
    static=np.load(static_path)
    (output/'background').mkdir()
    local_static_path=output/'background/depth_m.npy'
    shutil.copyfile(static_path,local_static_path)
    ws=template["frozen_endpoint_workspaces_world_aabb_m"]["plug"]
    observation=evaluate_te_rgbd_observability(rgb=rgb,depth_m=depth,static_depth_m=static,
        intrinsics=K,world_from_camera=pose,
        workspaces=(EndpointWorkspace(name="plug",minimum_world_m=tuple(ws["minimum"]),
            maximum_world_m=tuple(ws["maximum"]),minimum_points=150,smallest_key_chord_m=.0013208),),
        minimum_key_width_px=4.,minimum_foreground_depth_delta_m=.00025)
    manifest=copy.deepcopy(template)
    manifest.update(provider_scope="TRANSPORT_PLUG_ONLY",capture_id="body_start_"+uuid.uuid4().hex,
                    capture_contract_sha256=digest(camera_path),control_authorized=False,pose_result=None)
    manifest["capture_time"].update(capture_started_at_utc=started.isoformat(),
                                    observed_frame_timestamp_utc=datetime.now(timezone.utc).isoformat())
    manifest["observability"]=observation
    manifest["frozen_endpoint_workspaces_world_aabb_m"]={"plug":ws}
    manifest["te_cad_models"]={"plug":template["te_cad_models"]["plug"]}
    for field,path,encoding,shape in (
        ("ordinary_rgb",output/'observation/rgb.png',"png_rgb_uint8",rgb.shape),
        ("ordinary_depth",output/'observation/depth_m.npy',"npy_distance_to_image_plane_m_float32",depth.shape),
        ("ordinary_static_scene_depth",local_static_path,"npy_distance_to_image_plane_m_float32",static.shape)):
        import os
        manifest[field]={"encoding":encoding,"path_relative_to_manifest":os.path.relpath(path,output),
                         "sha256":digest(path),"shape":list(shape)}
    manifest["formal_gate"]={"required_outputs":["transport_grasp_pose"],"key_observation_required":False,
        "miss_authorizes_motion":False,"pose_result_from_scene_generation_truth_forbidden":True,
        "missing_endpoint_result":"MISS_ENDPOINT_NOT_FOUND","ambiguous_key_result":"NOT_APPLICABLE_TO_TRANSPORT_PLUG_ONLY",
        "unobserved_key_result":"NOT_APPLICABLE_TO_TRANSPORT_PLUG_ONLY"}
    manifest["provider_callable_contract"]={"implementation_status":"TE_TRANSPORT_PLUG_ONLY_SAME_RESET_V1",
        "input":"current ordinary RGB/depth and frozen calibration/background/CAD","output":"axis position without measured key yaw"}
    input_path=output/'provider_input.json';input_path.write_text(json.dumps(manifest,indent=2)+'\n')
    result=run_te_rgbd_pose_provider(input_path,repository)
    result_path=output/'pose_provider_result.json';result_path.write_text(json.dumps(result,indent=2)+'\n')
    axis=result["transport_grasp_pose"];body=body_grasp_frame(axis)
    record={"source":"CURRENT_RGBD_WITH_CAD_COAXIALITY_AND_STATIC_TABLE_SUPPORT_PRIOR",
        "capture":capture,"physics_time_s":before,"provider_result":str(result_path),
        "robot_sample_step":int(runtime["nail_body_ft_auditor"].samples[-1]["step"]),
        "provider_result_sha256":digest(result_path),"world_from_body_for_initial_grasp":body.tolist(),
        "key_angle_measured":False,"axial_frame_yaw_selected_for_grasp_not_measured":True,
        "required_later_key_observation":True,"captive_nut_axial_play_not_observed_by_ring":True,
        "body_grasp_must_be_verified_with_its_source_contact_region":True,
        "object_pose_or_semantic_or_contact_truth_read":False,"reset_after_frame":False}
    (output/'body_localization.json').write_text(json.dumps(record,indent=2)+'\n')
    world.play()
    return body,record


def check_initial_approach(repository,runtime,plan,observed_body,initial_hand):
    """Check the open-hand trajectory against CAD placed by the image.

    The tabletop is known static geometry. The socket's whole search volume
    is avoided here; its actual pose still requires later current images.
    This discrete check does not substitute for physical contact monitoring.
    """
    import fcl
    from pxr import Usd,UsdGeom
    from te_foundationpose_handoff_plan import FullRobotCollisionScene,_fcl_distance,_fcl_model
    from te_body_nut_regrasp import body_yaw_swept_bound

    scene=FullRobotCollisionScene(runtime["inputs"])
    geometry_path=repository/'artifacts/kcg_connector/isaac/te_full_assembly_20260905/nut_regrasp_01/cooked_finger_inspection/geometry_only_manifest.json'
    geometry=json.loads(geometry_path.read_text());mesh_path=Path(geometry["mesh_data"])
    if digest(runtime["robot_asset"])!=geometry["source_robot_asset_sha256"] or digest(mesh_path)!=geometry["mesh_data_sha256"]:
        raise ValueError("Initial approach geometry differs from the original robot")
    with np.load(mesh_path) as meshes:
        for name in ("f1Link3","f2Link2","f3Link3"):
            scene.objects[name]=fcl.CollisionObject(_fcl_model(meshes[name]))
    source=json.loads((repository/'src/kcg_connector/config/te_same_reset_rgbd_observe_v1.json').read_text())["frozen_sources"]
    template=json.loads((repository/source["provider_input_template"]).read_text())
    obstacles={}
    for name,box in template["known_static_scene_geometry"].items():
        obstacles[name]=fcl.CollisionObject(fcl.Box(*box["size_m"]),fcl.Transform(np.eye(3),box["center_world_m"]))
    for name,box in template["frozen_endpoint_workspaces_world_aabb_m"].items():
        if name=="plug":continue
        lo,hi=np.asarray(box["minimum"]),np.asarray(box["maximum"])
        obstacles[name+"_search_volume"]=fcl.CollisionObject(fcl.Box(*(hi-lo)),fcl.Transform(np.eye(3),(hi+lo)/2))
    # Read the frozen local CAD only, never live-stage rigid transforms.
    stage=Usd.Stage.Open(str(repository/'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'))
    for part,mesh in (("Body","SourceCadCollision"),("CouplingNut","ExternalSurfaceContact")):
        cad=UsdGeom.Mesh(stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/'+part+'/'+mesh))
        v=np.asarray(cad.GetPointsAttr().Get(),float);f=np.asarray(cad.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
        envelope,_=body_yaw_swept_bound(v,f,margin_m=.00025)
        obstacles[part]=fcl.CollisionObject(_fcl_model(envelope.triangles),
            fcl.Transform(observed_body[:3,:3],observed_body[:3,3]))
    opening=np.asarray(plan["pregrasp_hand_positions_rad"]);approach=np.asarray(plan["approach_arm_waypoints_rad"])
    states=[(np.zeros(7),(1.-t)*initial_hand+t*opening) for t in np.linspace(0.,1.,41)]
    states.extend((t*approach[0],opening) for t in np.linspace(0.,1.,161))
    for a,b in zip(approach[:-1],approach[1:]):
        states.extend(((1.-t)*a+t*b,opening) for t in np.linspace(0.,1.,6))
    minima={"self":float('inf'),**{name:float('inf') for name in obstacles}}
    limiting=None
    for index,(arm,hand) in enumerate(states):
        d,pair=scene.state_clearance(arm,hand)
        if d<minima["self"]:minima["self"]=d
        if d<=0:raise ValueError(f"Initial visual approach self collision at {index}: {pair}")
        for name,obj in scene.objects.items():
            for obstacle,other in obstacles.items():
                distance=_fcl_distance(obj,other)
                minima[obstacle]=min(minima[obstacle],distance)
                if distance<.0001:
                    limiting={"state":index,"link":name,"obstacle":obstacle,"clearance_m":distance}
                    raise ValueError("Initial visual approach has insufficient clearance: "+json.dumps(limiting))
    return {"status":"DISCRETE_SOURCE_GEOMETRY_CLEAR","sample_count":len(states),
        "minimum_clearances_m":minima,"robot_links":len(scene.objects),
        "cad_pose_source":"CURRENT_RGBD","unmeasured_body_yaw":"ALL_YAW_ENVELOPE",
        "pose_reserve_m":.00025,"continuous_collision_guarantee":False,
        "object_or_contact_truth_used":False,"physical_approach_still_monitored":True}
