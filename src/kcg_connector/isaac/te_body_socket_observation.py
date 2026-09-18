"""Observe the socket in current ordinary RGB-D, without reading scene poses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from te_runtime_paths import sam6d_runtime


GLOBAL_CAMERA_CONFIG = "src/kcg_connector/config/te_rgbd_camera_global_e50_v1.yaml"
HAND_CAMERA_CONFIG = "src/kcg_connector/config/d38999_keyed_v2_hand_camera_probe_v1.yaml"
SOCKET_TEMPLATES = "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/templates"
SOCKET_CAD_MM = "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj"


def _intrinsics(camera):
    width, height = map(int, camera["resolution_px"])
    focal = width * float(camera["focal_length_mm"]) / float(camera["horizontal_aperture_mm"])
    return np.array(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)))


def hand_camera_mount(repository, camera_name):
    """Return the existing candidate camera mount; this does not observe it."""
    path = Path(repository).resolve() / HAND_CAMERA_CONFIG
    rig = yaml.safe_load(path.read_text(encoding="utf-8"))["camera_rig"]
    return {
        "source_config": str(path),
        "parent_frame": rig["parent_frame"],
        "camera_name": camera_name,
        "prim_suffix": rig[camera_name]["prim_suffix"],
        "hand_from_camera_cv": rig[camera_name]["T_HC_cv"],
        "world_pose_rule": "world_from_hand_from_joint_fk @ hand_from_camera_cv",
        "intrinsics_3x3": _intrinsics(rig).tolist(),
        "resolution_px": list(rig["resolution_px"]),
        "focal_length_mm": rig["focal_length_mm"],
        "horizontal_aperture_mm": rig["horizontal_aperture_mm"],
        "clipping_range_m": list(rig["clipping_range_m"]),
        "camera_axes": "CV_X_RIGHT_Y_DOWN_Z_FORWARD",
        "mount_status": rig["mount_contract"],
        "current_socket_visibility_verified": False,
        "wrist_capture_executed": False,
        "wrist_refinement_executed": False,
    }


def wrist_camera_mount(repository):
    return hand_camera_mount(repository, "wrist")


def observe_socket_from_global_rgbd(
    repository, stage, world, rep, output_dir, *, sam6d_root=None, sam6d_python=None,
):
    """Capture a paused scene, run SAM-6D, then fit the lip and real slot pattern.

    The caller must already have stopped physics after a controlled hold.
    Only camera configuration, source CAD, RGB-D, and the image-derived seed
    enter estimation.  No arm motion or object/fixture transform read occurs.
    A global slot fit is saved separately from the still-unexecuted wrist view.
    """
    import cv2
    import trimesh
    from pxr import Gf, UsdGeom

    from kcg_connector.te_rgbd_pose_provider import estimate_receptacle_key_from_depth
    from te_foundationpose_handoff_runtime import (
        _author_camera, _camera_cv_pose_from_eye_target, _capture_rgbd,
        _close_rgbd_resources, _json_ready, _run_sam6d_frame,
    )

    if world.is_playing():
        raise RuntimeError("socket observation requires the caller's paused hold")
    repository = Path(repository).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    camera_path = repository / GLOBAL_CAMERA_CONFIG
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))["camera"]
    if camera["channels_exactly"] != ["rgb", "distance_to_image_plane"]:
        raise ValueError("socket observation requires ordinary RGB and optical depth only")
    sam_root, sam_python = sam6d_runtime(repository, root=sam6d_root, python=sam6d_python)
    templates, cad = repository / SOCKET_TEMPLATES, repository / SOCKET_CAD_MM
    intrinsics = _intrinsics(camera)
    world_from_camera = _camera_cv_pose_from_eye_target(camera["eye_world_m"], camera["target_world_m"])
    camera_json = output / "camera.json"
    camera_json.write_text(json.dumps({
        "cam_K": intrinsics.ravel().tolist(), "depth_scale": 1,
    }, indent=2) + "\n", encoding="utf-8")
    record = {
        "status": "GLOBAL_SOCKET_OBSERVATION_PENDING",
        "physics_time_s": float(world.current_time),
        "camera_config_path": str(camera_path), "camera": camera,
        "world_from_camera_cv": world_from_camera.tolist(),
        "intrinsics_3x3": intrinsics.tolist(),
        "sam6d_root": str(sam_root), "sam6d_python": str(sam_python),
        "templates": str(templates), "cad_mm": str(cad),
        "rgbd_directory": str(output / "global_rgbd"),
        "wrist_camera": wrist_camera_mount(repository),
        "online_object_or_contact_truth_used": False,
        "robot_motion_commanded": False,
        "wrist_refinement_executed": False,
        "accuracy_against_simulation_truth_evaluated": False,
    }
    resources = {}
    try:
        _author_camera(
            stage, camera["prim_path"], world_from_camera,
            resolution=tuple(camera["resolution_px"]),
            focal_length_mm=float(camera["focal_length_mm"]),
            horizontal_aperture_mm=float(camera["horizontal_aperture_mm"]),
            clipping_range_m=tuple(camera["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom,
        )
        world.render()
        record["capture"] = _capture_rgbd(
            rep=rep, resources=resources, camera_path=camera["prim_path"],
            resolution=tuple(camera["resolution_px"]), output_dir=output / "global_rgbd",
            warmup_frames=int(camera["warmup_frames"]), rt_subframes=4,
        )
        seed = _run_sam6d_frame(
            repository=repository, sam6d_root=sam_root, sam6d_python=sam_python,
            templates=templates, cad_mm=cad,
            rgb=output / "global_rgbd/rgb.png", depth_m=output / "global_rgbd/depth_m.npy",
            depth_mm=output / "global_rgbd/depth_mm.png", camera_json=camera_json,
            output_dir=output / "perception",
        )
        coarse_pose = world_from_camera @ np.asarray(seed["camera_from_object"])
        record["coarse_estimate"] = {**seed, "mask": str(seed["mask"])}
        record["coarse_world_from_receptacle_row_major"] = coarse_pose.ravel().tolist()
        mask = cv2.imread(str(seed["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError("current-image socket segmentation mask is unavailable")
        refined = estimate_receptacle_key_from_depth(
            np.load(output / "global_rgbd/depth_m.npy"), mask > 0, intrinsics,
            world_from_camera, seed["camera_from_object"],
        )
        observed = bool(refined["key_direction_measured"])
        record.update({
            "status": "GLOBAL_SLOT_PATTERN_MEASURED" if observed else "GLOBAL_SLOT_PATTERN_UNRESOLVED",
            "global_slot_measurement": refined,
            "world_from_receptacle_row_major": refined.get("world_from_receptacle_row_major"),
            "global_slot_refinement_succeeded": observed,
        })
        mesh = trimesh.load(cad, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
        pose = np.asarray(refined["world_from_receptacle_row_major"]).reshape(4, 4) if observed else coarse_pose
        record["receptacle_obstacle"] = {
            "mesh": str(cad), "mesh_scale_to_m": 0.001,
            "local_bounds_m": (np.asarray(mesh.bounds) * 0.001).tolist(),
            "world_from_receptacle_row_major": pose.ravel().tolist(),
            "pose_source": "CURRENT_GLOBAL_LIP_AND_SLOT_FIT" if observed else "CURRENT_GLOBAL_SAM6D_COARSE_ONLY",
            "conservative_cylinder": {
                "radius_m": float(np.max(np.linalg.norm(vertices[:, :2], axis=1))),
                "z_min_m": float(np.min(vertices[:, 2])), "z_max_m": float(np.max(vertices[:, 2])),
            },
            "contact_alignment_verified": False,
        }
    except Exception as error:
        record.update({"status": "GLOBAL_SOCKET_OBSERVATION_FAILED", "error": str(error)})
        raise
    finally:
        _close_rgbd_resources(resources)
        (output / "camera_and_estimate.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return _json_ready(record)


def observe_body_after_transport(
    repository, stage, world, rep, world_from_hand, world_from_body_seed, output_dir,
):
    """Reobserve the held body and key from ordinary RGB-D at a paused hold.

    The supplied body pose only aims the camera and initializes the image ROI.
    A new hand/body relation is returned only when the current depth image
    resolves the real key; neither scene transforms nor contact truth are read.
    """
    from pxr import Gf, UsdGeom

    from kcg_connector.te_rgbd_pose_provider import estimate_held_plug_key_from_depth
    from te_foundationpose_handoff_runtime import (
        _author_camera, _camera_cv_pose_from_eye_target, _capture_rgbd,
        _close_rgbd_resources, _json_ready,
    )

    if world.is_playing():
        raise RuntimeError("body reobservation requires the caller's paused hold")
    hand = np.asarray(world_from_hand, dtype=np.float64).reshape(4, 4)
    seed = np.asarray(world_from_body_seed, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(hand).all() or not np.isfinite(seed).all():
        raise ValueError("encoder hand pose and visual body seed must be finite")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    local_eye = np.array([0.070, 0.160, 0.031], dtype=np.float64)
    eye = seed[:3, 3] + seed[:3, :3] @ local_eye
    camera_pose = _camera_cv_pose_from_eye_target(eye, seed[:3, 3])
    camera = {
        "prim_path": "/World/BodyAfterTransportKeyCamera",
        "resolution_px": [1280, 720], "focal_length_mm": 24.0,
        "horizontal_aperture_mm": 20.955, "clipping_range_m": [0.02, 10.0],
        "eye_body_seed_m": local_eye.tolist(), "eye_world_m": eye.tolist(),
        "target_world_m": seed[:3, 3].tolist(),
        "channels_exactly": ["rgb", "distance_to_image_plane"],
    }
    intrinsics = _intrinsics(camera)
    record = {
        "status": "BODY_REOBSERVATION_PENDING",
        "physics_time_s": float(world.current_time), "camera": camera,
        "intrinsics_3x3": intrinsics.tolist(),
        "world_from_camera_cv": camera_pose.tolist(),
        "world_from_hand_encoder": hand.tolist(),
        "world_from_body_roi_seed": seed.tolist(),
        "seed_role": "CAMERA_AIM_AND_RGBD_ROI_ONLY",
        "key_measurement": {"key_direction_measured": False},
        "hand_from_body_visual_memory": None,
        "online_object_or_contact_truth_used": False,
        "robot_motion_commanded": False,
        "accuracy_against_simulation_truth_evaluated": False,
    }
    resources = {}
    try:
        _author_camera(
            stage, camera["prim_path"], camera_pose,
            resolution=tuple(camera["resolution_px"]),
            focal_length_mm=camera["focal_length_mm"],
            horizontal_aperture_mm=camera["horizontal_aperture_mm"],
            clipping_range_m=tuple(camera["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom,
        )
        world.render()
        record["capture"] = _capture_rgbd(
            rep=rep, resources=resources, camera_path=camera["prim_path"],
            resolution=tuple(camera["resolution_px"]), output_dir=output / "rgbd",
            warmup_frames=3, rt_subframes=4,
        )
        measurement = estimate_held_plug_key_from_depth(
            np.load(output / "rgbd/depth_m.npy"), intrinsics, camera_pose, seed,
        )
        record["key_measurement"] = measurement
        if measurement["key_direction_measured"]:
            observed = np.asarray(measurement["world_from_plug_row_major"]).reshape(4, 4)
            record["hand_from_body_visual_memory"] = (np.linalg.inv(hand) @ observed).tolist()
            record["status"] = "CURRENT_BODY_AND_KEY_REOBSERVED"
        else:
            record["status"] = "CURRENT_BODY_KEY_UNRESOLVED_NO_NEW_MEMORY"
    except Exception as error:
        record.update(status="BODY_REOBSERVATION_FAILED", error=str(error))
        raise
    finally:
        _close_rgbd_resources(resources)
        (output / "camera_and_estimate.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return _json_ready(record)


def observe_released_plug_from_rgbd(
    repository, stage, world, rep, world_from_hand_encoder, output,
):
    """Observe the released plug's position and axis in one current RGB-D frame.

    The fixed palm mount and joint FK locate the camera. The plug estimate comes
    from its visible rear face, without a previous hand/body relation or keyed yaw.
    Failure is recorded and returned; this observation commands no robot motion
    and does not decide whether the socket physically supports the plug.
    """
    from pxr import Gf, UsdGeom

    from te_foundationpose_handoff_runtime import (
        _author_camera, _camera_cv_pose_from_eye_target, _capture_rgbd,
        _close_rgbd_resources, _json_ready, _run_geometry_frame,
    )

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resources = {}
    record = {
        "status": "RELEASED_PLUG_OBSERVATION_PENDING",
        "rgbd_directory": str(output / "rgbd"),
        "physics_time_s": float(world.current_time),
        "position_and_axis_measured": False,
        "world_from_plug_five_dof": None,
        "metrics": None,
        "axial_yaw_measured": False,
        "yaw_status": "UNOBSERVED_ARBITRARY_TRANSVERSE_BASIS_IN_FIVE_DOF_POSE",
        "previous_hand_body_relation_used": False,
        "online_object_or_contact_truth_used": False,
        "robot_motion_commanded": False,
        "physical_socket_support_evaluated": False,
        "accuracy_against_simulation_truth_evaluated": False,
        "observation_only": True,
    }
    try:
        if world.is_playing():
            raise RuntimeError("released-plug observation requires the caller's paused hold")
        hand = np.asarray(world_from_hand_encoder, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(hand).all():
            raise ValueError("the current encoder hand pose is nonfinite")
        mount = hand_camera_mount(repository, "palm")
        camera_pose = hand @ np.asarray(mount["hand_from_camera_cv"])
        camera = {
            "prim_path": "/World/ReleasedPlugPalmCaptureCamera",
            "resolution_px": mount["resolution_px"], "focal_length_mm": mount["focal_length_mm"],
            "horizontal_aperture_mm": mount["horizontal_aperture_mm"],
            "clipping_range_m": mount["clipping_range_m"],
            "mount": mount,
            "channels_exactly": ["rgb", "distance_to_image_plane"],
        }
        intrinsics = _intrinsics(camera)
        assets = repository / "artifacts/kcg_connector/vision/sam6d_segmentation_run19_observation_v1"
        sam_root, sam_python = sam6d_runtime(repository)
        camera_json = output / "camera.json"
        camera_json.write_text(json.dumps({
            "cam_K": intrinsics.ravel().tolist(), "depth_scale": 1,
        }, indent=2) + "\n", encoding="utf-8")
        record.update({
            "camera": camera, "intrinsics_3x3": intrinsics.tolist(),
            "world_from_camera_cv": camera_pose.tolist(),
            "world_from_hand_encoder": hand.tolist(),
            "sam6d_root": str(sam_root), "sam6d_python": str(sam_python),
            "templates": str(assets / "templates"),
            "cad_mm": str(assets / "D38999_26FJ35PN_VISUAL.obj"),
        })
        _author_camera(
            stage, camera["prim_path"], camera_pose,
            resolution=tuple(camera["resolution_px"]),
            focal_length_mm=camera["focal_length_mm"],
            horizontal_aperture_mm=camera["horizontal_aperture_mm"],
            clipping_range_m=tuple(camera["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom,
        )
        world.render()
        record["capture"] = _capture_rgbd(
            rep=rep, resources=resources, camera_path=camera["prim_path"],
            resolution=tuple(camera["resolution_px"]), output_dir=output / "rgbd",
            warmup_frames=3, rt_subframes=4,
        )
        record["capture_physics_time_s"] = float(world.current_time)
        wrist = wrist_camera_mount(repository)
        wrist_pose = hand @ np.asarray(wrist["hand_from_camera_cv"])
        wrist_path = "/World/ReleasedPlugWristCaptureCamera"
        _author_camera(stage, wrist_path, wrist_pose,
            resolution=tuple(wrist["resolution_px"]),
            focal_length_mm=wrist["focal_length_mm"],
            horizontal_aperture_mm=wrist["horizontal_aperture_mm"],
            clipping_range_m=tuple(wrist["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom)
        world.render()
        record["wrist_clearance_image"] = {
            "mount": wrist, "world_from_camera_cv": wrist_pose.tolist(),
            "capture": _capture_rgbd(rep=rep, resources=resources, camera_path=wrist_path,
                resolution=tuple(wrist["resolution_px"]), output_dir=output / "wrist_rgbd",
                warmup_frames=3, rt_subframes=4),
            "used_for_pose_estimation": False,
        }
        measured = _run_geometry_frame(
            repository=repository, sam6d_root=sam_root, sam6d_python=sam_python,
            templates=assets / "templates", cad_mm=assets / "D38999_26FJ35PN_VISUAL.obj",
            rgb=output / "rgbd/rgb.png", depth_m=output / "rgbd/depth_m.npy",
            depth_mm=output / "rgbd/depth_mm.png", camera_json=camera_json,
            camera_matrix=intrinsics, output_dir=output / "perception",
            geometry_method="float_depth_circle",
        )
        pose = camera_pose @ np.asarray(measured["camera_from_object"], dtype=np.float64)
        record.update({
            "status": "RELEASED_PLUG_POSITION_AND_AXIS_OBSERVED",
            "position_and_axis_measured": True,
            "world_from_plug_five_dof": pose.tolist(),
            "metrics": measured["geometry"]["metrics"],
            "measurement": measured,
        })
    except Exception as error:
        record.update(status="RELEASED_PLUG_OBSERVATION_FAILED", reason=str(error))
    finally:
        _close_rgbd_resources(resources)
        (output / "camera_and_estimate.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return _json_ready(record)


def observe_tracked_plug_from_rgbd(repository,stage,world,rep,hand,output,context):
    """Fresh depth-circle measurement, reusing a visual identity ROI and camera.

    The previous image only locates a search region. Position and axis are fit
    again from this frame; no simulator pose or contact record is read.
    """
    import cv2
    from time import perf_counter
    from pxr import Gf,UsdGeom
    from te_foundationpose_handoff_runtime import _author_camera,_capture_rgbd,_json_ready
    from te_plug_five_dof_geometry import estimate_plug_rear_circle_from_float_depth
    repository,output=Path(repository).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    if world.is_playing():raise RuntimeError('tracked observation requires a paused physical state')
    started=perf_counter();before=float(world.current_time)
    seed=context['last_observation'];hand=np.asarray(hand).reshape(4,4)
    mount=hand_camera_mount(repository,'palm');camera_pose=hand@np.asarray(mount['hand_from_camera_cv'])
    camera={'resolution_px':mount['resolution_px'],'focal_length_mm':mount['focal_length_mm'],
        'horizontal_aperture_mm':mount['horizontal_aperture_mm']}
    K=_intrinsics(camera);path='/World/TrackedPlugPalmCaptureCamera'
    resources=context.setdefault('resources',{})
    _author_camera(stage,path,camera_pose,resolution=tuple(mount['resolution_px']),
        focal_length_mm=mount['focal_length_mm'],horizontal_aperture_mm=mount['horizontal_aperture_mm'],
        clipping_range_m=tuple(mount['clipping_range_m']),Gf=Gf,UsdGeom=UsdGeom)
    world.render()
    capture=_capture_rgbd(rep=rep,resources=resources,camera_path=path,resolution=tuple(mount['resolution_px']),
        output_dir=output/'rgbd',warmup_frames=3,rt_subframes=4)
    depth=np.load(output/'rgbd/depth_m.npy')
    mask_path=context.get('mask_path') or seed['measurement']['sam']['mask']
    old_mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if old_mask is None or old_mask.shape!=depth.shape:raise RuntimeError('visual tracking seed mask is unavailable')
    old_pose=np.asarray(seed['world_from_plug_five_dof']);old_camera=np.asarray(seed['world_from_camera_cv'])
    axial=float(seed['metrics']['visible_face_to_object_origin_m'])
    face=np.r_[old_pose[:3,3]-axial*old_pose[:3,2],1.]
    old=np.linalg.inv(old_camera)@face;new=np.linalg.inv(camera_pose)@face
    if min(old[2],new[2])<=0:raise RuntimeError('tracked rear face is outside the calibrated camera')
    old_uv=(K@old[:3])[:2]/old[2];new_uv=(K@new[:3])[:2]/new[2];scale=old[2]/new[2]
    if not .9<=scale<=1.1:raise RuntimeError('visual search motion exceeds the short tracking range')
    affine=np.array([[scale,0,new_uv[0]-scale*old_uv[0]],[0,scale,new_uv[1]-scale*old_uv[1]]])
    mask=cv2.warpAffine(old_mask,affine,(depth.shape[1],depth.shape[0]),flags=cv2.INTER_NEAREST)
    mask=cv2.dilate(mask,np.ones((25,25),np.uint8))>0
    mask &= np.isfinite(depth)&(depth>0)&(np.abs(depth-new[2])<.006)
    cad=repository/'artifacts/kcg_connector/vision/sam6d_segmentation_run19_observation_v1/D38999_26FJ35PN_VISUAL.obj'
    geometry=estimate_plug_rear_circle_from_float_depth(depth_m=depth,mask=mask,intrinsics=K,
        mesh_path=cad,pixel_center_offset_px=.5,plane_iterations=128)
    geometry['metrics']['mask_source']='PREVIOUS_VISUAL_IDENTITY_ROI_WITH_CURRENT_DEPTH_VALIDATION'
    geometry['metrics']['previous_object_pose_used_only_for_search_region']=True
    if geometry['metrics']['plane_ransac_sampled_inlier_fraction']<.45:
        raise RuntimeError('current tracked face plane has insufficient support')
    pose=camera_pose@np.asarray(geometry['camera_from_object'])
    if np.linalg.norm(pose[:3,3]-old_pose[:3,3])>.002:
        raise RuntimeError('tracked position change exceeds the bounded observation interval')
    mask_file=output/'tracking_seed_mask.png';cv2.imwrite(str(mask_file),(mask.astype(np.uint8)*255))
    if float(world.current_time)!=before:raise RuntimeError('tracking camera advanced physical time')
    record=_json_ready({'status':'CURRENT_DEPTH_TRACKED_PLUG_POSE','position_and_axis_measured':True,
        'rgbd_directory':str(output/'rgbd'),
        'physics_time_s':before,'capture_physics_time_s':before,'capture':capture,
        'world_from_hand_encoder':hand.tolist(),'world_from_camera_cv':camera_pose.tolist(),
        'world_from_plug_five_dof':pose.tolist(),'metrics':geometry['metrics'],
        'axial_yaw_measured':False,'online_object_or_contact_truth_used':False,
        'previous_hand_body_relation_used':False,'measurement':{'geometry':geometry},
        'tracking_wall_s':perf_counter()-started,'tracking_seed_mask':str(mask_file)})
    context.update(last_observation=record,mask_path=str(mask_file))
    (output/'camera_and_estimate.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def observe_current_plug_from_rgbd(repository,stage,world,rep,hand,output,runtime):
    """Reuse current-episode visual identity; reacquire when tracking is invalid."""
    session=runtime.get('four_camera_perception_session')
    if session is not None:
        return session.available_observation(output)
    context=runtime.setdefault('plug_visual_tracking_context',{})
    if context.get('last_observation') is not None:
        try:
            return observe_tracked_plug_from_rgbd(repository,stage,world,rep,hand,output,context)
        except (RuntimeError,ValueError) as error:
            fallback=Path(output).with_name(Path(output).name+'_reacquire')
            observed=observe_released_plug_from_rgbd(repository,stage,world,rep,hand,fallback)
            observed['full_reacquisition_reason']=str(error)
    else:
        observed=observe_released_plug_from_rgbd(repository,stage,world,rep,hand,output)
    if observed.get('position_and_axis_measured'):
        context.update(last_observation=observed)
        context.pop('mask_path',None)
    return observed
