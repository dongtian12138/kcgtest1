#!/usr/bin/env python3
"""Static original-hand wrench check against an explicitly mounted Nut fixture.

Initialization uses a completed run's pose and finite drive targets. The Nut is
a declared laboratory fixture, not an assembly success or hidden support.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--assembly-config",type=Path,
                    help="Explicit local physical-model configuration; defaults to the sealed source configuration.")
parser.add_argument("--mounted-grasp-recipe",type=Path,
                    help="Explicit fixed-Nut laboratory grasp/torque recipe, not an assembly run.")
parser.add_argument("--velocity-iterations", type=int, choices=(0,1,4), default=1)
parser.add_argument("--contact-convergence-check", action="store_true")
parser.add_argument("--diagnostic-center-socket-before-start", action="store_true")
parser.add_argument("--main-read-sequence", action="store_true",
                    help="Read the main runner's native gravity/projected/other-articulation signals; targets stay fixed.")
parser.add_argument("--wrist-reference-loads",action="store_true",
                    help="Keep the source hand open and apply bounded reference loads to its wrist sensor subtree; no grasp or assembly.")
parser.add_argument("--position-iterations",type=int,default=32)
parser.add_argument("--closing-drive-cap-nm",type=float,default=1.)
parser.add_argument("--frozen-connector-model",type=Path)
parser.add_argument("--cpu-wrist-reference",type=Path)
parser.add_argument('--probe-in-turn-feedback',action='store_true',
                    help='Use current RGBD and encoder feedback between bounded nut-turn intervals; no object truth.')
parser.add_argument('--probe-segmented-observation-only',action='store_true',
                    help='Declared ablation: acquire the same interval images but retain the prepared rigid grip relation.')
parser.add_argument("--probe-open-grasp-recipe",type=Path,
                    help="Declared pre-physics open-hand joint state for a fresh grasp on the free connector")
parser.add_argument("--fabric-gpu-interop",action="store_true",
                    help="Explicitly request GPU-resident Fabric output at process startup for the bounded rendering consistency check.")
parser.add_argument("--gpu-host-readback",action="store_true",
                    help="Retain GPU dynamics but allow host result readback before initialization; a bounded physics-output/data-path diagnostic.")
parser.add_argument("--independent-robot-rigid-frames",action="store_true",
                    help="Before reset, give nested robot rigid bodies independent transform stacks while preserving all source world poses and joint properties.")
parser.add_argument("--standard-render-steps",action="store_true",
                    help="Bounded output-path check: use the SDK normal render-step path with rendering_dt equal to physics_dt and verify exact step count.")
parser.add_argument("--arm-damping", type=float,
                    help="Default to the source sample's actual damping, including its earlier lift transition.")
parser.add_argument("--free-plug-in-socket", action="store_true",
                    help="Reuse the source assembly scene and its free Body-Nut joint, with no diagnostic mount.")
parser.add_argument("--paused-world-render", action="store_true",
                    help="At one second call the still-existing observation World.render once while paused.")
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0,1), default=1)
parser.add_argument("--hand-friction-effort-from-urdf", action="store_true")
parser.add_argument("--nut-fingertip-sdf", action="store_true")
parser.add_argument("--nut-fingertip-sdf-link",nargs='+',choices=('f1Link3','f2Link2','f3Link3'))
parser.add_argument("--physics-device", choices=("cuda:0","cpu"), default="cuda:0")
parser.add_argument("--solver-type", choices=("TGS","PGS"), default="TGS")
parser.add_argument("--robot-state-before-reset", action="store_true",
                    help="Author the complete source robot joint state and finite drives before the SDK's two warmup ticks.")
parser.add_argument("--audit-source-key-sdf", action="store_true",
                    help="After the static measurement, read native core SDF at original key-surface samples; never a controller input.")
parser.add_argument("--replay-source-drive-targets", action="store_true",
                    help="In the two-second diagnostic replay the following sealed robot commands, then hold the last one; no online object/contact inputs.")
parser.add_argument("--physics-hz", type=int, choices=(240,480,960), default=240,
                    help="Bounded time-resolution check for the two-second fixed-target diagnostic.")
parser.add_argument("--source-step", type=int)
parser.add_argument("--sensor-stage", default="nut_regrasp_after_index")
parser.add_argument("--source-rotation-stage", default="nut_rotation_after_index")
parser.add_argument("--source-grip-stage", default="nut_regrasp_after_index")
parser.add_argument("--probe-reobserve-axis", action="store_true")
parser.add_argument("--probe-additional-turn-deg", type=float)
parser.add_argument("--probe-speed-deg-s", type=float, default=2.)
parser.add_argument("--probe-arm-speed-limit-rad-s", type=float)
parser.add_argument("--probe-accel-deg-s2", type=float)
parser.add_argument("--probe-torsional-stop-nm", type=float)
parser.add_argument("--probe-lateral-stop-n", type=float)
parser.add_argument("--probe-grasp-axis-shift-m", type=float, default=0.)
parser.add_argument("--probe-grip-effort-scale", type=float, default=1.)
parser.add_argument("--closing-reaction-stop-nm", type=float, default=.9)
parser.add_argument("--probe-visual-alignment-feedback", action="store_true")
parser.add_argument("--probe-visual-position-feedback", action="store_true")
parser.add_argument("--probe-hold-grip-through-preparation", action="store_true")
parser.add_argument("--probe-grip-hold-stiffness-scale", type=float)
parser.add_argument("--probe-guided-fit-continuation", action="store_true")
parser.add_argument("--probe-geometric-fit-feedback", action="store_true")
parser.add_argument("--probe-pin-fit-budget-m", type=float, default=.00004)
parser.add_argument("--probe-planned-contact-force-stop-n", type=float)
parser.add_argument("--probe-grip-effort-weights", type=float, nargs=3)
parser.add_argument("--probe-grip-lateral-balance", action="store_true")
parser.add_argument("--probe-repeat-after-reindex", action="store_true")
parser.add_argument("--probe-unload-after-lateral-hold-stop", action="store_true")
parser.add_argument("--probe-second-grasp-axis-shift-m", type=float)
parser.add_argument("--probe-second-regrasp-effort-time-constant-s", type=float)
parser.add_argument("--probe-start-open", action="store_true")
parser.add_argument("--probe-regrasp-only", action="store_true")
parser.add_argument("--probe-hold-finger-targets", action="store_true")
parser.add_argument("--probe-planar-stiffness-n-m", type=float)
parser.add_argument("--probe-hold-planar-position", action="store_true")
parser.add_argument("--probe-freeze-planar-after-preparation", action="store_true")
parser.add_argument("--probe-force-consistent-planar-range",action='store_true')
parser.add_argument("--probe-guided-socket-pivot", action="store_true")
args = parser.parse_args()
if not 1 <= args.position_iterations <= 255:parser.error('position iterations out of range')
if not 0 < args.closing_drive_cap_nm <= 2.7:parser.error('closing motor cap outside the bounded simulation diagnostic range')
if args.contact_convergence_check and (args.velocity_iterations!=4 or args.physics_device!='cpu'
        or (args.frozen_connector_model is None and not args.wrist_reference_loads)
        or args.solver_type!='TGS' or not args.external_forces_every_iteration):
    parser.error('the explicit convergence check changes only CPU TGS velocity iterations from 1 to 4')
if args.velocity_iterations==4 and not args.contact_convergence_check:
    parser.error('four velocity iterations require the explicit unvalidated convergence diagnostic')
if args.probe_visual_position_feedback and not args.probe_visual_alignment_feedback:
    parser.error('actual centre feedback requires the current RGBD alignment-feedback mode')
if args.probe_geometric_fit_feedback and not args.probe_visual_position_feedback:
    parser.error('guided feature admission requires actual centre/axis visual feedback')
if args.probe_in_turn_feedback and (args.physics_device!='cpu' or not args.probe_start_open
        or not args.probe_geometric_fit_feedback or not args.probe_hold_grip_through_preparation
        or args.cpu_wrist_reference is None or args.probe_additional_turn_deg is None):
    parser.error('observed-turn pilot requires the CPU wrist reference, real open/regrasp, current pose fit and fixed finite grip')
if args.probe_segmented_observation_only and not args.probe_in_turn_feedback:
    parser.error('segmentation-only ablation requires the declared observed-turn mode')
if not .00004 <= args.probe_pin_fit_budget_m <= .000065:
    parser.error('nominal guided pin admission budget must remain between40 and65micrometres')
if args.probe_grip_hold_stiffness_scale is not None and (not args.probe_hold_grip_through_preparation
        or not 1. < args.probe_grip_hold_stiffness_scale <= 4.):
    parser.error('holding impedance scale requires fixed finite preload and is bounded to four')
if args.probe_guided_fit_continuation and (args.probe_start_open or args.probe_grasp_axis_shift_m
        or not args.probe_reobserve_axis or not args.probe_hold_grip_through_preparation
        or args.probe_grip_hold_stiffness_scale is not None):
    parser.error('guided-fit continuation requires a declared loaded source, current RGBD and unchanged-gain fixed preload')
if args.probe_planned_contact_force_stop_n is not None and (not args.probe_visual_position_feedback
        or not 3.0400615 <= args.probe_planned_contact_force_stop_n <= 5.):
    parser.error('the explicit centred-contact observation window is bounded from the original reference through 5 N')
if args.diagnostic_center_socket_before_start and (not args.free_plug_in_socket or args.probe_additional_turn_deg is not None):
    parser.error('diagnostic socket centering is only a declared pre-physics fixed-command comparison, not an assembly controller')
if args.wrist_reference_loads and (not args.mounted_grasp_recipe or args.physics_device!='cpu'):
    parser.error('wrist reference loading requires the explicit open-hand CPU mounted recipe')
if args.probe_open_grasp_recipe is not None and (not args.probe_start_open or not args.free_plug_in_socket):
    parser.error('open-grasp recipe requires a declared free-connector open-hand start')
if args.solver_type == "PGS" and args.external_forces_every_iteration:
    parser.error("PGS does not support per-iteration external forces; pass --external-forces-every-iteration 0")
if args.gpu_host_readback and args.physics_device=="cpu":
    parser.error("host-readback diagnostic must retain the GPU dynamics scene")
if args.standard_render_steps and not args.mounted_grasp_recipe:
    parser.error("standard render-step check is confined to the mounted Nut diagnostic")
if args.probe_regrasp_only:
    if args.probe_additional_turn_deg is not None:
        parser.error("regrasp-only mode has no requested rotation")
    args.probe_additional_turn_deg = 0.
args.run = args.run.resolve(); args.output = args.output.resolve()
if any(Path(value).name != value for value in (args.sensor_stage,args.source_rotation_stage,args.source_grip_stage)):
    parser.error("source stages must be local directory names")
if args.probe_additional_turn_deg is not None and (not args.free_plug_in_socket or args.source_step is None):
    parser.error("a turn probe requires the free plug scene and an explicit completed source step")
if args.probe_additional_turn_deg is not None and args.physics_hz != 240 and args.frozen_connector_model is None:
    parser.error("the time-resolution check currently applies only to the fixed-target diagnostic")
if args.frozen_connector_model is not None and (not args.free_plug_in_socket or args.physics_device!='cpu'
        or args.physics_hz!=960 or args.position_iterations!=128 or args.cpu_wrist_reference is None):
    parser.error('the delivered connector requires its validated CPU/960Hz/128 configuration and a CPU wrist reference')
if args.audit_source_key_sdf and (not args.free_plug_in_socket or args.physics_device == "cpu" or args.probe_additional_turn_deg is not None):
    parser.error("the source-key SDF audit is a post-measurement read on the static GPU/free-connector diagnostic")
if args.replay_source_drive_targets and (args.probe_additional_turn_deg is not None or args.source_step is None):
    parser.error("the bounded source-command replay requires a declared source step and no controller probe")
if args.mounted_grasp_recipe and (args.free_plug_in_socket or args.probe_additional_turn_deg is not None
                                  or not args.robot_state_before_reset):
    parser.error("the mounted grasp recipe uses the explicit Nut fixture and pre-reset robot initialization")
args.output.mkdir(parents=True,exist_ok=False)
(args.output/'launch.json').write_text(json.dumps({
    'argv':[sys.executable,*sys.argv],
    'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2)+'\n')
(args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
sys.path.insert(0,str(Path(__file__).with_name("carts_v2")))
from isaacsim import SimulationApp
app = SimulationApp({"headless":args.probe_additional_turn_deg is None,"multi_gpu":False,
                     "active_gpu":0,"physics_gpu":0,"fast_shutdown":True,
                     "extra_args":((["--/rtx/hydra/supportMultiTickRate=false"]
                                    if args.probe_additional_turn_deg is not None else [])
                                   +(["--/physics/fabricUseGPUInterop=true"] if args.fabric_gpu_interop else []))})
failed=False
try:
    import carb
    import numpy as np
    import omni.usd
    from scipy.spatial.transform import Rotation
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.experimental.utils.backend import use_backend
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    import controller

    repo=Path(__file__).resolve().parents[3]
    local_metadata=json.loads((args.run/"trace_metadata.json").read_text())
    base_run=(args.run if "motion_plan" in local_metadata
              else Path(local_metadata.get("base_run",local_metadata["source_run"])))
    metadata=(local_metadata if base_run==args.run else json.loads((base_run/"trace_metadata.json").read_text()))
    control_record=json.loads((args.run/"socket_transport"/args.source_rotation_stage/"nut_rotation_controller_result.json").read_text())
    sample_step=int(control_record["first_step"])-1 if args.source_step is None else args.source_step
    with gzip.open(args.run/"socket_transport"/args.sensor_stage/"joint_ft_samples.json.gz","rt") as f:
        source_sensor_records=json.load(f)
        sensor_sample=next(s for s in source_sensor_records if s["step"]==sample_step)
    replay_records=([s for s in source_sensor_records if sample_step<s["step"]<=sample_step+2*args.physics_hz]
                    if args.replay_source_drive_targets else [])
    del source_sensor_records
    if args.replay_source_drive_targets and (not replay_records or [s["step"] for s in replay_records]!=list(range(sample_step+1,replay_records[-1]["step"]+1))):
        raise ValueError("the sealed source command interval is empty or discontinuous")
    trace=args.run/"truth_samples.jsonl"
    with trace.open() as f:
        f.seek(max(0,trace.stat().st_size-64000000));f.readline()
        physical_sample=next((json.loads(line) for line in f if f'"step":{sample_step},' in line[:100]),None)
    if physical_sample is None:
        # An earlier completed open-hand boundary need not be in the tail.
        # Scan prefixes only, avoiding decoding unrelated large contact arrays.
        with trace.open() as f:
            physical_sample=next((json.loads(line) for line in f if f'"step":{sample_step},' in line[:100]),None)
    if physical_sample is None:
        raise ValueError(f"the declared source step {sample_step} is absent from the sealed trace")
    lab_recipe=None;lab_fixture_pose=None
    if args.probe_open_grasp_recipe is not None:
        import copy
        from kcg_connector.robot_model import expand_active_hand_positions
        opening=json.loads(args.probe_open_grasp_recipe.read_text())
        hand=np.asarray(opening['open_hand_positions_rad'],float)
        sensor_sample=copy.deepcopy(sensor_sample);physical_sample=copy.deepcopy(physical_sample)
        sensor_sample['active_positions_rad'][7:]=hand.tolist()
        sensor_sample['active_targets_rad'][7:]=hand.tolist()
        for name,value in expand_active_hand_positions(hand).items():
            physical_sample['arm_control']['hand_joint_diagnostic']['joints'][name]['position_rad']=float(value)
        (args.output/'declared_open_hand_start.json').write_text(json.dumps({
            'recipe':str(args.probe_open_grasp_recipe),'hand_positions_rad':hand.tolist(),
            'scope':'PRE_PHYSICS_INITIAL_CONDITION_FOR_FRESH_GRASP_NO_POST_START_OBJECT_POSE_WRITE'},indent=2)+'\n')
    if args.mounted_grasp_recipe:
        import copy
        from kcg_connector.robot_model import expand_active_hand_positions
        lab_recipe=json.loads(args.mounted_grasp_recipe.read_text())
        if args.standard_render_steps:lab_recipe["standard_render_steps"]=True
        H=np.eye(4);H[:3,:3]=np.asarray(sensor_sample["handbase_rotation_world_row_major"]).reshape(3,3)
        H[:3,3]=sensor_sample["handbase_position_world_m"]
        lab_fixture_pose=H@np.linalg.inv(np.asarray(lab_recipe["canonical_body_from_hand_for_nut_grasp"]))
        sensor_sample=copy.deepcopy(sensor_sample);physical_sample=copy.deepcopy(physical_sample)
        hand=np.asarray(lab_recipe["open_hand_positions_rad"],float)
        sensor_sample["active_positions_rad"][7:]=hand.tolist()
        sensor_sample["active_targets_rad"][7:]=hand.tolist()
        for name,value in expand_active_hand_positions(hand).items():
            physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"][name]["position_rad"]=float(value)
    dt=1/float(args.physics_hz)
    half_second=round(.5/dt)
    SimulationManager.set_physics_sim_device(args.physics_device)
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=dt if args.standard_render_steps else 1/60,
                backend="numpy",device=args.physics_device,
                sim_params={"use_gpu_pipeline":args.physics_device!="cpu"})
    if args.gpu_host_readback:
        # SimulationManager.set_device('cuda') has already enabled GPU
        # dynamics. Change only suppression of host output, before reset;
        # do not call set_device('cpu'), which would disable GPU dynamics.
        carb.settings.get_settings().set_bool("/physics/suppressReadback",False)
    stage=omni.usd.get_context().get_stage()
    scene=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    if args.physics_device=='cpu':
        scene.CreateEnableGPUDynamicsAttr(False);scene.CreateBroadphaseTypeAttr('MBP')
    scene.CreateSolverTypeAttr(args.solver_type);scene.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))
    scene.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    scene.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    robot_root=UsdGeom.Xform.Define(stage,"/World/HandArm")
    robot_root.GetPrim().GetReferences().AddReference(metadata["robot_asset"])
    if args.independent_robot_rigid_frames:
        from te_hand_rigid_transform_scene import author_independent_robot_rigid_frames
        frame_report=author_independent_robot_rigid_frames(stage,"/World/HandArm",
            physics_started=bool(world.is_playing() or world.current_time>0.))
        (args.output/"robot_rigid_frame_authoring.json").write_text(json.dumps(frame_report,indent=2)+"\n")
    hand_friction_report = None
    if args.hand_friction_effort_from_urdf:
        from te_hand_joint_friction import author_urdf_hand_friction_efforts
        hand_friction_report = author_urdf_hand_friction_efforts(repo,stage,"/World/HandArm")
        (args.output/"hand_joint_friction_authoring.json").write_text(json.dumps(hand_friction_report,indent=2)+"\n")
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            a=PhysxSchema.PhysxArticulationAPI(prim);a.CreateSolverPositionIterationCountAttr(args.position_iterations)
            a.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            if args.wrist_reference_loads:a.CreateSleepThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            body_api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body_api.CreateSolverPositionIterationCountAttr(args.position_iterations)
            body_api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            if args.wrist_reference_loads:body_api.CreateSleepThresholdAttr(0.)
    material=UsdShade.Material.Define(stage,"/World/FixtureMaterial")
    m=UsdPhysics.MaterialAPI.Apply(material.GetPrim());m.CreateStaticFrictionAttr(.45)
    m.CreateDynamicFrictionAttr(.45);m.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("min")
    fingertip=stage.GetPrimAtPath("/World/HandArm/PhysicsMaterials/fingertip_pad")
    fm=UsdPhysics.MaterialAPI(fingertip);fm.CreateStaticFrictionAttr(.45);fm.CreateDynamicFrictionAttr(.45)
    PhysxSchema.PhysxMaterialAPI.Apply(fingertip).CreateFrictionCombineModeAttr("min")

    if args.free_plug_in_socket:
        import yaml
        from isaacsim.core.utils.stage import add_reference_to_stage
        from run_grasp_lift import prepare_dynamic_scene, _apply_contact_friction_perturbation
        from te_body_assembly_scene import prepare_body_assembly_scene
        from te_grounding_band_scene import install_grounding_band_contact_model
        launch=json.loads(base_run.with_suffix(".launch.json").read_text())["argv"]
        config=yaml.safe_load((repo/launch[launch.index("--config")+1]).read_text())
        assembly_path=(args.run/"local_assembly_control.yaml" if (args.run/"local_assembly_control.yaml").exists()
                       else Path(local_metadata["body_assembly_control_config"])
                       if "body_assembly_control_config" in local_metadata
                       else repo/launch[launch.index("--body-assembly-collision-config")+1])
        if args.assembly_config is not None:
            assembly_path=args.assembly_config.resolve()
        entry=dict(config["dynamic"]["object_scenes"][metadata["object_id"]])
        def saved_pose(i):
            T=np.eye(4);q=np.array(physical_sample["object_part_orientations_wxyz"][i])
            T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            T[:3,3]=physical_sample["object_part_positions_m"][i]
            return T
        body_T,nut_T=saved_pose(0),saved_pose(1)
        entry["frozen_settled_world_from_object_row_major"]=body_T.ravel().tolist()
        assembly=prepare_dynamic_scene(repo,stage,entry,add_reference_to_stage,
                                       metadata["robustness_perturbation"])
        nut_xform=UsdGeom.Xformable(stage.GetPrimAtPath(assembly["part_prim_paths"][1]))
        nut_xform.ClearXformOpOrder()
        nut_xform.AddTransformOp().Set(Gf.Matrix4d(*(np.linalg.inv(body_T)@nut_T).T.ravel().tolist()))
        prepared=prepare_body_assembly_scene(repo,stage,assembly,assembly_path)
        assembly=prepared["scene"]
        if args.diagnostic_center_socket_before_start:
            socket_prim=stage.GetPrimAtPath('/World/TEVisualHandoff/FixedReceptaclePose')
            socket_xform=UsdGeom.Xformable(socket_prim)
            before=np.asarray(socket_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.copy()
            axis=before[:3,2]/np.linalg.norm(before[:3,2])
            shift=(np.eye(3)-np.outer(axis,axis))@(body_T[:3,3]-before[:3,3])
            if np.linalg.norm(shift)>.001:
                raise ValueError('diagnostic fixture-centering offset exceeds 1 mm')
            after=before.copy();after[:3,3]+=shift
            parent=np.asarray(UsdGeom.Xformable(socket_prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            local=np.linalg.inv(parent)@after
            socket_xform.ClearXformOpOrder()
            socket_xform.AddTransformOp(opSuffix='DeclaredColdCentering').Set(Gf.Matrix4d(*local.T.ravel().tolist()))
            centering={'scope':'PRE_PHYSICS_FIXTURE_PLACEMENT_COUNTERFACTUAL_NOT_ROBOT_ALIGNMENT',
                'socket_position_before_world_m':before[:3,3].tolist(),
                'socket_position_after_world_m':after[:3,3].tolist(),
                'socket_shift_world_m':shift.tolist(),
                'robot_and_body_nut_initial_poses_changed':False,
                'geometry_materials_mass_inertia_changed':False,'post_start_pose_write':False}
            prepared['report']['diagnostic_initial_socket_centering']=centering
            prepared['report']['socket_initial_position_world_m']=after[:3,3].tolist()
            (args.output/'diagnostic_initial_socket_centering.json').write_text(json.dumps(centering,indent=2)+'\n')
        passive_joint=stage.GetPrimAtPath(prepared["report"]["passive_joint_solver"]["internal_joint_path"])
        source_joint_audit=physical_sample.get("object_internal_joint_audit")
        if source_joint_audit is not None:
            source_passive_angle=float(source_joint_audit["positions_rad"][0])
        elif args.frozen_connector_model is not None:
            # The delivered maximal joint has no articulation coordinate
            # reader. Its declared cold initial angle follows the recorded
            # native Body/Nut poses and original joint frames, modulo 2 pi.
            j=UsdPhysics.RevoluteJoint(passive_joint)
            def local_rotation(q):
                return Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
            relative=(local_rotation(j.GetLocalRot0Attr().Get()).T @ body_T[:3,:3].T
                      @ nut_T[:3,:3] @ local_rotation(j.GetLocalRot1Attr().Get()))
            q=Rotation.from_matrix(relative).as_quat()
            axis_index=("X","Y","Z").index(j.GetAxisAttr().Get())
            source_passive_angle=float(2*np.arctan2(q[axis_index],q[3]))
            (args.output/'passive_angle_from_saved_native_poses.json').write_text(json.dumps({
                'source_step':sample_step,'angle_rad_modulo_two_pi':source_passive_angle,
                'source':'DECLARED_COLD_NATIVE_BODY_NUT_POSES_AND_ORIGINAL_JOINT_FRAMES',
                'used_after_physics_start':False})+'\n')
        else:
            raise ValueError('source has no passive-joint coordinate or declared maximal-joint model')
        initial_joint_state=PhysxSchema.JointStateAPI.Apply(passive_joint,UsdPhysics.Tokens.angular)
        initial_joint_state.CreatePositionAttr(float(np.degrees(source_passive_angle)))
        authored_passive_angle = float(np.radians(initial_joint_state.GetPositionAttr().Get()))
        authored_phase_error = float(np.arctan2(np.sin(authored_passive_angle-source_passive_angle),
                                                np.cos(authored_passive_angle-source_passive_angle)))
        if abs(authored_phase_error) > 2e-6:
            raise RuntimeError("the authored passive-joint angle does not represent the declared source value")
        # This is a static local diagnostic. Reported TGS velocity is not a
        # reliable derivative of the saved pose and is deliberately not copied.
        initial_joint_state.CreateVelocityAttr(0.)
        _apply_contact_friction_perturbation(stage,assembly,Usd,UsdPhysics,UsdShade,PhysxSchema)
        band=yaml.safe_load(assembly_path.read_text())["grounding_band_contact_model"]
        body_path,fixture_path=assembly["part_prim_paths"]
        other=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
               and str(p.GetPath())!=body_path]
        prepared["report"]["grounding_band_contact_model"]=install_grounding_band_contact_model(
            stage,body_path,repo/band["geometry_manifest"],
            stiffness_n_m=band["native_per_contact_stiffness_n_m"],damping_ns_m=band["native_per_contact_damping_ns_m"],
            socket_collision_path=prepared["receptacle_collision_path"],
            non_socket_contact_paths=[*other,assembly["roots"]["table"],assembly["roots"]["fixture"]])
        mating=yaml.safe_load(assembly_path.read_text()).get("mating_contact_model")
        if mating and mating.get("enabled"):
            from te_mating_contact_scene import install_socket_contact_interior
            install_socket_contact_interior(repo,stage,prepared,mating["manifest"])
        if args.frozen_connector_model is not None:
            import importlib.util
            installer=args.frozen_connector_model.resolve().parent/'install_model.py'
            spec=importlib.util.spec_from_file_location('validated_connector_installer',installer)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            frozen_install=module.install_model(stage,model_path=args.frozen_connector_model,prepared=prepared)
            (args.output/'frozen_model_installation.json').write_text(json.dumps(frozen_install,indent=2)+'\n')
        (args.output/"assembly_scene.json").write_text(json.dumps(prepared["report"],indent=2)+"\n")
        fixture_sensor_path=body_path
        contact_filters=list(dict.fromkeys([body_path,fixture_path,prepared["receptacle_collision_path"],
            *prepared["contact_recording"]["additional_required_contact_filter_paths"]]))
    else:
        fixture_path="/World/DeclaredNutFixture"
        fixture=UsdGeom.Xform.Define(stage,fixture_path)
        position=physical_sample["object_part_positions_m"][1];quat=physical_sample["object_part_orientations_wxyz"][1]
        if lab_fixture_pose is not None:
            position=lab_fixture_pose[:3,3].tolist()
            q=Rotation.from_matrix(lab_fixture_pose[:3,:3]).as_quat();quat=q[[3,0,1,2]].tolist()
        fixture.AddTranslateOp().Set(Gf.Vec3d(*position));fixture.AddOrientOp().Set(Gf.Quatf(quat[0],Gf.Vec3f(*quat[1:])))
        UsdPhysics.RigidBodyAPI.Apply(fixture.GetPrim())
        mass_stage=Usd.Stage.Open(str(repo/"artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"))
        original_mass=mass_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/CouplingNut")
        UsdPhysics.MassAPI.Apply(fixture.GetPrim())
        for name in ("physics:mass","physics:centerOfMass","physics:diagonalInertia","physics:principalAxes"):
            a=original_mass.GetAttribute(name);fixture.GetPrim().CreateAttribute(name,a.GetTypeName()).Set(a.Get())
        mesh=UsdGeom.Mesh.Define(stage,fixture_path+"/SourceThreadedNut")
        geometry=repo/"artifacts/kcg_connector/isaac/te_full_assembly_20260905/representative_thread_geometry_clean_01/coupling_nut_standard_inner_thread.npz"
        data=np.load(geometry);v=data["vertices_m"].astype(np.float32);f=data["faces"].astype(np.int32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f),3,np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()));mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
        sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim());sdf.CreateSdfResolutionAttr(1024)
        sdf.CreateSdfSubgridResolutionAttr(6);sdf.CreateSdfNarrowBandThicknessAttr(.002);sdf.CreateSdfTriangleCountReductionFactorAttr(1.)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh.GetPrim()).CreateWeldToleranceAttr(0.)
        col=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());col.CreateContactOffsetAttr(.00005);col.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose="physics")
        if lab_recipe is not None and lab_recipe.get("render_original_nut_visual",False):
            from build_te_free_split_plug import NUT_VISUAL
            # The capped thread SDF remains the same collision-only diagnostic
            # representation. Display the original supplier-derived visual
            # layer instead of exposing its hidden collision closure faces.
            UsdGeom.Imageable(mesh.GetPrim()).GetVisibilityAttr().Set("invisible")
            visual=UsdGeom.Xform.Define(stage,fixture_path+"/OriginalNutCadVisual")
            visual.GetPrim().GetReferences().AddReference(str(NUT_VISUAL))
            if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                   for p in Usd.PrimRange(visual.GetPrim())):
                raise RuntimeError("the selected original Nut visual contains physics APIs")
            if not UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get():
                raise RuntimeError("display selection disabled the required Nut collider")
            (args.output/"mounted_nut_visual_authoring.json").write_text(json.dumps({
                "original_visual_source":str(NUT_VISUAL),"collision_mesh_path":str(mesh.GetPath()),
                "collision_enabled":True,"collision_geometry_or_material_changed":False,
                "visual_source_contains_physics":False,"source_mass_and_inertia_changed":False,
                "authored_before_physics":True,"not_a_formal_connector_model_delivery":True},indent=2)+"\n")
        clamp=UsdPhysics.FixedJoint.Define(stage,"/World/DeclaredFixtureMount")
        clamp.CreateBody1Rel().SetTargets([fixture_path]);clamp.CreateLocalPos0Attr(Gf.Vec3f(*position))
        clamp.CreateLocalRot0Attr(Gf.Quatf(quat[0],Gf.Vec3f(*quat[1:])))
        UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
        fixture_sensor_path=fixture_path
        contact_filters=[fixture_path]
    if args.nut_fingertip_sdf:
        from te_nut_fingertip_sdf import author_nut_only_fingertip_sdf
        fingertip_sdf_report=author_nut_only_fingertip_sdf(stage,"/World/HandArm",fixture_path,
            links=tuple(args.nut_fingertip_sdf_link or ('f1Link3','f2Link2','f3Link3')))
        (args.output/"nut_fingertip_sdf_authoring.json").write_text(json.dumps(fingertip_sdf_report,indent=2)+"\n")
    hand_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                and ("/handbase_link/" in str(p.GetPath()) or str(p.GetPath()).endswith("/handbase_link"))]
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI(prim).CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    contact_capacity=(max(4096,int(prepared["contact_recording"].get("minimum_contact_records",4096)))
                      if args.free_plug_in_socket else 4096)
    contacts=RigidPrim(hand_paths,resolve_paths=False,contact_filter_paths=contact_filters,max_contact_count=contact_capacity)
    probe_contact_paths=([str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                          and str(p.GetPath()).startswith("/World/HandArm/")] + [body_path,fixture_path]
                         if args.probe_additional_turn_deg is not None else [])
    probe_contacts=(RigidPrim(probe_contact_paths,resolve_paths=False,contact_filter_paths=contact_filters,max_contact_count=contact_capacity)
                    if probe_contact_paths else None)
    static_part_contact_recording = args.free_plug_in_socket and args.probe_additional_turn_deg is None
    parts=RigidPrim([body_path,fixture_path] if args.free_plug_in_socket else [fixture_path],resolve_paths=False,
        **({'contact_filter_paths':contact_filters,'max_contact_count':contact_capacity}
           if static_part_contact_recording else {}))
    tree=(None if args.robot_state_before_reset else world.scene.add(
        SingleArticulation("/World/HandArm/Geometry/world",name="wrist",reset_xform_properties=False)))
    fixture_tree = (world.scene.add(SingleArticulation(fixture_sensor_path,name="fixture_sensor",reset_xform_properties=False))
                    if args.main_read_sequence and not args.free_plug_in_socket else None)
    if scene.GetSolverTypeAttr().Get() != args.solver_type:
        raise RuntimeError("scene preparation changed the declared solver type")
    robot_initialization_report=None
    if args.robot_state_before_reset:
        # World.reset advances physics before tensor setters become available.
        # Supply the saved physical hand support during those ticks as well.
        # No connector pose or friction/collision parameter is changed here.
        source_positions={f"iiwa_joint_{i+1}":v for i,v in enumerate(sensor_sample["active_positions_rad"][:7])}
        source_positions.update({k:v["position_rad"] for k,v in physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"].items()})
        active_names=controller.ARM_JOINT_NAMES+controller.ACTIVE_HAND_JOINT_NAMES
        source_targets=dict(zip(active_names,np.r_[sensor_sample["arm_control"]["drive_target_rad"],sensor_sample["active_targets_rad"][7:]]))
        source_arm_damping=(float(physical_sample["arm_control"]["effective_arm_damping_nm_s_rad"])
                            if args.arm_damping is None else args.arm_damping)
        authored=[]
        for prim in stage.Traverse():
            name=prim.GetName()
            if not str(prim.GetPath()).startswith("/World/HandArm/") or name not in source_positions or not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            state=PhysxSchema.JointStateAPI.Apply(prim,UsdPhysics.Tokens.angular)
            state.CreatePositionAttr(float(np.degrees(source_positions[name])))
            state.CreateVelocityAttr(0.)
            drive=UsdPhysics.DriveAPI.Apply(prim,"angular")
            is_active=name in source_targets;is_arm=name in controller.ARM_JOINT_NAMES
            kp=(2500. if is_arm else 12.) if is_active else 0.
            kd=(source_arm_damping if is_arm else 2.) if is_active else 0.
            cap=(100. if is_arm else args.closing_drive_cap_nm if name in ('f1j2','f2j1','f3j2') else 1.) if is_active else 0.
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(np.radians(kp)))
            drive.CreateDampingAttr(float(np.radians(kd)))
            drive.CreateMaxForceAttr(cap)
            drive.CreateTargetPositionAttr(float(np.degrees(source_targets.get(name,source_positions[name]))))
            drive.CreateTargetVelocityAttr(0.)
            authored.append(name)
        if set(authored)!=set(source_positions):
            raise RuntimeError("source robot state did not map to every original joint")
        robot_initialization_report={"authored_before_world_reset":True,"source_positions_rad":source_positions,
            "source_applied_drive_targets_rad":source_targets,"all_fifteen_joints_authored":True,
            "angular_drive_usd_conversion":"Nm/rad to Nm/degree by pi/180; finite caps retained",
            "legacy_robot_reset_writer_attached":False,"object_pose_modified":False}
        (args.output/"robot_initialization_before_reset.json").write_text(json.dumps(robot_initialization_report,indent=2)+"\n")
    world.reset();world.pause()
    if args.physics_device=='cpu' and args.frozen_connector_model is not None:
        import carb.logging
        carb.logging.acquire_logging().set_level_threshold_for_source(
            'omni.physx.plugin',carb.logging.LogSettingBehavior.OVERRIDE,carb.logging.LEVEL_ERROR)
        (args.output/'logging_scope.json').write_text(json.dumps({
            'source':'omni.physx.plugin','after_reset_level':'ERROR',
            'reason':'Previously recorded repeated CPU SDF material-face report warnings; error logging retained',
            'physics_or_contact_processing_changed':False})+'\n')
    if args.robot_state_before_reset:
        tree=SingleArticulation("/World/HandArm/Geometry/world",name="wrist",reset_xform_properties=False)
        tree.initialize()
        native_initial=tree.get_joint_positions()
        native_initial=(native_initial.detach().cpu().numpy() if hasattr(native_initial,"detach") else np.asarray(native_initial)).reshape(-1)
        robot_initialization_report["native_positions_after_warmup_rad"]=dict(zip(tree.dof_names,map(float,native_initial)))
        robot_initialization_report["reset_physics_time_s"]=float(world.current_time)
        (args.output/"robot_initialization_before_reset.json").write_text(json.dumps(robot_initialization_report,indent=2)+"\n")
    passive_initialization=None
    if args.free_plug_in_socket and args.frozen_connector_model is None:
        # A legacy reader present during World.reset resets joint positions to
        # its cached drive targets (zero for the passive joint). Construct this
        # read-only observer after the initial reset to retain the USD state.
        fixture_tree=SingleArticulation(fixture_sensor_path,name="fixture_sensor",reset_xform_properties=False)
        fixture_tree.initialize()
        native_joint=fixture_tree.get_joint_positions()
        if hasattr(native_joint,"detach"):native_joint=native_joint.detach().cpu().numpy()
        else:native_joint=np.asarray(native_joint)
        observed_passive_angle=float(native_joint.reshape(-1)[0])
        phase_error=float(np.arctan2(np.sin(observed_passive_angle-source_passive_angle),
                                     np.cos(observed_passive_angle-source_passive_angle)))
        passive_initialization={"source_angle_rad":source_passive_angle,
                                "authored_before_world_reset":True,
                                "authored_angle_rad":authored_passive_angle,
                                "authored_phase_error_rad":authored_phase_error,
                                "native_angle_after_reset_rad":observed_passive_angle,
                                "read_only_observer_constructed_after_reset":True,
                                "phase_error_modulo_full_revolution_rad":phase_error,
                                "initial_velocity_rad_s":0.,"contact_warm_start_restored":False,
                                "scope":"LOCAL_STATIC_INITIAL_CONDITION_NOT_EXACT_FULL_EPISODE_REPLAY"}
        # The exact source-state authoring was checked before simulation above.
        # After reset's two physical ticks, a free joint may legitimately move.
        # In static diagnosis check the current joint against current body poses;
        # do not misclassify physical settling as a failed zero-time write.
        native_phase_bound=float(np.deg2rad(.01))
        passive_initialization.update(reset_physics_time_s=float(world.current_time),
            native_phase_check_bound_rad=native_phase_bound,
            native_phase_check_role="BOUNDED_INITIAL_SETTLING_AFTER_RESET_NOT_ZERO_TIME_IDENTITY")
        if args.probe_additional_turn_deg is None:
            with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
                current_p,current_q=parts.get_world_poses()
            current_rot=Rotation.from_quat(current_q.numpy()[:,[1,2,3,0]]).as_matrix()
            joint=UsdPhysics.RevoluteJoint(passive_joint)
            local_rot=joint.GetLocalRot0Attr().Get()
            joint_frame=Rotation.from_quat([*local_rot.GetImaginary(),local_rot.GetReal()]).as_matrix()
            axis=joint_frame@np.eye(3)[:,("X","Y","Z").index(joint.GetAxisAttr().Get())]
            source_relative=body_T[:3,:3].T@nut_T[:3,:3]
            current_relative=current_rot[0].T@current_rot[1]
            predicted_relative=Rotation.from_rotvec(axis*phase_error).as_matrix()@source_relative
            pose_error=float(Rotation.from_matrix(current_relative@predicted_relative.T).magnitude())
            zero_reset=bool(abs(np.arctan2(np.sin(observed_passive_angle),np.cos(observed_passive_angle)))<2e-6 and abs(np.arctan2(np.sin(source_passive_angle),np.cos(source_passive_angle)))>native_phase_bound)
            passive_initialization.update(
                native_phase_check_role="POST_WARMUP_JOINT_AND_NATIVE_POSE_CONSISTENCY_FOR_STATIC_DIAGNOSIS",
                old_phase_drift_reference_exceeded=abs(phase_error)>native_phase_bound,
                source_to_current_motion_is_recorded_not_an_assembly_pass=True,
                joint_pose_rotation_consistency_error_rad=pose_error,
                current_relative_rotation=current_relative.tolist(),
                source_relative_rotation=source_relative.tolist(),
                expected_body_frame_joint_axis=axis.tolist(),
                joint_pose_consistency_bound_rad=5e-5,
                unexpected_zero_reset_detected=zero_reset)
            if pose_error>5e-5 or zero_reset or abs(float(world.current_time)-2*dt)>1e-6:
                (args.output/"passive_initialization_readback.json").write_text(json.dumps(passive_initialization,indent=2)+"\n")
                raise RuntimeError("static initialization has inconsistent joint/pose state, a zero reset, or unexpected warmup time")
        (args.output/"passive_initialization_readback.json").write_text(json.dumps(passive_initialization,indent=2)+"\n")
        if args.probe_additional_turn_deg is not None and abs(phase_error)>native_phase_bound:
            raise RuntimeError(f"local passive-joint initial phase differs: source={source_passive_angle}, native={observed_passive_angle}")
    elif args.free_plug_in_socket:
        fixture_tree=None
        passive_initialization={'representation':'VALIDATED_REGULAR_REVOLUTE_WITH_FINITE_PASSIVE_RESISTANCE',
            'frozen_model':str(args.frozen_connector_model.resolve()),'object_articulation_reader_used':False,
            'source_pose_is_declared_cold_initial_state_not_restored_contact_history':True}
    settings={"arm_control_law":"NATIVE_FORCE_DRIVE_GRAVITY_EQUIVALENT_POSITION_BIAS_V1",
              "arm_stiffness":2500.,"arm_damping":60.,"hand_stiffness":12.,"hand_damping":2.,
              "arm_drive_maximum_effort_nm":100.,"hand_drive_maximum_effort_nm":1.}
    settings['closing_drive_maximum_effort_nm']=args.closing_drive_cap_nm
    settings["arm_damping"] = (float(physical_sample["arm_control"]["effective_arm_damping_nm_s_rad"])
                               if args.arm_damping is None else args.arm_damping)
    names=metadata["controller_outcome"]["native_drive_audit"]["dof_names"]
    robot_data=controller.create_native_gravity_compensated_robot(
        "/World/HandArm/Geometry/world",names,settings,
        initial_arm_positions=sensor_sample["active_positions_rad"][:7],
        initial_hand_positions=sensor_sample["active_positions_rad"][7:])
    robot,active,*_=robot_data
    if args.gpu_host_readback:
        readback={"gpu_dynamics_enabled":bool(scene.GetEnableGPUDynamicsAttr().Get()),
            "broadphase":str(scene.GetBroadphaseTypeAttr().Get()),
            "suppress_readback":carb.settings.get_settings().get("/physics/suppressReadback"),
            "data_device":str(SimulationManager.get_physics_sim_device()),
            "solver":args.solver_type,"physics_dt_s":dt,
            "scope":"GPU_DYNAMICS_WITH_HOST_RESULT_READBACK_NOT_CPU_SOLVER_MIGRATION"}
        (args.output/"physics_readback_mode.json").write_text(json.dumps(readback,indent=2)+"\n")
        if not readback["gpu_dynamics_enabled"] or readback["broadphase"]!="GPU" or readback["suppress_readback"]:
            raise RuntimeError("the requested data-path check did not retain GPU dynamics with host readback")
    if hand_friction_report is not None:
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            native_friction = [value.numpy()[0] for value in robot.get_dof_friction_properties(indices=0)]
        readback=[]
        for row in hand_friction_report["joints"]:
            index=robot.dof_names.index(row["joint"])
            values=[float(value[index]) for value in native_friction]
            if not np.allclose(values,[row["source_urdf_friction_nm"],row["source_urdf_friction_nm"],0.],rtol=1e-6,atol=1e-9):
                raise RuntimeError("native hand joint-friction tensor differs from the authored effort model")
            readback.append({"joint":row["joint"],"static_nm":values[0],"dynamic_nm":values[1],"viscous_nm_s_rad":values[2]})
        hand_friction_report["native_tensor_readback_after_reset"]=readback
        (args.output/"hand_joint_friction_authoring.json").write_text(json.dumps(hand_friction_report,indent=2)+"\n")
    actual={f"iiwa_joint_{i+1}":x for i,x in enumerate(sensor_sample["active_positions_rad"][:7])}
    actual.update({k:x["position_rad"] for k,x in physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"].items()})
    robot.set_dof_positions(np.array([[actual[n] for n in robot.dof_names]]))
    targets=np.r_[sensor_sample["arm_control"]["drive_target_rad"],sensor_sample["active_targets_rad"][7:]]
    robot.set_dof_position_targets(targets[None,:],indices=0,dof_indices=active)
    replay_targets=np.asarray([np.r_[s["arm_control"]["drive_target_rad"],s["active_targets_rad"][7:]] for s in replay_records])
    if replay_records:
        lo,hi=robot.get_dof_limits(indices=0,dof_indices=active)
        if not np.isfinite(replay_targets).all() or np.any(replay_targets<lo.numpy()[0]) or np.any(replay_targets>hi.numpy()[0]):
            raise ValueError("a recorded replay target is outside the existing joint limits")
    hb=next(i for i,p in enumerate(hand_paths) if p.endswith("/handbase_link"))
    row_index=tree._articulation_view._metadata.joint_indices["hand2arm"]+1
    with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
        masses=contacts.get_masses().numpy().reshape(-1);coms=contacts.get_coms()[0].numpy()
    stage.GetRootLayer().Export(str(args.output/"calibration_after_initialization.usda"))
    if args.wrist_reference_loads:
        from te_robot_wrist_reference_loads import run_robot_wrist_reference_loads
        result=run_robot_wrist_reference_loads(repository=repo,world=world,robot_data=robot_data,
            ft_tree=tree,contact_view=contacts,hand_paths=hand_paths,recipe=lab_recipe,
            source_sensor=sensor_sample,source_metadata=metadata,base_run=base_run,
            settings=settings,output=args.output)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    if lab_recipe is not None:
        from te_mounted_nut_torque_probe import run_mounted_nut_torque_probe
        result=run_mounted_nut_torque_probe(repository=repo,world=world,robot_data=robot_data,
            ft_tree=tree,contact_view=contacts,hand_paths=hand_paths,recipe=lab_recipe,
            fixture_pose=lab_fixture_pose,source_sensor=sensor_sample,source_metadata=metadata,
            base_run=base_run,settings=settings,output=args.output)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    if args.probe_additional_turn_deg is not None:
        from te_saved_nut_probe import run_probe
        result=run_probe(repository=repo, args=args, world=world, robot_data=robot_data,
                         ft_tree=tree, plug_tree=fixture_tree, contact_view=probe_contacts,
                         contact_paths=probe_contact_paths, prepared_scene=prepared,
                         source_metadata=metadata, source_sensor=sensor_sample,
                         source_rotation=control_record, passive_initialization=passive_initialization,
                         base_run=base_run, assembly_config_path=assembly_path)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    def host(v):return v.detach().cpu().numpy() if hasattr(v,"detach") else v.numpy() if hasattr(v,"numpy") else np.asarray(v)
    rows=[];read_deltas=[];joint_rows=[];part_rows=[];normal_load_rows=[];part_contact_rows=[];render_audit=None
    contact_component_rows=[];contact_actor_pairs=set();world.play()
    for step in range(round(2./dt)):
        if replay_records and step<len(replay_records):
            robot.set_dof_position_targets(replay_targets[step][None,:],indices=0,dof_indices=active)
        if step==round(1./dt) and args.paused_world_render:
            def snapshot():
                return {"positions":robot.get_dof_positions(indices=0).numpy().copy(),
                        "velocities":robot.get_dof_velocities(indices=0).numpy().copy(),
                        "wrist":host(tree.get_measured_joint_forces())[row_index].copy()}
            world.pause();before=snapshot();world.render();after=snapshot();world.play()
            render_audit={k:{"before":before[k].tolist(),"after":after[k].tolist(),
                             "maximum_change":float(np.max(np.abs(after[k]-before[k])))} for k in before}
        if args.main_read_sequence:
            robot.get_dof_positions(indices=0).numpy()
            robot.get_dof_velocities(indices=0).numpy()
            robot.get_dof_gravity_compensation_forces(indices=0).numpy()
        world.step(render=False)
        if args.main_read_sequence:
            before_queries=host(tree.get_measured_joint_forces())[row_index].copy()
            robot.get_dof_positions(indices=0).numpy()
            robot.get_dof_velocities(indices=0).numpy()
            robot.get_dof_projected_joint_forces(indices=0).numpy()
            after_projected=host(tree.get_measured_joint_forces())[row_index].copy()
            fixture_tree.get_measured_joint_forces()
            after_fixture=host(tree.get_measured_joint_forces())[row_index].copy()
            read_deltas.append([*(after_projected-before_queries),*(after_fixture-after_projected)])
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            pos,ori=contacts.get_world_poses();vel,angvel=contacts.get_velocities()
        pos=pos.numpy();rot=Rotation.from_quat(ori.numpy()[:,[1,2,3,0]]).as_matrix();origin=pos[hb]
        centers=pos+np.einsum("nij,nj->ni",rot,coms);gravity=masses[:,None]*np.array([0.,0.,-9.81])
        expected=np.r_[gravity.sum(0),np.cross(centers-origin,gravity).sum(0)]
        gravity_wrench=expected.copy();normal_wrench=np.zeros(6);friction_wrench=np.zeros(6)
        normal,points,normals,_,counts,starts,actor_ids=contacts.get_raw_contact_data()
        normal=normal.numpy().ravel();points=points.numpy();normals=normals.numpy()
        normal_load=np.zeros(len(hand_paths))
        # Starts/counts have one row per sensor and one column per filter.
        # A free connector uses multiple filters; flattened indices are not
        # hand-link indices and must not be used to label force records.
        contact_starts=starts.numpy().reshape(len(hand_paths),-1)
        contact_counts=counts.numpy().reshape(len(hand_paths),-1)
        for sensor_index in range(len(hand_paths)):
            for start,count in zip(contact_starts[sensor_index],contact_counts[sensor_index]):
                sl=slice(int(start),int(start+count));F=normal[sl,None]*normals[sl]/dt
                wrench=np.r_[F.sum(0),np.cross(points[sl]-origin,F).sum(0)]
                expected+=wrench;normal_wrench+=wrench
                normal_load[sensor_index]+=float(np.linalg.norm(F,axis=1).sum())
                if count:
                    for partner in contacts.get_actor_paths_from_ids(actor_ids[sl].to("cpu")):
                        contact_actor_pairs.add((hand_paths[sensor_index],partner))
        friction,points,counts,starts=contacts.get_friction_data();friction=friction.numpy()/dt;points=points.numpy()
        for start,count in zip(starts.numpy().ravel(),counts.numpy().ravel()):
            sl=slice(int(start),int(start+count));F=friction[sl]
            wrench=np.r_[F.sum(0),np.cross(points[sl]-origin,F).sum(0)]
            expected+=wrench;friction_wrench+=wrench
        net_force=contacts.get_net_contact_forces(dt=dt).numpy().sum(axis=0)
        filtered_force=contacts.get_contact_force_matrix(dt=dt).numpy().sum(axis=(0,1))
        contact_component_rows.append(np.r_[gravity_wrench,normal_wrench,friction_wrench,net_force,filtered_force])
        raw=host(tree.get_measured_joint_forces())[row_index];canonical=np.r_[-rot[hb]@raw[:3],-rot[hb]@raw[3:]]
        momentum=(masses[:,None]*vel.numpy()).sum(0)
        rows.append([*canonical,*expected,*(canonical-expected),*momentum,*origin])
        joint_rows.append([*robot.get_dof_positions(indices=0).numpy()[0],
                           *robot.get_dof_velocities(indices=0).numpy()[0]])
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            part_pos,part_ori=parts.get_world_poses();part_vel,part_angvel=parts.get_velocities()
        part_rows.append(np.c_[part_pos.numpy(),part_ori.numpy(),part_vel.numpy(),part_angvel.numpy()])
        normal_load_rows.append(normal_load)
        if static_part_contact_recording:
            f,_,_,separation,counts,starts,_=parts.get_raw_contact_data()
            f=f.numpy().ravel();separation=separation.numpy().ravel();stats=[]
            for row_starts,row_counts in zip(starts.numpy().reshape(2,-1),counts.numpy().reshape(2,-1)):
                force=0.;minimum=0.;number=0
                for start,count in zip(row_starts,row_counts):
                    if not count:continue
                    sl=slice(int(start),int(start+count));force+=float(np.maximum(0.,f[sl]).sum()/dt)
                    minimum=min(minimum,float(separation[sl].min()));number+=int(count)
                stats.append([force,minimum,number])
            part_contact_rows.append(stats)
    a=np.array(rows);result={"scope":("LOCAL_FREE_BODY_NUT_SOCKET_CONTACT_WITH_ORIGINAL_HAND_NOT_FULL_ASSEMBLY"
        if args.free_plug_in_socket else "STATIC_ORIGINAL_HAND_WITH_EXPLICITLY_MOUNTED_NUT_FIXTURE_NOT_ASSEMBLY"),
      "source_run":str(args.run),"source_step":sample_step,"velocity_iterations":args.velocity_iterations,
      "physics_device":args.physics_device,
      "solver_type_readback":scene.GetSolverTypeAttr().Get(),
      "physics_dt_s":dt,"physics_hz":args.physics_hz,
      "position_iterations":args.position_iterations,"external_forces_every_iteration":bool(args.external_forces_every_iteration),"hand_paths":hand_paths,
      "native_drive_settings":settings,
      "robot_dof_names":list(robot.dof_names),
      "hand_joint_friction_authoring":hand_friction_report,
      "robot_initialization_before_reset":robot_initialization_report,
      "passive_joint_initialization":passive_initialization,
      "native_articulation_velocity_iterations": {str(p.GetPath()):PhysxSchema.PhysxArticulationAPI(p).GetSolverVelocityIterationCountAttr().Get()
          for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxArticulationAPI)},
      "main_native_read_sequence":args.main_read_sequence,
      "paused_world_render_audit":render_audit,
      "maximum_query_wrist_change": (np.max(np.abs(read_deltas),axis=0).tolist() if read_deltas else None),
      "native_hand_mass_kg":float(masses.sum()),"robot_pose_initialized_while_paused_before_measurement":True,
      "nut_pose_or_drive_targets_changed_after_measurement_start":bool(replay_records),
      "connector_pose_changed_after_measurement_start":False,
      "nut_fixture_mount_added":not args.free_plug_in_socket,
      "last_half_second_mean":a[-half_second:].mean(0).tolist(),"columns":"canonical_wrist_world[6],gravity_plus_contact_world[6],difference[6],hand_linear_momentum[3],wrist_position[3]",
      "half_second_before_midpoint_mean":a[half_second:2*half_second].mean(0).tolist(),
      "last_half_second_wrist_position_range_m":np.ptp(a[-half_second:,-3:],axis=0).tolist(),
      "native_momentum_change_per_second_n":((a[-1,18:21]-a[-half_second,18:21])/((half_second-1)*dt)).tolist()}
    result["last_half_second_hand_normal_load_n"]=np.mean(normal_load_rows[-half_second:],axis=0).tolist()
    if replay_records:
        result["scope"]="BOUNDED_SEALED_ROBOT_COMMAND_REPLAY_THEN_STATIC_HOLD_NOT_ASSEMBLY"
        result["source_command_replay"]={"source_step_ids":[s["step"] for s in replay_records],
            "replayed_command_count":len(replay_records),"then_hold_last_target":True,
            "original_drive_gains_caps_and_joint_limits_retained":True,
            "online_feedback_or_contact_truth_used":False,
            "initial_velocity_and_contact_history_not_restored":True}
        np.savez_compressed(args.output/"replayed_drive_targets.npz",targets_rad=replay_targets,
            source_steps=np.asarray([s["step"] for s in replay_records]))
    result["last_half_second_part_position_range_m"]=np.ptp(np.array(part_rows)[-half_second:,:,:3],axis=0).tolist()
    result["all_raw_hand_contact_actor_pairs"]=sorted(contact_actor_pairs)
    result["contact_component_columns"]="gravity[6],raw_normal_wrench[6],filtered_friction_wrench[6],native_net_contact_force[3],native_filtered_contact_force_matrix_sum[3]"
    result["last_half_second_contact_component_mean"]=np.mean(contact_component_rows[-half_second:],axis=0).tolist()
    np.savez_compressed(args.output/"samples.npz",values=a,joints=np.array(joint_rows),
                        parts=np.array(part_rows),hand_normal_load=np.array(normal_load_rows),
                        contact_components=np.array(contact_component_rows),part_contact_stats=np.array(part_contact_rows))
    if static_part_contact_recording:
        result['part_contact_stats_columns']='positive_normal_force_sum_n,minimum_separation_m,contact_count'
        result['maximum_part_contact_normal_sum_n']=np.max(np.array(part_contact_rows)[:,:,0],axis=0).tolist()
    if args.audit_source_key_sdf:
        import torch
        world.pause()
        core_path=prepared["report"]["grounding_band_contact_model"]["rigid_core_collision"]
        core=UsdGeom.Mesh.Get(stage,core_path)
        vertices=np.asarray(core.GetPointsAttr().Get(),dtype=np.float64)
        faces=np.asarray(core.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).reshape(-1,3)
        slab=(vertices[:,2]>=-.007646)&(vertices[:,2]<=-.000761)
        protruding=slab&(np.linalg.norm(vertices[:,:2],axis=1)>.0183)
        selected=np.flatnonzero(np.any(protruding[faces],axis=1)&np.all(slab[faces],axis=1))
        if len(selected)!=106:
            raise RuntimeError("source-key SDF query must retain the original 106 key triangles")
        triangles=vertices[faces[selected]]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        bary=np.asarray([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3,1/3,1/3]])
        surface=np.einsum("ij,kjl->kil",bary,triangles).reshape(-1,3)
        directions=np.repeat(normals,len(bary),axis=0)
        resolution=int(PhysxSchema.PhysxSDFMeshCollisionAPI(core.GetPrim()).GetSdfResolutionAttr().Get())
        spacing=float(np.ptp(vertices,axis=0).max()/resolution)
        points=np.concatenate([surface-spacing*directions,surface,surface+spacing*directions])
        before=robot.get_dof_positions(indices=0).numpy().copy()
        view=world.physics_sim_view.create_sdf_shape_view(core_path,len(points))
        if view.count!=1:
            raise RuntimeError("SDF reader did not bind exactly the existing rigid core")
        queried=host(view.get_sdf_and_gradients(torch.as_tensor(points[None,:,:],dtype=torch.float32,device=args.physics_device))).copy()[0]
        after=robot.get_dof_positions(indices=0).numpy().copy()
        # This installed tensor returns gradient XYZ then SDF, contrary to the
        # current high-level docstring. Paired normal-offset queries retain a
        # direct unit/layout check; never treat a unit gradient as metres.
        count=len(surface);sdf=queried[count:2*count,3]
        derivative=(queried[2*count:,3]-queried[:count,3])/(2*spacing)
        centroid_indices=np.arange(6,count,7)
        gradient_norm=np.linalg.norm(queried[count:2*count,:3],axis=1)
        if not (.8<float(np.median(derivative[centroid_indices]))<1.2
                and .8<float(np.median(gradient_norm))<1.2):
            raise RuntimeError("native SDF result layout did not satisfy the source-normal offset check")
        np.savez_compressed(args.output/"source_key_native_sdf_query.npz",query_points_mesh_local_m=points,
            raw_sdf_and_gradient=queried,source_surface_points_m=surface,source_face_indices=selected,
            face_sample_barycentric=bary,offset_directions=directions)
        sdf_report={"scope":"POST_MEASUREMENT_NATIVE_SDF_VS_ORIGINAL_KEY_SURFACE_SAMPLES",
            "collision_path":core_path,"source_key_triangle_count":106,"sampled_surface_point_count":count,
            "sdf_resolution":resolution,"grid_spacing_m":spacing,
            "source_surface_sdf_min_median_max_m":[float(sdf.min()),float(np.median(sdf)),float(sdf.max())],
            "source_surface_max_absolute_sdf_m":float(np.max(np.abs(sdf))),
            "normal_offset_finite_difference_median":float(np.median(derivative)),
            "face_centroid_normal_offset_finite_difference_median":float(np.median(derivative[centroid_indices])),
            "maximum_robot_state_change_during_queries_rad":float(np.max(np.abs(after-before))),
            "field_layout":"LOCAL_GRADIENT_XYZ_THEN_SDF_METERS_VERIFIED_WITH_NORMAL_OFFSETS",
            "sampling_is_not_a_global_error_bound_or_assembly_pass":True,
            "no_physics_step_or_asset_authoring_during_query":True,"online_control_use":False}
        (args.output/"source_key_native_sdf_query.json").write_text(json.dumps(sdf_report,indent=2)+"\n")
        result["post_measurement_source_key_sdf_audit"]=sdf_report
    if read_deltas:
        np.savez_compressed(args.output/"native_query_changes.npz",values=np.array(read_deltas))
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
except Exception:
    failed=True
    import traceback
    error=traceback.format_exc();(args.output/"error.txt").write_text(error);print(error,flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
