"""Reproduce the recorder's seven-second stop on one moving cube, not a grasp."""
import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--robot-display-settings', action='store_true')
p.add_argument('--inspection-refresh', action='store_true')
args = p.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo/'src/kcg_connector/isaac'))
from isaacsim import SimulationApp
app = SimulationApp({'headless': not args.robot_display_settings, 'multi_gpu': False, 'fast_shutdown': True,
                     'extra_args': ['--/rtx/hydra/supportMultiTickRate=false'] if args.robot_display_settings else []})
try:
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from te_multiview_video import MultiViewVideoRecorder,create_inspection_camera,refresh_inspection_view
    from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target

    SimulationManager.set_physics_sim_device('cpu')
    world = World(stage_units_in_meters=1., physics_dt=1/960, rendering_dt=1/60,
                  backend='numpy', device='cpu', sim_params={'use_gpu_pipeline': False})
    stage = omni.usd.get_context().get_stage()
    scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    scene.CreateEnableGPUDynamicsAttr(False)
    scene.CreateBroadphaseTypeAttr('MBP')
    UsdPhysics.Scene(scene.GetPrim()).CreateGravityMagnitudeAttr(0.)
    cube = UsdGeom.Cube.Define(stage, '/World/MovingCube')
    cube.CreateSizeAttr(.05)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateVelocityAttr(Gf.Vec3f(.001, 0., 0.))
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.05)
    view = RigidPrim(['/World/MovingCube'], resolve_paths=False)
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
    paths = {name: '/World/Cameras/'+name for name in ('main', 'global', 'palm', 'wrist')}
    for path in paths.values():
        _author_camera(stage, path, _camera_cv_pose_from_eye_target([.2, -.2, .15], [0., 0., 0.]),
                       resolution=(640, 360), focal_length_mm=24., horizontal_aperture_mm=36.,
                       clipping_range_m=(.001, 5.), Gf=Gf, UsdGeom=UsdGeom)
    world.reset()
    world.pause()
    recorder = MultiViewVideoRecorder(rep=rep, world=world, camera_paths=paths,
                                      output_path=out/'diagnostic.mp4', physics_hz=960, fps=5)
    inspection = create_inspection_camera(world, paths['global']) if args.inspection_refresh else None
    inspection_expected = None
    tl = omni.timeline.get_timeline_interface()
    rows = []
    stream = (out/'capture_checks.jsonl').open('x', buffering=1)
    world.play()
    failure = None
    for step in range(1, 9601):
        world.step(render=False)
        if args.robot_display_settings and step in (1920, 5314):
            world.pause()
            product = rep.create.render_product(paths['palm'], (1280, 720))
            depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
            depth.attach([product.path])
            for _ in range(3):
                recorder._render_views()
            depth.detach()
            product.destroy()
            world.play()
        if step % 192:
            continue
        if inspection and step == 4800:
            camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(inspection))
            old_transform = camera_xform.GetLocalTransformation()
            old_transform.SetTranslateOnly(old_transform.ExtractTranslation()+Gf.Vec3d(.02, 0., 0.))
            camera_xform.MakeMatrixXform().Set(old_transform)
            inspection_expected = np.asarray(old_transform).copy()
        row = {'step': step, 'physics_time_before_s': float(world.current_time),
               'physics_index_before': int(world.current_time_step_index),
               'timeline_before_s': float(tl.get_current_time()),
               'timeline_end_s': float(tl.get_end_time()), 'auto_update_before': tl.is_auto_updating()}
        position_before = np.asarray(view.get_world_poses()[0]).copy()
        try:
            if args.inspection_refresh:
                row['inspection_refresh'] = refresh_inspection_view(world)
            recorder.capture_step(step=step, phase='moving_cube', simulation_time_s=step/960)
        except Exception as error:
            failure = str(error)
            row['failure'] = failure
        row.update(physics_time_after_s=float(world.current_time),
                   physics_index_after=int(world.current_time_step_index),
                   timeline_after_s=float(tl.get_current_time()),
                   native_position_change_m=float(np.max(abs(np.asarray(view.get_world_poses()[0])-position_before))))
        row['render_clock_audit'] = getattr(recorder, 'last_render_audit', None)
        if inspection_expected is not None:
            actual = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(inspection)).GetLocalTransformation())
            row['inspection_camera_edit_retained_error'] = float(np.max(abs(actual-inspection_expected)))
        stream.write(json.dumps(row)+'\n')
        rows.append(row)
        if failure:
            break
    world.pause()
    stream.close()
    result = {'scope': 'RECORDER_TIMELINE_BOUNDARY_REPRODUCTION_ONLY', 'failure': failure,
              'last_capture': rows[-1], 'capture_count': len(rows), 'video': recorder.close()}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
finally:
    app.close()
