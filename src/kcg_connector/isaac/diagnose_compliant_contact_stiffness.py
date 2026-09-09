#!/usr/bin/env python3
"""Check native compliant-contact stiffness on two gravity-loaded primitives.

No connector, hand, recorded assembly state, or robot controller is loaded.
This identifies solver/material semantics, not a TE grounding-spring constant.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False})
failed = False
try:
    import numpy as np
    import carb
    import omni.physx
    import omni.usd
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from scipy.spatial.transform import Rotation

    dt, mass, half_extent, stiffness, damping = 1/240, .1, .01, 1000., 5.
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    physics = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    physics.CreateGravityMagnitudeAttr(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    scene_api.CreateSolverTypeAttr("TGS")
    scene_api.CreateMinVelocityIterationCountAttr(1)
    scene_api.CreateMaxVelocityIterationCountAttr(1)
    material = UsdShade.Material.Define(stage, "/World/CompliantMaterial")
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(0.)
    mat.CreateDynamicFrictionAttr(0.)
    mat.CreateRestitutionAttr(0.)
    compliant = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    compliant.CreateCompliantContactStiffnessAttr(stiffness)
    compliant.CreateCompliantContactDampingAttr(damping)
    compliant.CreateCompliantContactAccelerationSpringAttr(False)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.01))
    floor.AddScaleOp().Set(Gf.Vec3d(.3, .1, .02))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor_collision = PhysxSchema.PhysxCollisionAPI.Apply(floor.GetPrim())
    floor_collision.CreateRestOffsetAttr(0.)
    floor_collision.CreateContactOffsetAttr(.002)
    floor_material = UsdShade.Material.Define(stage, "/World/RigidFrictionlessMaterial")
    floor_mat = UsdPhysics.MaterialAPI.Apply(floor_material.GetPrim())
    floor_mat.CreateStaticFrictionAttr(0.)
    floor_mat.CreateDynamicFrictionAttr(0.)
    floor_mat.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        floor_material, UsdShade.Tokens.strongerThanDescendants, "physics")
    views, names = [], ["Sphere", "Box"]
    for index, name in enumerate(names):
        path = "/World/"+name
        if name == "Sphere":
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(half_extent)
        else:
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(2*half_extent)
        shape.AddTranslateOp().Set(Gf.Vec3d((index-.5)*.08, 0., half_extent+.0005))
        prim = shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        properties = UsdPhysics.MassAPI.Apply(prim)
        properties.CreateMassAttr(mass)
        properties.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        inertia = .4*mass*half_extent**2 if name == "Sphere" else mass*(2*half_extent)**2/6
        properties.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia))
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body_api.CreateSolverPositionIterationCountAttr(32)
        body_api.CreateSolverVelocityIterationCountAttr(1)
        body_api.CreateLinearDampingAttr(0.)
        body_api.CreateAngularDampingAttr(0.)
        body_api.CreateSleepThresholdAttr(0.)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateRestOffsetAttr(0.)
        collision.CreateContactOffsetAttr(.002)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        views.append(world.scene.add(SingleRigidPrim(prim_path=path, name=name, reset_xform_properties=False)))
    counts, impulses = np.zeros(2, int), np.zeros((2, 3))

    def on_contact(headers, data):
        for header in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.actor0, header.actor1)]
            if "/World/Floor" not in paths:
                continue
            for i, name in enumerate(names):
                if "/World/"+name not in paths:
                    continue
                counts[i] += int(header.num_contact_data)
                sign = 1. if paths[0] == "/World/"+name else -1.
                for j in range(int(header.contact_data_offset), int(header.contact_data_offset+header.num_contact_data)):
                    impulses[i] += sign*np.asarray(data[j].impulse)

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    contact_view = TensorRigidPrim(["/World/"+name for name in names], resolve_paths=False,
                                  contact_filter_paths=["/World/Floor"], max_contact_count=64)
    stage.GetRootLayer().Export(str(args.output / "coupon_before_reset.usda"))
    world.reset()
    if not contact_view.is_physics_tensor_entity_valid():
        raise RuntimeError("the coupon contact tensor view is invalid")
    rows, contacts, forces = [], [], []
    tensor_counts, tensor_forces = [], []

    def array(v):
        return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)

    for _ in range(480):
        counts.fill(0)
        impulses.fill(0.)
        world.step(render=False)
        state = []
        for view in views:
            position, orientation = view.get_world_pose()
            state.append(np.r_[array(position), array(orientation), array(view.get_linear_velocity())])
        rows.append(state)
        contacts.append(counts.copy())
        forces.append(impulses.copy()/dt)
        impulse, _, normal, _, count, start, _ = contact_view.get_raw_contact_data(dt=1.)
        impulse = impulse.numpy().reshape(-1)
        normal = normal.numpy().reshape(-1, 3)
        count, start = count.numpy().reshape(-1), start.numpy().reshape(-1)
        net = np.zeros((2, 3))
        for i, (n, a) in enumerate(zip(count, start)):
            a, n = int(a), int(n)
            net[i] = (impulse[a:a+n, None]*normal[a:a+n]).sum(0)/dt
        tensor_counts.append(count.copy())
        tensor_forces.append(net)
    values, contacts, forces = np.asarray(rows), np.asarray(contacts), np.asarray(forces)
    tensor_counts, tensor_forces = np.asarray(tensor_counts), np.asarray(tensor_forces)
    cases = []
    for index, name in enumerate(names):
        z = values[-120:, index, 2]
        R = Rotation.from_quat(np.roll(values[-120:, index, 3:7], -1, axis=1)).as_matrix()
        bottom_extent = np.full(len(z), half_extent) if name == "Sphere" else half_extent*np.abs(R[:, 2, :]).sum(1)
        penetration = bottom_extent-z
        cases.append({"shape": name, "runtime_mass_kg": float(array(views[index].get_mass())),
                      "final_half_second_mean_surface_penetration_m": float(penetration.mean()),
                      "final_half_second_penetration_range_m": [float(penetration.min()), float(penetration.max())],
                      "effective_total_stiffness_from_weight_over_penetration_n_m": float(mass*9.81/penetration.mean()),
                      "native_contact_count_range": [int(contacts[-120:, index].min()), int(contacts[-120:, index].max())],
                      "mean_recorded_contact_force_world_n": forces[-120:, index].mean(0).tolist(),
                      "native_tensor_contact_count_range": [int(tensor_counts[-120:, index].min()), int(tensor_counts[-120:, index].max())],
                      "mean_native_tensor_contact_force_world_n": tensor_forces[-120:, index].mean(0).tolist(),
                      "final_linear_velocity_m_s": values[-1, index, 7:].tolist()})
    result = {"scope": "NATIVE_COMPLIANT_CONTACT_SEMANTICS_ON_SYNTHETIC_PRIMITIVES",
              "source_connector_assets_used_or_modified": False,
              "physics_dt_s": dt, "position_iterations": 32, "velocity_iterations": 1,
              "gpu_dynamics": world.get_physics_context().is_gpu_dynamics_enabled(),
              "material_stiffness_n_m": stiffness, "material_damping_ns_m": damping,
              "acceleration_spring": compliant.GetCompliantContactAccelerationSpringAttr().Get(),
              "single_contact_static_prediction_m": mass*9.81/stiffness,
              "post_start_object_pose_writes": False,
              "contact_processing_enabled": not carb.settings.get_settings().get_as_bool(SETTING_DISABLE_CONTACT_PROCESSING),
              "manufacturer_grounding_spring_calibrated": False, "cases": cases}
    np.savez_compressed(args.output / "samples.npz", state=values, contact_counts=contacts,
                        contact_forces_n=forces, tensor_contact_counts=tensor_counts,
                        tensor_contact_forces_n=tensor_forces)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    failed = True
    import traceback
    failure = traceback.format_exc()
    (args.output / "error.txt").write_text(failure)
    print(failure, flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
