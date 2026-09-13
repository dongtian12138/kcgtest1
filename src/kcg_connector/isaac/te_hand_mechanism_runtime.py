"""Shared source-hand transmission for all stages of one physical episode.

The original links and hinges remain. One palm motor drives both layout axes;
three finger motors drive measured four-bars. Only motor commands, native
drive forces and internal constraint tangents change after reset.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from te_hand_fourbar import author_fourbar_rods, load_fourbar_contract, update_fourbar_tangents
from te_worm_drive import WormDrive, WormReference

ACTIVE_HAND = ("f1j1", "f1j2", "f2j1", "f3j2")
FOLLOWERS = {"f1j1": "f3j1", "f1j2": "f1j3", "f2j1": "f2j2", "f3j2": "f3j3"}


def host(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def normalized_hand_reference(values, intervals):
    result = np.asarray(values, float).copy()
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Four finite independent hand references are required")
    for i, name in enumerate(ACTIVE_HAND):
        low, high = intervals[name]
        # Exact source zero differs from the feasible closed-loop lower stop
        # by CAD/STL rounding. Correct the command, never the physical state.
        if result[i] == 0. and 0. < low < 1e-4:
            result[i] = low
        if not low-1e-10 <= result[i] <= high+1e-10:
            raise ValueError(f"Hand command outside measured source travel: {name}={result[i]}")
    return result


def author_hand_mechanism(stage, repository, config_path, joint_parent, initial_positions):
    """Author before the first reset; preserve source geometry and travel."""
    from pxr import PhysxSchema, Sdf, UsdPhysics
    from omni.physx.bindings._physx import (
        JOINT_AXIS_API, JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
        JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
        JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)

    repository = Path(repository)
    config_path = Path(config_path)
    if not config_path.is_absolute(): config_path = repository/config_path
    settings = json.loads(config_path.read_text())
    if settings.get("schema") != "kcg.source_hand_runtime.v1":
        raise ValueError("Unknown source-hand runtime configuration")
    contract_path = repository/settings["fourbar_contract"]
    contract, couplings = load_fourbar_contract(contract_path)
    positions = dict(initial_positions)
    positions["f3j1"] = positions["f1j1"]
    intervals = {}
    for source in ACTIVE_HAND:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_parent+"/"+source))
        bounds = (math.radians(joint.GetLowerLimitAttr().Get()),math.radians(joint.GetUpperLimitAttr().Get()))
        if source == "f1j1": intervals[source] = bounds
        else:
            follower = FOLLOWERS[source]
            distal = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_parent+"/"+follower))
            distal_bounds = (math.radians(distal.GetLowerLimitAttr().Get()),math.radians(distal.GetUpperLimitAttr().Get()))
            intervals[source] = couplings[follower].source_interval(bounds,distal_bounds)
    refs = normalized_hand_reference([positions[n] for n in ACTIVE_HAND],intervals)
    for name,value in zip(ACTIVE_HAND,refs):positions[name]=float(value)
    for follower,coupling in couplings.items():
        positions[follower]=coupling.position_and_derivative(positions[coupling.source_joint])[0]
    for name,value in positions.items():
        prim=stage.GetPrimAtPath(joint_parent+"/"+name)
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError("Initial source revolute joint missing: "+name)
        state=PhysxSchema.JointStateAPI.Apply(prim,"angular")
        state.CreatePositionAttr(math.degrees(value));state.CreateVelocityAttr(0.)

    friction_rows=[]
    for name in (*ACTIVE_HAND,*FOLLOWERS.values()):
        prim=stage.GetPrimAtPath(joint_parent+"/"+name)
        api=PhysxSchema.PhysxJointAPI.Apply(prim)
        friction_rows.append({"joint":name,"previous_coefficient":api.GetJointFrictionAttr().Get(),
                              "urdf_provenance":prim.GetAttribute("urdf:dynamics:friction").Get()})
        api.CreateJointFrictionAttr(0.)
        prim.ApplyAPI(JOINT_AXIS_API,"angular")
        for attr in (JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
                     JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR):
            prim.CreateAttribute(attr,Sdf.ValueTypeNames.Float).Set(0.)
        active=name in ACTIVE_HAND
        drive=UsdPhysics.DriveAPI.Apply(prim,"angular");drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(math.radians(settings["transmission_stiffness_nm_rad"]) if active else 0.)
        drive.CreateDampingAttr(math.radians(settings["output_viscosity_nm_s_rad"]) if active else 0.)
        cap=settings["palm_transmission_boundary_nm"] if name=="f1j1" else settings["finger_transmission_boundary_nm"]
        drive.CreateMaxForceAttr(cap if active else 0.)
        drive.CreateTargetPositionAttr(math.degrees(positions[name]));drive.CreateTargetVelocityAttr(0.)

    rods=author_fourbar_rods(stage,joint_parent,contract_path,representation="tangent")
    palm=stage.GetPrimAtPath(joint_parent+"/f3j1")
    old_palm={a.GetName():str(a.Get()) for a in palm.GetAttributes() if "mimic" in a.GetName().lower()}
    if palm.HasAPI("NewtonMimicAPI"):palm.RemoveAPI("NewtonMimicAPI")
    if palm.GetAttribute("newton:mimicEnabled"):palm.GetAttribute("newton:mimicEnabled").Set(False)
    for axis in ("rotX","rotY","rotZ"):
        if palm.HasAPI(PhysxSchema.PhysxMimicJointAPI,axis):palm.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,axis)
    mimic=PhysxSchema.PhysxMimicJointAPI.Apply(palm,"rotX")
    mimic.CreateReferenceJointRel().SetTargets([joint_parent+"/f1j1"])
    mimic.CreateReferenceJointAxisAttr("rotX");mimic.CreateGearingAttr(-1.)
    mimic.CreateOffsetAttr(0.);mimic.CreateNaturalFrequencyAttr(0.)
    return {"settings":settings,"config_path":str(config_path),"contract_path":str(contract_path),
            "mechanism_id":contract["mechanism_id"],"couplings":couplings,"joint_parent":joint_parent,
            "initial_positions":positions,"intervals":intervals,"rods":rods,
            "friction_replacement":friction_rows,"old_palm_mimic":old_palm,
            "palm_active_motor_and_full_original_travel_retained":True}


class HandMechanismRuntime:
    def __init__(self, world, robot, setup, output, *, active_effort_caps):
        import omni.usd
        from pxr import UsdPhysics
        from isaacsim.core.experimental.prims import RigidPrim
        self.world,self.robot,self.setup=world,robot,setup
        self.stage=omni.usd.get_context().get_stage()
        self.names=list(robot.dof_names)
        self.indices=[self.names.index(n) for n in ACTIVE_HAND]
        self.followers=[self.names.index(FOLLOWERS[n]) for n in ACTIVE_HAND]
        self.reference=normalized_hand_reference([setup["initial_positions"][n] for n in ACTIVE_HAND],setup["intervals"])
        self.phase="initial_hold";self.steps=0;self.failure=None
        self.last_time=float(world.current_time)
        self.settings=setup["settings"];self.drives={}
        self.contract=json.loads(Path(setup["contract_path"]).read_text())
        wanted={row[k] for row in self.contract["finger_joints"].values()
                for k in ("parent_link","proximal_link","distal_link")}
        paths={p.GetName():str(p.GetPath()) for p in self.stage.Traverse()
               if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in wanted}
        if set(paths)!=wanted:raise ValueError("Source link pose witnesses are incomplete")
        self.pose_names=sorted(paths);self.pose_view=RigidPrim([paths[n] for n in self.pose_names],resolve_paths=False)
        self.set_caps(active_effort_caps,initial=True)
        self._authored_drive_properties=None
        self.last_native_state=None
        self.last_native_state_time=None
        self.wall_times={name:0. for name in ('read_before_s','fourbar_update_s','motor_solve_s',
            'native_physics_s','read_after_s','evidence_and_boundary_s')}
        self._original_step=world.step
        world.step=self.step
        world.hand_mechanism=self
        self.stream=gzip.open(Path(output)/"hand_mechanism_samples.jsonl.gz","wt",compresslevel=1)
        serial={k:v for k,v in setup.items() if k!="couplings"}
        serial.update(active_effort_caps=list(map(float,active_effort_caps)),source_names=ACTIVE_HAND,
                      physical_q_or_qd_writes_after_reset=False,all_world_step_calls_covered=True)
        (Path(output)/"hand_mechanism_authoring.json").write_text(json.dumps(serial,indent=2)+"\n")

    def set_caps(self, caps, *, initial=False):
        caps=np.asarray(caps,float)
        limits=np.array([self.settings["palm_transmission_boundary_nm"],*[self.settings["finger_transmission_boundary_nm"]]*3])
        if caps.shape!=(4,) or not np.isfinite(caps).all() or np.any(caps<=0.) or np.any(caps>limits):
            raise ValueError("Motor active effort references must remain inside the declared finite transmission range")
        for i,name in enumerate(ACTIVE_HAND):
            if initial:
                ref=WormReference(transmission_stiffness=self.settings["transmission_stiffness_nm_rad"],transmission_damping=0.,
                    output_viscosity=self.settings["output_viscosity_nm_s_rad"],input_viscosity=self.settings["input_viscosity"],
                    load_friction_ratio=self.settings["load_friction_ratio"],output_active_effort_reference=float(caps[i]),
                    transmission_effort_boundary=float(limits[i]))
                self.drives[name]=WormDrive(self.setup["initial_positions"][name],reference=ref,integration="passive_split")
            else:
                self.drives[name].reference=replace(self.drives[name].reference,output_active_effort_reference=float(caps[i]))

    def submit(self, hand_reference, phase):
        self.reference=normalized_hand_reference(hand_reference,self.setup["intervals"])
        self.phase=str(phase)

    def step(self, *args, **kwargs):
        from time import perf_counter
        from carts_v2.fast_json import dumps as encode_row
        started=perf_counter()
        if self.failure:raise RuntimeError(self.failure)
        h=float(self.world.get_physics_dt())
        if abs(float(self.world.current_time)-self.last_time)>1e-7:
            raise RuntimeError("Physics advanced outside the shared hand integration")
        if self.last_native_state is not None and self.last_native_state_time==float(self.world.current_time):
            q,v,_=self.last_native_state
        else:
            q=host(self.robot.get_dof_positions(indices=0))[0];v=host(self.robot.get_dof_velocities(indices=0))[0]
        if not np.isfinite(np.r_[q,v]).all():raise RuntimeError("Nonfinite hand state before physical step")
        before_fourbar=perf_counter();self.wall_times['read_before_s']+=before_fourbar-started
        tangents=update_fourbar_tangents(self.stage,self.setup["joint_parent"],self.setup["couplings"],
                                       {name:float(q[index]) for name,index in zip(ACTIVE_HAND,self.indices)})
        slopes={row["source"]:row["slope"] for row in tangents};slopes["f1j1"]=1.
        before_motor=perf_counter();self.wall_times['fourbar_update_s']+=before_motor-before_fourbar
        laws=[]
        for i,name in enumerate(ACTIVE_HAND):
            laws.append(self.drives[name].prepare_position(float(q[self.indices[i]]),float(v[self.indices[i]]),
                float(self.reference[i]),h,stiffness=self.settings["motor_position_kp"],damping=self.settings["motor_position_kd"]))
        properties=tuple((r['stiffness'],r['damping'],r['max_effort']) for r in laws)
        if properties!=self._authored_drive_properties:
            self.robot.set_dof_gains(np.array([[r["stiffness"] for r in laws]]),np.array([[r["damping"] for r in laws]]),indices=0,dof_indices=self.indices)
            self.robot.set_dof_max_efforts(np.array([[r["max_effort"] for r in laws]]),indices=0,dof_indices=self.indices)
            self._authored_drive_properties=properties
        self.robot.set_dof_position_targets(np.array([[r["position_target"] for r in laws]]),indices=0,dof_indices=self.indices)
        before_native=perf_counter();self.wall_times['motor_solve_s']+=before_native-before_motor
        result=self._original_step(*args,**kwargs)
        after_native=perf_counter();self.wall_times['native_physics_s']+=after_native-before_native
        elapsed=float(self.world.current_time)-self.last_time
        if abs(elapsed-h)>1e-7:
            raise RuntimeError("Shared hand integration requires exactly one physical tick")
        self.last_time=float(self.world.current_time)
        after=host(self.robot.get_dof_positions(indices=0))[0];velocity=host(self.robot.get_dof_velocities(indices=0))[0]
        effort=host(self.robot.get_dof_projected_joint_forces(indices=0))[0]
        self.last_native_state=(after,velocity,effort)
        self.last_native_state_time=float(self.world.current_time)
        after_read=perf_counter();self.wall_times['read_after_s']+=after_read-after_native
        from scipy.spatial.transform import Rotation
        p,r=self.pose_view.get_world_poses();p,r=host(p),host(r)
        rotations=Rotation.from_quat(r[:,[1,2,3,0]]).as_matrix()
        pose_index={n:i for i,n in enumerate(self.pose_names)}
        closure={}
        for data in self.contract["finger_joints"].values():
            ai=pose_index[data["parent_link"]];ci=pose_index[data["distal_link"]]
            a=p[ai]+rotations[ai]@np.asarray(data["base_anchor_parent_local_m"])
            c=p[ci]+rotations[ci]@np.asarray(data["distal_anchor_local_m"])
            closure[data["source_joint"]]=float(np.linalg.norm(c-a)-data["rod_length_m"])
        for i,name in enumerate(ACTIVE_HAND):
            measured=float(effort[self.indices[i]]+slopes[name]*effort[self.followers[i]])
            row=self.drives[name].complete(float(after[self.indices[i]]),float(velocity[self.indices[i]]),observed_drive_effort=measured)
            row.update(step=self.steps,physics_time_s=float(self.world.current_time),phase=self.phase,joint=name)
            row["rod_length_error_from_actual_body_poses_m"]=closure.get(name)
            if name=="f1j1":
                row["source_link_pose_witnesses"]={n:{"position_m":p[j].tolist(),"orientation_wxyz":r[j].tolist()}
                                                    for j,n in enumerate(self.pose_names)}
            self.stream.write(encode_row(row)+"\n")
            if row["elastic_effort_boundary_exceeded"] or row["drive_saturation"] or row["friction_heat_j"] < -1e-9:
                self.failure="Finite hand transmission boundary: "+name
        self.steps+=1
        self.wall_times['evidence_and_boundary_s']+=perf_counter()-after_read
        if self.steps%240==0:self.stream.flush()
        if self.failure:raise RuntimeError(self.failure)
        return result

    def close(self):
        if self.stream is not None:self.stream.close();self.stream=None
