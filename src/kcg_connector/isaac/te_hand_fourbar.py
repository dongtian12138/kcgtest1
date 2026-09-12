"""Author measured internal finger rods without changing source rigid bodies."""
import json
from pathlib import Path

from kcg_connector.grasp.robust.finger_fourbar import load_finger_fourbars


def load_fourbar_contract(path):
    path = Path(path).resolve()
    return load_finger_fourbars(path)


def author_fourbar_rods(stage, joint_parent_path, contract_path, *, finger_names=None,
                       representation="distance", axial_stiffness=1e7, axial_damping=100.):
    """Replace finger mimic relations before physics, using measured pin axes.

    A PhysX distance joint outside the articulation represents each massless
    connecting rod. Source link masses/inertias remain their existing lumped
    values. This is an internal mechanism constraint, not an object fixture.
    """
    import omni.timeline
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0:
        raise RuntimeError("Finger rods must be authored before physical time advances")
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-12:
        raise ValueError("Measured hand mechanism coordinates require a metre stage")
    if representation not in ("distance", "axial", "tangent"):
        raise ValueError("Unknown internal rod representation")
    if representation == "axial" and not (0<axial_stiffness<=1e8 and 0<=axial_damping<=1e4):
        raise ValueError("Finite bounded axial rod references required")
    contract, couplings = load_fourbar_contract(contract_path)
    selected = set(contract["finger_joints"]) if finger_names is None else set(finger_names)
    if not selected or not selected <= set(contract["finger_joints"]):
        raise ValueError("Unknown finger selection")
    records = []
    for finger in sorted(selected):
        row = contract["finger_joints"][finger]
        proximal = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_parent_path+"/"+row["source_joint"]))
        distal = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_parent_path+"/"+row["follower_joint"]))
        if not proximal or not distal:
            raise ValueError("Measured finger hinge not found in this asset")
        base = proximal.GetBody0Rel().GetTargets()
        near = proximal.GetBody1Rel().GetTargets()
        distal_parent = distal.GetBody0Rel().GetTargets()
        tip = distal.GetBody1Rel().GetTargets()
        if (len(base)!=1 or len(tip)!=1 or near!=distal_parent
                or base[0].name!=row["parent_link"] or tip[0].name!=row["distal_link"]):
            raise ValueError("Measured rod endpoints do not match the source rigid-body tree")
        prim = distal.GetPrim()
        old = {a.GetName():str(a.Get()) for a in prim.GetAttributes() if "mimic" in a.GetName().lower()}
        if prim.HasAPI("NewtonMimicAPI"):
            prim.RemoveAPI("NewtonMimicAPI")
        if prim.GetAttribute("newton:mimicEnabled"):
            prim.GetAttribute("newton:mimicEnabled").Set(False)
        for axis in ("rotX", "rotY", "rotZ"):
            if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(0.)
        drive.CreateMaxForceAttr(0.)
        path = joint_parent_path+"/"+finger+"_measured_rod"
        if stage.GetPrimAtPath(path):
            raise ValueError("A measured rod is already authored")
        if representation == "distance":
            rod = UsdPhysics.DistanceJoint.Define(stage, path)
            rod.CreateBody0Rel().SetTargets([Sdf.Path(base[0])])
            rod.CreateBody1Rel().SetTargets([Sdf.Path(tip[0])])
            rod.CreateLocalPos0Attr(Gf.Vec3f(*row["base_anchor_parent_local_m"]))
            rod.CreateLocalPos1Attr(Gf.Vec3f(*row["distal_anchor_local_m"]))
            rod.CreateMinDistanceAttr(float(row["rod_length_m"]))
            rod.CreateMaxDistanceAttr(float(row["rod_length_m"]))
            rod.CreateExcludeFromArticulationAttr(True)
            rod.CreateCollisionEnabledAttr(False)
        elif representation == "axial":
            # A two-attachment spatial tendon is a bilateral spring: PhysX
            # explicitly permits both push and pull. No length limit is used.
            # Its fixed rest length is the measured rigid rod length. The
            # finite stiffness is an uncalibrated rigid-rod approximation.
            root_name=finger+"_rod_A"
            leaf_name=finger+"_rod_C"
            root_api=PhysxSchema.PhysxTendonAttachmentRootAPI.Apply(stage.GetPrimAtPath(base[0]),root_name)
            root_attachment=PhysxSchema.PhysxTendonAttachmentAPI(stage.GetPrimAtPath(base[0]),root_name)
            root_attachment.CreateLocalPosAttr(Gf.Vec3f(*row["base_anchor_parent_local_m"]))
            root_api.CreateStiffnessAttr(float(axial_stiffness))
            root_api.CreateDampingAttr(float(axial_damping))
            root_api.CreateOffsetAttr(0.)
            leaf_api=PhysxSchema.PhysxTendonAttachmentLeafAPI.Apply(stage.GetPrimAtPath(tip[0]),leaf_name)
            leaf_attachment=PhysxSchema.PhysxTendonAttachmentAPI(stage.GetPrimAtPath(tip[0]),leaf_name)
            leaf_attachment.CreateLocalPosAttr(Gf.Vec3f(*row["distal_anchor_local_m"]))
            leaf_attachment.CreateParentLinkRel().SetTargets(base)
            leaf_attachment.CreateParentAttachmentAttr(root_name)
            leaf_attachment.CreateGearingAttr(1.)
            leaf_api.CreateRestLengthAttr(float(row["rod_length_m"]))
        else:
            import math
            initial_degrees=PhysxSchema.JointStateAPI(proximal.GetPrim(),"angular").GetPositionAttr().Get()
            if initial_degrees is None:
                raise ValueError("Tangent closure requires a declared initial source angle")
            q=math.radians(float(initial_degrees))
            value,slope=couplings[row["follower_joint"]].position_and_derivative(q)
            native=PhysxSchema.PhysxMimicJointAPI.Apply(prim,"rotX")
            native.CreateReferenceJointRel().SetTargets([proximal.GetPath()])
            native.CreateReferenceJointAxisAttr("rotX")
            native.CreateGearingAttr(-slope)
            native.CreateOffsetAttr(math.degrees(slope*q-value))
            native.CreateNaturalFrequencyAttr(0.)
            path=str(distal.GetPath())
        records.append({"finger":finger,"path":path,"body0":str(base[0]),"body1":str(tip[0]),
                        "anchor0_m":row["base_anchor_parent_local_m"],"anchor1_m":row["distal_anchor_local_m"],
                        "rod_length_m":row["rod_length_m"],"removed_linear_mimic":old,
                        "representation":representation,
                        "axial_stiffness_n_m":axial_stiffness if representation=="axial" else None,
                        "axial_damping_n_s_m":axial_damping if representation=="axial" else None})
    return {"mechanism_id":contract["mechanism_id"],"contract":str(Path(contract_path).resolve()),
            "source_rigid_body_geometry_mass_inertia_changed":False,"object_constraints_added":False,
            "connecting_rod_inertia":"MASSLESS_KINEMATIC_LINK_WITH_EXISTING_SOURCE_LUMPED_LINK_INERTIA_RETAINED",
            "axial_stiffness_and_damping_hardware_calibrated":False,
            "constraint_error_requires_physical_run_review":True,"rods":records}


def update_fourbar_tangents(stage, joint_parent_path, couplings, measured_source_angles):
    """Linearize fixed CAD closure at measured robot angles before a step.

    p - f'(q0)*q + f'(q0)*q0 - f(q0) = 0 has exactly the constraint
    Jacobian of the four-bar at q0. Native mimic impulses act on both DOFs;
    this does not command a distal motor or overwrite either actual angle.
    """
    import math
    from pxr import PhysxSchema
    rows=[]
    for follower,coupling in couplings.items():
        if coupling.source_joint not in measured_source_angles:
            continue
        q=float(measured_source_angles[coupling.source_joint])
        value,slope=coupling.position_and_derivative(q)
        api=PhysxSchema.PhysxMimicJointAPI(stage.GetPrimAtPath(joint_parent_path+"/"+follower),"rotX")
        if not api:
            raise ValueError("The tangent constraint was not created before physics")
        api.GetGearingAttr().Set(-slope)
        api.GetOffsetAttr().Set(math.degrees(slope*q-value))
        rows.append({"source":coupling.source_joint,"q_rad":q,"slope":slope,"offset_rad":slope*q-value})
    return rows
