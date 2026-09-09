"""Map the source URDF hand friction efforts to dimensional PhysX joint friction.

URDF revolute friction is an effort in N m. The legacy PhysX jointFriction
attribute is a load-dependent, unitless coefficient and is not the same model.
This authoring helper never modifies the source asset or runs after physics.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def author_urdf_hand_friction_efforts(repository, stage, robot_root):
    import omni.timeline
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics
    from omni.physx.bindings._physx import (
        JOINT_AXIS_API, JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
        JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
        JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0.:
        raise RuntimeError("hand friction may only be authored before physics")
    source = Path(repository)/"src/iiwa_description/urdf/hand.xacro"
    expected = ("f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3")
    xml_joints = {j.attrib["name"]: j for j in ET.parse(source).getroot().iter("joint")}
    prims = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_root))
             if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in expected}
    if set(prims) != set(expected):
        raise ValueError("the original eight hand revolute joints were not found")
    rows = []
    for name in expected:
        joint = prims[name]
        value = float(xml_joints[name].find("dynamics").attrib["friction"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("the source URDF friction effort must be finite and nonnegative")
        imported = joint.GetAttribute("urdf:dynamics:friction").Get()
        if imported is None or not np.isclose(imported, value, rtol=1e-6, atol=1e-9):
            raise ValueError("imported URDF friction provenance differs from the source")
        preserved = {a.GetName(): str(a.Get()) for a in joint.GetAttributes()
                     if a.GetName().startswith(("drive:", "physics:"))}
        legacy = PhysxSchema.PhysxJointAPI.Apply(joint)
        before = legacy.GetJointFrictionAttr().Get()
        legacy.CreateJointFrictionAttr(0.)
        joint.ApplyAPI(JOINT_AXIS_API, UsdPhysics.Tokens.angular)
        attributes = {
            JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR: value,
            JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR: value,
            JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR: 0.}
        for attr, setting in attributes.items():
            joint.CreateAttribute(attr, Sdf.ValueTypeNames.Float).Set(setting)
        if preserved != {a.GetName(): str(a.Get()) for a in joint.GetAttributes()
                         if a.GetName().startswith(("drive:", "physics:"))}:
            raise RuntimeError("friction authoring changed a joint frame, limit or drive")
        readback = {attr: float(joint.GetAttribute(attr).Get()) for attr in attributes}
        if not all(np.isclose(readback[k], v, rtol=1e-6, atol=1e-9) for k,v in attributes.items()):
            raise RuntimeError("dimensional friction did not read back")
        rows.append({"joint": name, "path": str(joint.GetPath()),
                     "source_urdf_friction_nm": value,
                     "previous_load_dependent_coefficient": before,
                     "load_dependent_coefficient_after": legacy.GetJointFrictionAttr().Get(),
                     "native_axis_readback": readback})
    return {"scope": "URDF_FRICTION_EFFORT_SEMANTICS_NOT_HARDWARE_CALIBRATION",
            "source_urdf": str(source), "joints": rows,
            "source_assets_mass_inertia_contact_materials_geometry_and_drives_changed": False,
            "joint_friction_representation_changed": True,
            "coulomb_assumption": "Use the sole source static effort as both static and dynamic Coulomb efforts; no separate dynamic value is calibrated.",
            "existing_drive_damping_retained": True,
            "body_nut_resistance_changed": False,
            "physics_after_start_authoring": False,
            "source_definitions": [
                "https://docs.ros.org/en/humble/Tutorials/URDF/Adding-Physical-and-Collision-Properties-to-a-URDF-Model.html",
                "https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/_build/physx/latest/class_px_articulation_joint_reduced_coordinate.html"]}
