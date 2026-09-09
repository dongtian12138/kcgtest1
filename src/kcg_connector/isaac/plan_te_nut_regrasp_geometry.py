#!/usr/bin/env python3
"""Check a finite nut-regrasp candidate against the current original nail hand.

Offline relative geometry only: no object truth, Isaac dynamics, or robot command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fcl
import numpy as np
import yaml

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from build_te_free_split_plug import _load_single_usd_mesh, BODY_VISUAL, NUT_VISUAL


def bvh(vertices, faces):
    model = fcl.BVHModel()
    model.beginModel(len(faces), len(vertices))
    model.addSubModel(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model)


def plan(repository: Path, output: Path, axial_shift_m: float, *, base_geometry=None,
         layout_deg=None, yaw_deg=0., source_joint_range=False):
    output.mkdir(parents=True, exist_ok=False)
    config = repository / "src/kcg_connector/config/te_nail_tip_body_grasp_v1.yaml"
    document = yaml.safe_load(config.read_text())
    inputs = load_v2_inputs(repository, config_path=config, object_id="te_deutsch_d38999_26fj35pn_step")
    source_plan = document["dynamic"]["nail_body_grasp_control_plan"]
    original_hand = np.asarray(source_plan["object_from_hand_row_major"]).reshape(4, 4)
    pregrasp = np.asarray(source_plan["pregrasp_joint_positions_rad"])
    open_hand = pregrasp - np.array([0.0, 0.08, 0.08, 0.08])
    closing_upper = np.asarray(source_plan["final_joint_positions_rad"])
    if base_geometry is not None:
        base=json.loads(Path(base_geometry).read_text())
        original_hand=np.asarray(base["canonical_body_from_hand_for_nut_grasp"])
        open_hand=np.asarray(base["open_hand_positions_rad"])
        closing_upper=np.asarray(base.get("finite_closing_goal_rad",closing_upper))
    if layout_deg is not None:
        open_hand[0]=np.deg2rad(layout_deg);closing_upper[0]=open_hand[0]
    if source_joint_range:
        for i,name in enumerate(("f1j2","f2j1","f3j2"),start=1):
            closing_upper[i]=inputs.robot_model.joints[name].limit.upper
        if not np.isfinite(closing_upper).all():raise ValueError("finite source joint limits required")
    target_hand = original_hand.copy()
    target_hand[:3, 3] += np.array([0.0, 0.0, axial_shift_m])
    angle=np.deg2rad(yaw_deg)
    turn=np.eye(4);turn[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    target_hand=turn@target_hand
    parts = {}
    for name, path in (("body", BODY_VISUAL), ("nut", NUT_VISUAL)):
        vertices, faces = _load_single_usd_mesh(path)
        parts[name] = bvh(vertices, faces)
    hand_objects = {}
    for name, triangles in inputs.hand_collision_triangles_by_link.items():
        triangles = np.asarray(triangles)
        hand_objects[name] = bvh(triangles.reshape(-1, 3), np.arange(triangles.size // 3).reshape(-1, 3))

    def update(q, object_from_hand):
        fk = inputs.robot_model.forward_kinematics(tuple(np.r_[np.zeros(7), q]), enforce_limits=False)
        inv_hand = np.linalg.inv(np.asarray(fk["handbase_link"]))
        poses={}
        for name, obj in hand_objects.items():
            pose = object_from_hand @ inv_hand @ np.asarray(fk[name])
            obj.setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
            poses[name]=pose
        return poses

    def distance(first, second):
        result = fcl.DistanceResult()
        return max(0.0, float(fcl.distance(first, second, fcl.DistanceRequest(), result)))

    approach = []
    for fraction in np.linspace(0, 1, 49):
        pose = original_hand.copy()
        pose[:3, 3] += fraction * np.array([0.0, 0.0, axial_shift_m])
        a=angle*fraction;R=np.eye(4);R[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        pose=R@pose
        update(open_hand, pose)
        distances = [(distance(obj, part), link, part_name) for link, obj in hand_objects.items()
                     for part_name, part in parts.items()]
        closest = min(distances)
        approach.append({"axial_shift_m": float(fraction * axial_shift_m), "minimum_plug_clearance_m": closest[0],
                         "hand_link": closest[1], "plug_part": closest[2]})
    contact_q, contacts = open_hand.copy(), []
    for index, name in ((1, "f1Link3"), (2, "f2Link2"), (3, "f3Link3")):
        q = open_hand.copy()
        if source_joint_range:
            import xml.etree.ElementTree as ET
            tree=ET.parse(repository/"src/iiwa_description/urdf/hand.xacro").getroot()
            follower={"f1Link3":"f1j3","f2Link2":"f2j2","f3Link3":"f3j3"}[name]
            origin=np.fromstring(tree.find(f"joint[@name='{follower}']/origin").get("xyz"),sep=" ")
            radius=float(np.linalg.norm(np.asarray(inputs.hand_collision_triangles_by_link[name]).reshape(-1,3),axis=1).max())
            # Both source hinges move one radian per actuator radian. Triangle
            # inequality bounds every point speed by distal spacing + 2*radius.
            speed_bound=float(np.linalg.norm(origin)+2*radius)
            low=high=float(open_hand[index])
            for _ in range(600):
                q[index]=high;update(q,target_hand)
                gap=distance(hand_objects[name],parts["nut"])
                body_gap=distance(hand_objects[name],parts["body"])
                if gap<=0:break
                if body_gap<=0:raise ValueError(f"{name} reaches Body before Nut")
                if high>=closing_upper[index]:raise ValueError(f"{name} does not contact Nut in its source joint range")
                low=high
                high=min(float(closing_upper[index]),high+min(.02,max(2e-6,.8*min(gap,body_gap)/speed_bound)))
            else:raise ValueError(f"conservative contact advancement did not converge for {name}")
        else:
            q[index] = closing_upper[index]
            update(q, target_hand)
            if distance(hand_objects[name], parts["nut"]) > 0:
                raise ValueError(f"finite closure does not reach the nut for {name}")
            low, high = open_hand[index], closing_upper[index]
        for _ in range(24):
            middle = (low + high) / 2
            q[index] = middle
            update(q, target_hand)
            if distance(hand_objects[name], parts["nut"]) > 0:
                low = middle
            else:
                high = middle
        contact_q[index] = high
        q[index]=max(float(open_hand[index]),low-1e-6)
        poses=update(q,target_hand)
        nearest=fcl.DistanceResult()
        gap=fcl.distance(hand_objects[name],parts["nut"],
                         fcl.DistanceRequest(enable_nearest_points=True),nearest)
        near_hand=np.asarray(nearest.nearest_points[0]);near_nut=np.asarray(nearest.nearest_points[1])
        local=poses[name][:3,:3].T@(near_hand-poses[name][:3,3])
        import trimesh
        raw=trimesh.load(repository/f"src/iiwa_description/meshes/hand/{name}.STL",force="mesh",process=False)
        raw.vertices*=1000.
        _,source_distance,source_face=trimesh.proximity.closest_point(raw,local[None,:]*1000.)
        pad=np.load(repository/f"artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TERMINAL_PAD_EXACT_SOURCE_V2/{name}_PAD_BODY_raw_source_local_m.npz")
        pad_hit=bool(np.isin(source_face,pad["source_face_indices"])[0])
        q[index] = high + 1e-6
        update(q, target_hand)
        hit = fcl.CollisionResult()
        fcl.collide(hand_objects[name], parts["nut"], fcl.CollisionRequest(num_max_contacts=20, enable_contact=True), hit)
        contacts.append({"link": name, "first_contact_joint_rad": float(high),
                         "source_nut_contact_positions_m": [np.asarray(c.pos).tolist() for c in hit.contacts],
                         "source_contact_normals": [np.asarray(c.normal).tolist() for c in hit.contacts],
                         "positive_gap_before_first_contact_m":float(gap),
                         "nearest_hand_point_before_contact_body_frame_m":near_hand.tolist(),
                         "nearest_nut_point_before_contact_body_frame_m":near_nut.tolist(),
                         "nearest_original_source_face_is_pad":pad_hit,
                         "nearest_hand_point_source_surface_distance_m":float(source_distance[0]*.001),
                         "body_clearance_m_at_first_nut_contact": distance(hand_objects[name], parts["body"])})
    update(contact_q, target_hand)
    nonterminal = {name: min(distance(obj, part) for part in parts.values()) for name, obj in hand_objects.items()
                   if name not in {"f1Link3", "f2Link2", "f3Link3"}}
    result = {"scope": "OFFLINE_NUT_REGRASP_RELATIVE_GEOMETRY_NOT_DYNAMIC_SUCCESS",
        "source_config": str(config), "hand_variant": inputs.hand_variant,
        "base_nut_geometry_plan":str(base_geometry) if base_geometry else None,
        "hand_yaw_about_canonical_nut_axis_deg":yaw_deg,
        "source_joint_range_contact_search":source_joint_range,
        "geometry_search_upper_bounds_rad":closing_upper.tolist(),
        "palm_layout_rad": float(open_hand[0]), "candidate_axial_shift_from_body_grasp_m": axial_shift_m,
        "extra_open_joint_increment_rad": .08, "open_hand_positions_rad": open_hand.tolist(),
        "canonical_body_from_hand_for_nut_grasp": target_hand.tolist(),
        "first_contact_hand_positions_rad": contact_q.tolist(), "first_contacts": contacts,
        "minimum_open_approach_clearance_m": min(row["minimum_plug_clearance_m"] for row in approach),
        "approach_clearances": approach, "nonterminal_plug_clearances_at_contact_m": nonterminal,
        "object_pose_source_for_future_motion": "FRESH_PALM_POSITION_AND_AXIS_WITH_HAND_TRANSVERSE_BASIS",
        "axial_key_yaw_required_for_axisymmetric_nut_grasp": False,
        "remaining_checks": ["arm IK and current scene path", "current-image visibility and accuracy", "dynamic force-limited nut contact", "no source-surface forbidden contacts"]}
    (output / "geometry_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "approach_clearances"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--axial-shift-mm", type=float, default=12.0)
    parser.add_argument("--base-geometry",type=Path)
    parser.add_argument("--layout-deg",type=float)
    parser.add_argument("--yaw-deg",type=float,default=0.)
    parser.add_argument("--source-joint-range",action="store_true")
    args = parser.parse_args()
    plan(Path(__file__).resolve().parents[3], args.output.resolve(), args.axial_shift_mm * .001,
         base_geometry=args.base_geometry,layout_deg=args.layout_deg,yaw_deg=args.yaw_deg,
         source_joint_range=args.source_joint_range)
