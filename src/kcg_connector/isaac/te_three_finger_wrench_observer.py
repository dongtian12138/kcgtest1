"""Quasi-static three-pad force estimate using existing robot-side signals.

Three CAD pad points give nine unknown force components. The six-dimensional
wrist wrench and three physical proximal-joint reaction projections provide
nine balance equations. No simulated object pose or contact record is an input.
Distributed pad moments, contact-point migration and inertia remain model error;
numerical solvability is not a physical accuracy certificate.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np


class ThreeFingerWrenchObserver:
    links=("f1Link3","f2Link2","f3Link3")
    joints=("f1j2","f2j1","f3j2")
    subtrees=(("f1Link2","f1Link3"),("f2Link1","f2Link2"),("f3Link2","f3Link3"))

    def __init__(self,repository,model,source_geometry_plan):
        self.model=model
        self.geometry_path=Path(source_geometry_plan)
        if not self.geometry_path.is_absolute():self.geometry_path=Path(repository)/self.geometry_path
        geometry=json.loads(self.geometry_path.read_text())
        self.body_from_hand=np.asarray(geometry["canonical_body_from_hand_for_nut_grasp"],float)
        self.hand_from_body=np.linalg.inv(self.body_from_hand)
        q=np.r_[np.zeros(7),geometry["first_contact_hand_positions_rad"]]
        fk={k:np.asarray(v) for k,v in model.forward_kinematics(q,enforce_limits=False).items()}
        body=fk["handbase_link"]@self.hand_from_body
        records={r["link"]:r for r in geometry["first_contacts"]}
        self.points_local={}
        for name in self.links:
            r=records[name]
            if not r["nearest_original_source_face_is_pad"] or r["positive_gap_before_first_contact_m"]<=0:
                raise ValueError("source CAD positive-gap pad points are required")
            point=np.asarray(r["nearest_nut_point_before_contact_body_frame_m"],float)
            world=body[:3,:3]@point+body[:3,3];link=fk[name]
            self.points_local[name]=link[:3,:3].T@(world-link[:3,3])
        self.inertials={}
        for link in ET.parse(Path(repository)/"src/iiwa_description/urdf/hand.xacro").getroot().findall("link"):
            i=link.find("inertial")
            if i is not None:
                self.inertials[link.get("name")]=(float(i.find("mass").get("value")),
                    np.fromstring(i.find("origin").get("xyz"),sep=" "))
        self.joint_origins={name:model.joints[name].origin_transform() for name in self.joints}
        self.tare_reaction=None;self.tare_gravity=None
        # Characteristic geometry length only normalizes mixed force/moment
        # units when reporting matrix condition; it does not change the solve.
        self.row_scale=np.r_[np.ones(3),np.full(6,1/.15)]

    @staticmethod
    def _cross_matrix(p):
        return np.array([[0.,-p[2],p[1]],[p[2],0.,-p[0]],[-p[1],p[0],0.]])

    def _system(self,q,fk=None):
        if fk is None:fk=self.model.forward_kinematics(q,enforce_limits=False)
        fk={k:np.asarray(v) for k,v in fk.items()};hand=fk["handbase_link"]
        A=np.zeros((9,9));points=[];gravity=[]
        body=hand@self.hand_from_body
        axis=body[:3,2];origin=body[:3,3]
        for i,(name,jname,subtree) in enumerate(zip(self.links,self.joints,self.subtrees)):
            link=fk[name];point=link[:3,:3]@self.points_local[name]+link[:3,3];points.append(point)
            joint=self.model.joints[jname];frame=fk[joint.parent_link]@self.joint_origins[jname]
            joint_axis=frame[:3,:3]@joint.axis;joint_origin=frame[:3,3]
            A[:3,3*i:3*i+3]=np.eye(3)
            A[3:6,3*i:3*i+3]=self._cross_matrix(point-hand[:3,3])
            A[6+i,3*i:3*i+3]=-np.cross(joint_axis,point-joint_origin)
            g=0.
            for child in subtree:
                mass,com=self.inertials[child];position=fk[child][:3,:3]@com+fk[child][:3,3]
                g-=float(np.cross(position-joint_origin,[0.,0.,-9.81*mass])@joint_axis)
            gravity.append(g)
        points=np.asarray(points);normals=points-origin
        normals-=np.outer(normals@axis,axis);radius=np.linalg.norm(normals,axis=1)
        if np.any(radius<1e-6):raise ValueError("CAD force point lies on the planned grasp axis")
        normals/=radius[:,None]
        return A,points,np.asarray(gravity),normals

    def calibrate_free_space(self,encoder_samples,projected_reaction_samples):
        q=np.asarray(encoder_samples,float);reaction=np.asarray(projected_reaction_samples,float)
        if q.ndim!=2 or q.shape[1]!=11 or reaction.shape!=(len(q),3) or not len(q):
            raise ValueError("free-space encoder samples and three joint reactions must match")
        if not np.isfinite(q).all() or not np.isfinite(reaction).all():
            raise ValueError("nonfinite free-space sensor data")
        self.tare_reaction=reaction.mean(0)
        self.tare_gravity=np.mean([self._system(row)[2] for row in q],axis=0)

    def estimate(self,encoder_positions,projected_reactions,external_hand_wrench_world,*,fk=None):
        """Return estimated radial pad loads; retain negative/uncertain results.

        The wrist input is expressed in world axes about the handbase origin,
        after the caller's source-hand gravity and free-space bias compensation.
        The planned circular axis follows encoder FK and the source grasp plan.
        It is not read from a simulator object or from a contact table.
        """
        if self.tare_reaction is None:raise RuntimeError("free-space calibration is missing")
        q=np.asarray(encoder_positions,float);reaction=np.asarray(projected_reactions,float)
        wrench=np.asarray(external_hand_wrench_world,float)
        if q.shape!=(11,) or reaction.shape!=(3,) or wrench.shape!=(6,) or not np.isfinite(np.r_[q,reaction,wrench]).all():
            raise ValueError("finite encoder, joint-reaction and wrist signals are required")
        A,points,gravity,normals=self._system(q,fk=fk)
        b=np.r_[wrench,reaction-self.tare_reaction-(gravity-self.tare_gravity)]
        scaled=A*self.row_scale[:,None];singular=np.linalg.svd(scaled,compute_uv=False)
        if singular[-1]<=np.finfo(float).eps*9*singular[0]:
            raise ValueError("the current CAD/sensor force system is rank deficient")
        forces=np.linalg.solve(scaled,b*self.row_scale).reshape(3,3)
        projection=np.zeros((3,9))
        for i in range(3):projection[i,3*i:3*i+3]=normals[i]
        normal_map=projection@np.linalg.inv(A)
        return {"normal_force_n":np.sum(forces*normals,axis=1),
                "force_world_n":forces,"cad_force_points_world_m":points,
                "normalized_condition":float(singular[0]/singular[-1]),
                "normal_force_sensor_map":normal_map,
                "source_joint_gravity_reaction_nm":gravity,
                "axis_source":"ENCODER_HAND_POSE_AND_ORIGINAL_CAD_GRASP_PLAN",
                "object_or_contact_truth_used":False,
                "accuracy_bound_verified_online":False}
