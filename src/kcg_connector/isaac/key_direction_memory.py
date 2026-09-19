"""Carry one observed key direction using encoders and palm position/axis.

The palm's arbitrary transverse axes are never interpreted as measured yaw.
Small relative tilts are transported continuously with minimum rotation.
Additional axial slip remains unobserved and must be bounded by grip evidence.
"""
import math
import numpy as np


def _pose(value):
    p=np.asarray(value,dtype=float)
    if (p.shape!=(4,4) or not np.isfinite(p).all()
            or not np.allclose(p[3],[0,0,0,1],atol=1e-10,rtol=0)
            or not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-7,rtol=0)
            or np.linalg.det(p[:3,:3])<0):
        raise ValueError('A finite proper rigid transform is required')
    return p


def _unit(value):
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-10:
        raise ValueError('A finite nonzero axis is required')
    return v/np.linalg.norm(v)


def minimum_axis_rotation(old,new):
    a,b=_unit(old),_unit(new);v=np.cross(a,b);c=float(np.clip(a@b,-1.,1.))
    if c < -1+1e-8:raise ValueError('A reversed palm axis cannot update a held-body key reference')
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/(1+c)


class KeyDirectionMemory:
    def __init__(self):
        self.initialized=False;self.active=False;self.update_count=0;self.anchor_count=0

    def initialize(self,world_from_hand,world_from_body_with_key,sample_time_s):
        if self.initialized:raise RuntimeError('A second key initialization is not permitted in this episode')
        h,b=_pose(world_from_hand),_pose(world_from_body_with_key)
        if not math.isfinite(sample_time_s):raise ValueError('Finite sample time required')
        self.position_hand=h[:3,:3].T@(b[:3,3]-h[:3,3])
        self.axis_hand=h[:3,:3].T@b[:3,2]
        self.key_hand=h[:3,:3].T@b[:3,1]
        self.sample_time_s=float(sample_time_s)
        self.initialized=True;self.active=True
        self.anchor_count=1

    def reobserve_after_coarse_turn(self,world_from_hand,world_from_body_with_key,sample_time_s):
        """Explicit second visual anchor, after a completed physical coarse turn."""
        if not self.active or self.anchor_count!=1:
            raise RuntimeError('Exactly one active coarse key anchor must precede refinement')
        if not math.isfinite(sample_time_s) or sample_time_s<=self.sample_time_s:
            raise ValueError('The refinement must use a newer synchronized key image')
        h,b=_pose(world_from_hand),_pose(world_from_body_with_key)
        predicted=self.predict(h)
        axis=b[:3,2]
        old=_unit(predicted[:3,1]-axis*(axis@predicted[:3,1]))
        measured=b[:3,1]
        correction=math.degrees(math.atan2(axis@np.cross(old,measured),old@measured))
        self.position_hand=h[:3,:3].T@(b[:3,3]-h[:3,3])
        self.axis_hand=h[:3,:3].T@axis
        self.key_hand=h[:3,:3].T@measured
        self.sample_time_s=float(sample_time_s);self.anchor_count=2
        return {'previous_prediction_to_new_measurement_axial_deg':correction,
                'source':'SECOND_CURRENT_GLOBAL2_IMAGE_NOT_ENCODER_TARGET',
                'sample_time_s':self.sample_time_s}

    def update_palm(self,hand_from_camera,camera_from_body_five_dof,sample_time_s):
        if not self.active:raise RuntimeError('Key reference is not bound to the current Body grasp')
        if not math.isfinite(sample_time_s) or sample_time_s<self.sample_time_s:
            raise ValueError('Out-of-order palm measurement')
        hc,cb=_pose(hand_from_camera),_pose(camera_from_body_five_dof)
        position=hc[:3,:3]@cb[:3,3]+hc[:3,3]
        axis=_unit(hc[:3,:3]@cb[:3,2])
        swing=minimum_axis_rotation(self.axis_hand,axis)
        key=swing@self.key_hand
        key=_unit(key-axis*(axis@key))
        report={'translation_update_hand_m':(position-self.position_hand).tolist(),
                'tilt_update_rad':math.acos(float(np.clip(self.axis_hand@axis,-1,1))),
                'palm_axial_yaw_used':False,'additional_axial_slip_observed':False,
                'key_was_reobserved':False,'sample_time_s':float(sample_time_s)}
        self.position_hand=position;self.axis_hand=axis;self.key_hand=key
        self.sample_time_s=float(sample_time_s);self.update_count+=1
        return report

    def hand_from_body(self):
        if not self.active:raise RuntimeError('The Body grasp has ended; its hand relation is retired')
        result=np.eye(4)
        result[:3,:3]=np.column_stack((np.cross(self.key_hand,self.axis_hand),self.key_hand,self.axis_hand))
        result[:3,3]=self.position_hand
        return result

    def predict(self,world_from_hand):
        return _pose(world_from_hand)@self.hand_from_body()

    def retire(self,reason):
        if not reason:raise ValueError('Retirement reason required')
        self.active=False;self.retirement_reason=str(reason)

    def report(self):
        return {'initialized':self.initialized,'active_body_grasp':self.active,'anchor_count':self.anchor_count,
                'palm_update_count':self.update_count,'sample_time_s':getattr(self,'sample_time_s',None),
                'extra_axial_rotation_is_measured':False,
                'assumption':'ADDITIONAL_BODY_AXIAL_SLIP_MUST_FIT_REMAINING_KEYWAY_CLEARANCE_AND_REQUIRES_VERIFICATION',
                'retirement_reason':getattr(self,'retirement_reason',None)}
