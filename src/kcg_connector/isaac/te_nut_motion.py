"""Small trajectory and readiness calculations shared by nut manipulation."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ScalarMotion:
    distance: float
    maximum_speed: float
    maximum_acceleration: float

    def __post_init__(self):
        if not all(map(math.isfinite, (self.distance, self.maximum_speed, self.maximum_acceleration))):
            raise ValueError('finite motion parameters required')
        if min(self.maximum_speed, self.maximum_acceleration) <= 0:
            raise ValueError('positive speed and acceleration required')

    @property
    def ramp_time(self):
        return min(self.maximum_speed/self.maximum_acceleration,
                   math.sqrt(abs(self.distance)/self.maximum_acceleration))

    @property
    def duration(self):
        if self.distance == 0:return 0.
        return abs(self.distance)/(self.maximum_acceleration*self.ramp_time)+self.ramp_time

    def at(self, elapsed):
        """Position and velocity; zero velocity at both endpoints."""
        t=float(np.clip(elapsed,0.,self.duration));r=self.ramp_time
        if r == 0:return 0.,0.
        a=self.maximum_acceleration;v=a*r;T=self.duration
        if t<r:x=.5*a*t*t;speed=a*t
        elif t<=T-r:x=.5*a*r*r+v*(t-r);speed=v
        else:x=abs(self.distance)-.5*a*(T-t)**2;speed=a*(T-t)
        sign=math.copysign(1.,self.distance)
        return sign*x,sign*speed


def joint7_return_path(arm_target, target_joint7, lower, upper, dt,
                       maximum_speed=1., maximum_acceleration=2.):
    """Only joint7 changes its commanded position. Caller checks the swept path."""
    start=np.asarray(arm_target,dtype=float).copy();lower=np.asarray(lower);upper=np.asarray(upper)
    if start.shape!=(7,) or not math.isfinite(dt) or dt<=0:raise ValueError('invalid wrist-return input')
    if not np.isfinite(np.r_[start,target_joint7]).all():raise ValueError('nonfinite wrist return')
    if not lower[6]<=target_joint7<=upper[6]:raise ValueError('wrist return would exceed joint7 limits')
    profile=ScalarMotion(float(target_joint7-start[6]),maximum_speed,maximum_acceleration)
    count=max(1,math.ceil(profile.duration/dt))
    times=np.linspace(0.,count*dt,count+1)
    path=np.repeat(start[None,:],count+1,axis=0)
    path[:,6]+=np.array([profile.at(t)[0] for t in times])
    path[-1,6]=target_joint7
    if np.any(path<lower) or np.any(path>upper):raise ValueError('wrist return leaves arm bounds')
    return path,{'mode':'FREE_SPACE_JOINT7_ONLY','duration_s':count*dt,
        'joint7_travel_rad':float(target_joint7-start[6]),
        'maximum_planned_speed_rad_s':float(np.max(np.abs(np.diff(path[:,6])))/dt),
        'maximum_acceleration_rad_s2':maximum_acceleration,
        'other_joint_target_change_rad':0.}


def limit_joint_velocity_without_changing_direction(velocity, maximum_speed):
    values=np.asarray(velocity,dtype=float)
    scale=min(1.,float(maximum_speed)/max(float(np.max(np.abs(values))),1e-15))
    return values*scale,scale


def preload_readiness(moments, targets, elastic_margins, *, relative_tolerance=.05,
                      relative_range=.15, minimum_margin_nm=.10):
    values=np.asarray(moments,dtype=float);targets=np.asarray(targets,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3 or len(values)<2 or targets.shape!=(3,) or np.any(targets<=0):
        raise ValueError('three fingers and a measured readiness window required')
    mean=values.mean(axis=0);error=np.abs(mean-targets)/targets
    span=np.ptp(values,axis=0)/targets;margins=np.asarray(elastic_margins,dtype=float)
    reasons=[]
    if not np.isfinite(values).all() or not np.isfinite(margins).all():reasons.append('NONFINITE_GRIP_SIGNAL')
    if np.any(error>relative_tolerance):reasons.append('PRELOAD_TARGET_NOT_REACHED')
    if np.any(span>relative_range):reasons.append('PRELOAD_NOT_STABLE')
    if margins.shape!=(3,) or np.any(margins<minimum_margin_nm):reasons.append('INSUFFICIENT_TRANSMISSION_RESERVE')
    return {'ready':not reasons,'reasons':reasons,'mean_base_moments_nm':mean.tolist(),
        'relative_mean_error':error.tolist(),'relative_peak_to_peak':span.tolist(),
        'elastic_margins_nm':margins.tolist(),'window_samples':len(values)}


def grasp_relation_residual(error_world, axis_world, passive_axial_travel_m=0.):
    """Separate pad displacement from the captive nut's permitted axial play.

    The camera observes Body while the fingers hold Nut. Their initial axial
    coordinate is unobserved, so its possible change spans the working travel.
    This only changes interpretation of the image, never the physical limits.
    """
    error=np.asarray(error_world,dtype=float);axis=np.asarray(axis_world,dtype=float)
    travel=float(passive_axial_travel_m)
    if error.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[error,axis,travel]).all() or travel<0:
        raise ValueError('finite relative displacement, axis and nonnegative axial travel required')
    norm=float(np.linalg.norm(axis))
    if norm<1e-12:raise ValueError('nonzero connector axis required')
    axis=axis/norm;axial=float(axis@error);lateral=float(np.linalg.norm(error-axis*axial))
    unexplained_axial=max(0.,abs(axial)-travel)
    return {'lateral_m':lateral,'axial_m':axial,'unexplained_axial_m':unexplained_axial,
            'unexplained_norm_m':math.hypot(lateral,unexplained_axial)}


def classify_observed_progress(previous, current, settings, progress_anchor=None):
    """Classify current visual progress; seating is a candidate for final audit."""
    depth=float(current['depth_m']);torque=abs(float(current['torsion_nm']))
    if depth>float(settings['nominal_seated_depth_m'])+float(settings['visual_seating_tolerance_m']):
        return 'OBSERVED_DEPTH_OVERRUN'
    if previous is None:return 'OBSERVING'
    advance=depth-float(previous['depth_m'])
    angle=abs(float(current['command_deg'])-float(previous['command_deg']))
    total=abs(float(current['command_deg']))
    duration=float(current['time_s'])-float(previous['time_s'])
    if duration<=0:return 'OBSERVING'
    if (depth>=float(settings['nominal_seated_depth_m'])-float(settings['visual_seating_tolerance_m'])
            and abs(advance)<=float(settings['stable_depth_tolerance_m'])
            and torque>=float(settings['seating_minimum_torque_nm'])):
        return 'VISUAL_SEATING_CANDIDATE'
    anchor=progress_anchor if progress_anchor is not None else previous
    progress=depth-float(anchor['depth_m'])
    progress_angle=abs(float(current['command_deg'])-float(anchor['command_deg']))
    play=float(settings.get('captive_nut_axial_travel_m',0.))
    minimum_progress=float(settings['minimum_observed_progress_m'])
    if play>0:
        # A stationary Body cannot distinguish lost grip from clearance takeup
        # until the commanded lead exceeds the entire permitted relative travel.
        predicted=float(settings['thread_lead_m'])*progress_angle/360.
        unexplained=predicted-play-float(settings['maximum_grasp_relation_error_m'])
        enough_motion=unexplained>=minimum_progress
        minimum_progress=max(minimum_progress,unexplained)
    else:
        enough_motion=progress_angle>=float(settings['minimum_progress_check_angle_deg'])
    if (total>=float(settings.get('minimum_loaded_command_deg',40.))
            and enough_motion and progress<minimum_progress):
        if float(current.get('grasp_relation_error_m',0.))>float(settings.get('slip_relation_error_m',.0001)):
            return 'RECOVERABLE_GRIP_SLIP'
        if torque>=float(settings.get('stall_torque_nm',.8)):
            return 'RECOVERABLE_CONTACT_STALL'
        return 'RECOVERABLE_NO_PROGRESS'
    return 'ADVANCING' if advance>0 else 'OBSERVING'
