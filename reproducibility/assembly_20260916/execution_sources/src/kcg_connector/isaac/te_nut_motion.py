"""Small trajectory and readiness calculations shared by nut manipulation."""
from dataclasses import dataclass
import math
import numpy as np


MAXIMUM_NUT_STROKE_DEG = 120.


def bounded_release_hold_velocity(target, position, velocity, gravity, *, stiffness, damping, effort_limit):
    """Choose the closest-to-zero velocity reference feasible for a PD hold.

    An early turn can still have substantial velocity. An instantaneous zero
    reference must not request more braking torque than the existing drive.
    One percent command headroom covers numeric conversion; the hard gate stays.
    """
    arrays=[np.asarray(v,dtype=float) for v in (target,position,velocity,gravity)]
    if any(v.shape!=(7,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Seven finite arm signals are required for a release hold')
    kp,kd,limit=map(float,(stiffness,damping,effort_limit))
    if not all(map(math.isfinite,(kp,kd,limit))) or min(kp,kd,limit)<=0:
        raise ValueError('Finite positive PD gains and drive limit are required')
    target,position,velocity,gravity=arrays
    zero_effort=kp*(target-position)-kd*velocity+gravity
    budget=.99*limit
    reference=np.clip(np.zeros(7),(-budget-zero_effort)/kd,(budget-zero_effort)/kd)
    if np.any(abs(reference)>abs(velocity)+1e-9):
        raise ValueError('The release hold cannot brake within the current drive limit')
    return reference,{'zero_velocity_request_effort_nm':zero_effort.tolist(),
        'bounded_velocity_reference_rad_s':reference.tolist(),
        'predicted_drive_effort_nm':(zero_effort+kd*reference).tolist(),
        'unchanged_drive_limit_nm':limit,'numeric_headroom_fraction':.01}


def source_probe_rotation_degrees(recipe):
    """Validate the diagnostic command before the simulator starts.

    Use the same finite stroke range as the continuation controller, including
    a last stroke that compensates an earlier reserve-triggered regrasp.
    """
    if 'rotation_degrees' not in recipe:
        return None
    degrees = float(recipe['rotation_degrees'])
    if not math.isfinite(degrees) or not 0 < degrees <= MAXIMUM_NUT_STROKE_DEG:
        raise ValueError('a local control stroke must be finite and within 0..120 degrees')
    return degrees


def remaining_command_stroke(planned_deg, executed_total_deg, budget_deg, *, final_interval):
    """Budget actual issued angles after a reserve-triggered early regrasp."""
    values=(float(planned_deg),float(executed_total_deg),float(budget_deg))
    planned,executed,budget=values
    if not all(map(math.isfinite,values)) or not 0<planned<=MAXIMUM_NUT_STROKE_DEG or min(executed,budget)<0:
        raise ValueError('finite positive stroke and nonnegative command budget required')
    remaining=budget-executed
    if remaining<=.01:return 0.
    return min(remaining,MAXIMUM_NUT_STROKE_DEG if final_interval else planned)


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


def bound_interface_velocity(velocity, axes, turn_axis, planar_speed, axial_speed,
                             tilt_speed, twist_speed):
    """Bound the total Cartesian request after pose-error feedback is added."""
    v=np.asarray(velocity,float);R=np.asarray(axes,float);axis=np.asarray(turn_axis,float)
    limits=np.asarray([planar_speed,axial_speed,tilt_speed,twist_speed],float)
    if (v.shape!=(6,) or R.shape!=(3,3) or axis.shape!=(3,)
            or not np.isfinite(np.r_[v,R.ravel(),axis,limits]).all() or np.any(limits<=0)
            or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.norm(axis)<1e-12):
        raise ValueError('Finite Cartesian velocity, orthogonal axes and positive bounds required')
    axis=axis/np.linalg.norm(axis)
    linear=R.T@v[:3]
    linear[:2]*=min(1.,limits[0]/max(np.linalg.norm(linear[:2]),1e-15))
    linear[2]=np.clip(linear[2],-limits[1],limits[1])
    along=float(v[3:]@axis);perpendicular=v[3:]-along*axis
    perpendicular*=min(1.,limits[2]/max(np.linalg.norm(perpendicular),1e-15))
    return np.r_[R@linear,perpendicular+axis*np.clip(along,-limits[3],limits[3])]


def successful_stroke_tilt(previous, axes, bending_moment, dt):
    """Bounded tilt law used by the archived successful segmented controller.

    Its 0.1 rad/(Nm s), 40 Nm/rad, 2 deg/s and 0.75 deg references
    are preserved. Only robot-side bending moment is an input.
    """
    from scipy.spatial.transform import Rotation
    previous=np.asarray(previous,float);axes=np.asarray(axes,float)
    moment=np.asarray(bending_moment,float)
    if moment.shape!=(2,) or not np.isfinite(moment).all() or dt<=0:
        raise ValueError('finite two-axis bending moment and positive step required')
    offset=axes.T@Rotation.from_matrix(previous).as_rotvec()
    omega=axes@np.r_[.1*(moment-40.*offset[:2]),0.]
    omega*=min(1.,np.deg2rad(2.)/max(np.linalg.norm(omega),1e-15))
    next_rotation=Rotation.from_rotvec(omega*dt).as_matrix()@previous
    vector=Rotation.from_matrix(next_rotation).as_rotvec();bound=np.deg2rad(.75)
    if np.linalg.norm(vector)>bound:
        next_rotation=Rotation.from_rotvec(vector*bound/np.linalg.norm(vector)).as_matrix()
        omega=Rotation.from_matrix(next_rotation@previous.T).as_rotvec()/dt
    return next_rotation,omega


def successful_stroke_joint_velocity(jacobian, twist, error, arm, bounds, maximum_speed):
    """Archived measured-pivot feedback and bounded redundant velocity solve."""
    from scipy.optimize import lsq_linear
    P=np.asarray(jacobian,float);arm=np.asarray(arm,float);bounds=np.asarray(bounds,float)
    error=np.asarray(error,float);twist=np.asarray(twist,float)
    if P.shape!=(6,7) or bounds.shape!=(7,2) or error.shape!=(6,) or twist.shape!=(6,) or maximum_speed<=0:
        raise ValueError('bounded seven-joint pivot control inputs required')
    feedback=3.*error;feedback[:3]=20.*error[:3]
    requested=twist+feedback
    low=np.maximum(-maximum_speed,2.*(bounds[:,0]+.005-arm))
    high=np.minimum(maximum_speed,2.*(bounds[:,1]-.005-arm))
    if np.any(low>=high):raise ValueError('successful-stroke joint velocity feasibility boundary')
    solved=lsq_linear(np.vstack((P,.0005*np.eye(7))),np.r_[requested,np.zeros(7)],
        bounds=(low,high),method='bvls',tol=1e-9,max_iter=30)
    if not solved.success:raise RuntimeError('successful-stroke bounded velocity solve did not converge')
    return solved.x,P@solved.x-requested


@dataclass
class LoadedTurnProgress:
    """One pausable angular reference; callers commit only after a physical step."""
    distance: float
    maximum_speed: float
    maximum_acceleration: float
    position: float = 0.
    speed: float = 0.

    def candidate(self, dt, scale):
        if not (dt > 0 and self.maximum_speed > 0 and self.maximum_acceleration > 0
                and all(map(math.isfinite,(dt,scale,self.distance,self.position,self.speed)))):
            raise ValueError('finite positive loaded-motion parameters required')
        remaining=max(0.,abs(self.distance)-abs(self.position));a=self.maximum_acceleration
        stopping_speed=max(0.,math.sqrt((a*dt)**2+2*a*remaining)-a*dt)
        target=min(self.maximum_speed*float(np.clip(scale,0.,1.)),stopping_speed)
        speed=float(np.clip(target,max(0.,self.speed-a*dt),self.speed+a*dt))
        travel=min(remaining,speed*dt)
        if remaining <= .5*a*dt*dt:
            travel=remaining;speed=0.
        sign=math.copysign(1.,self.distance)
        return self.position+sign*travel,speed,sign*travel/dt

    def commit(self, candidate):
        self.position,self.speed=candidate[:2]


def coordinated_turn_scale(error, previous_error, filtered_growth, speed, acceleration,
                           soft_error, hard_error, dt, unloading):
    """Brake before the tracking stop, preserving a bounded unload direction."""
    if not 0 < soft_error < hard_error or acceleration<=0 or dt<=0:
        raise ValueError('ordered tracking margins and positive timing required')
    growth=0. if previous_error is None else (error-previous_error)/dt
    filtered_growth+=dt/(.05+dt)*(growth-filtered_growth)
    predicted=error+max(0.,filtered_growth)*max(.05,speed/acceleration)
    scale=float(np.clip((hard_error-predicted)/(hard_error-soft_error),0.,1.))
    return (0. if unloading else scale),filtered_growth,predicted


def requires_axial_unload(correction_velocity, signed_turn_speed, lead, threshold):
    """A positive correction can still accompany net inward screw motion."""
    return correction_velocity+lead*signed_turn_speed/(2*math.pi)>threshold


def released_grip_readiness(geometry_clear, observed_moments, moment_tolerance):
    """Robot-side release evidence; a commanded open pose is insufficient."""
    if not math.isfinite(moment_tolerance) or moment_tolerance<=0:
        raise ValueError('positive finite release observation tolerance required')
    moments=np.asarray(observed_moments,dtype=float)
    valid=moments.shape==(3,) and bool(np.isfinite(moments).all())
    unloaded=valid and bool(np.max(np.abs(moments))<=moment_tolerance)
    return {'ready':bool(geometry_clear and unloaded),'actual_geometry_clear':bool(geometry_clear),
            'observed_root_moments_nm':moments.tolist(),'load_observation_valid':valid,
            'unloaded':unloaded,'moment_tolerance_nm':moment_tolerance,
            'physical_contact_truth_used':False}


def combine_with_axial_force_control(motion_velocity,position_feedback,axis):
    """Keep axial admittance free of a competing axial position integrator."""
    motion=np.asarray(motion_velocity,dtype=float);feedback=np.asarray(position_feedback,dtype=float)
    axis=np.asarray(axis,dtype=float)
    if (motion.shape!=(6,) or feedback.shape!=(6,) or axis.shape!=(3,)
            or not np.isfinite(np.r_[motion,feedback,axis]).all() or not np.isclose(np.linalg.norm(axis),1.)):
        raise ValueError('Finite six-axis velocities and a unit force-control axis required')
    command=motion+feedback
    command[:3]-=axis*float(axis@feedback[:3])
    return command


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
