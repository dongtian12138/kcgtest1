"""Experimental two-port worm transmission coupled to an implicit joint drive.

The motor input angle is an internal state referred to the output shaft. It
is not the simulated rigid-body angle. Load-dependent input friction can hold
that state without motor power; a finite motor effort can still drive either
direction. Parameters are development references, not measured hand ratings.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WormReference:
    transmission_stiffness: float = 120.
    transmission_damping: float = 2.
    input_viscosity: float = 1.
    load_friction_ratio: float = 1.2
    no_load_friction_effort: float = 0.
    output_active_effort_reference: float = 3.5
    transmission_effort_boundary: float = 3.5
    load_zero_smoothing: float = 0.
    output_viscosity: float = 0.

    @property
    def input_effort_boundary(self):
        # Preserve the declared forward output effort reference. This is a
        # loss-model conversion, never another multiplication by the gear ratio.
        return (1+self.load_friction_ratio)*self.output_active_effort_reference+self.no_load_friction_effort


def position_reference_for_input_velocity(drive, output_angle, desired_velocity, dt,
                                          motor_stiffness, motor_damping, lower, upper, *,
                                          clip_to_feasible=False):
    """Invert the existing passive-split motor PD/friction law, without state writes.

    The returned angle is a torque-equivalent motor reference, not an output
    position jump. The desired input velocity and native effort remain bounded.
    At zero requested speed, zero motor effort uses the existing self-lock.
    """
    q,v,h,kp,kd,low,high=map(float,(output_angle,desired_velocity,dt,motor_stiffness,motor_damping,lower,upper))
    if (drive.integration!='passive_split' or not all(map(math.isfinite,(q,v,h,kp,kd,low,high)))
            or h<=0 or kp<=0 or kd<0 or low>high):
        raise ValueError('finite passive-split motor command parameters required')
    p=drive.reference;z=float(drive.input_angle)
    def candidate(speed):
        next_z=z+h*speed;spring=p.transmission_stiffness*(next_z-q)
        if abs(spring)>p.transmission_effort_boundary:
            raise ValueError('requested input motion exceeds the unchanged elastic boundary')
        friction=math.copysign(p.no_load_friction_effort+p.load_friction_ratio*abs(spring),speed) if speed else 0.
        effort=p.input_viscosity*speed+spring+friction if speed else 0.
        if abs(effort)>p.input_effort_boundary:
            raise ValueError('requested input motion exceeds the unchanged motor effort boundary')
        target=next_z+(effort+kd*speed)/kp
        if not low<=target<=high:
            raise ValueError('friction-aware motor reference exceeds the unchanged grip position interval')
        return target,{'desired_input_velocity_rad_s':speed,'requested_motor_effort_nm':effort,
            'predicted_spring_effort_nm':spring,'friction_feedforward_nm':friction,
            'motor_position_reference_rad':target,'output_position_or_velocity_written':False,
            'unlimited_input_velocity_rad_s':v,'input_velocity_limited':False}
    try:return candidate(v)
    except ValueError as error:
        if not clip_to_feasible:raise
        reason=str(error)
    # Reduce the requested input speed, not the actual state or either limit.
    # Existing elastic violations still fail when the zero-speed state is tested.
    best=candidate(0.);low_scale=0.;high_scale=1.
    for _ in range(48):
        scale=(low_scale+high_scale)/2
        try:trial=candidate(scale*v)
        except ValueError:high_scale=scale
        else:best=trial;low_scale=scale
    best[1].update(input_velocity_limited=True,limiting_reason=reason)
    return best


def finger_force_motor_targets(mechanism, output_angles, targets, desired_moments,
                              measured_moments, dt, stiffness_reference, response_time,
                              maximum_speed, deadband, lower, upper, *, allow_closing=True):
    """Shared preload/turn actuator semantics; moments are not pad normal forces."""
    import numpy as np
    result=np.asarray(targets,float).copy()
    error=np.asarray(desired_moments,float)-np.asarray(measured_moments,float)
    if error.shape!=(3,) or not np.isfinite(error).all() or min(dt,stiffness_reference,maximum_speed)<=0:
        raise ValueError('three finite finger moments and positive drive references required')
    velocity=error/(stiffness_reference*max(response_time,dt))
    velocity[np.abs(error)<=deadband]=0.
    velocity=np.clip(velocity,-maximum_speed,maximum_speed)
    if not allow_closing:velocity=np.minimum(velocity,0.)
    rows=[]
    for i,name in enumerate(('f1j2','f2j1','f3j2')):
        result[i+1],row=position_reference_for_input_velocity(mechanism.drives[name],output_angles[i],
            velocity[i],dt,mechanism.settings['motor_position_kp'],mechanism.settings['motor_position_kd'],
            lower[i+1],upper[i+1])
        rows.append(row)
    return result,rows


class WormDrive:
    def __init__(self, initial_output_angle, *, reference=None, integration="endpoint",
                 friction_step="tangent"):
        self.reference = reference or WormReference()
        self.input_angle = float(initial_output_angle)
        self.input_velocity = 0.
        self.pending = None
        if integration not in ("endpoint","midpoint_native","passive_split"):
            raise ValueError("Unknown actuator integration scheme")
        self.integration=integration
        if friction_step not in ("tangent","frozen_magnitude"):
            raise ValueError("Unknown friction discretization")
        self.friction_step=friction_step
        self.last_observed_effort=None
        self.previous_output_velocity=None
        p = self.reference
        if not all(math.isfinite(x) and x>=0 for x in vars(p).values()):
            raise ValueError("Finite nonnegative transmission references required")
        if min(p.transmission_stiffness,p.input_viscosity,p.transmission_effort_boundary)<=0 or p.load_friction_ratio<1:
            raise ValueError("Self-lock hypothesis requires positive stiffness/viscosity and load friction ratio >=1")
        if integration=="passive_split" and (p.transmission_damping!=0 or p.load_zero_smoothing!=0):
            raise ValueError("Passive splitting uses an elastic transmission and explicit output viscosity; no series dashpot or friction smoothing")

    def _prepare_split(self,q,w,u,h,motor_pd=None):
        """Implicit motor substep with the measured output held fixed.

        B*v + K*(z+h*v-q) + friction = u. Solve the exact static or
        sliding inclusion, including a load sign crossing. Next PhysX moves
        the actual output with z fixed. Each elastic substep is passive;
        no future contact prediction or correction of the rigid state is used.
        The output viscous reference is distinct from a series dashpot.
        """
        p=self.reference;K=p.transmission_stiffness;B=p.input_viscosity;c=p.load_friction_ratio
        if B<=(c-1)*K*h:
            raise ValueError("Motor substep is not monotone at this timestep")
        old_z=self.input_angle;s=K*(old_z-q)
        if motor_pd is not None:
            target,kp,kd=motor_pd
            def motor_effort(vv):
                raw=kp*(target-old_z-h*vv)-kd*vv
                return max(-p.input_effort_boundary,min(p.input_effort_boundary,raw))
            u=motor_effort(0.)
        if abs(u-s)<=p.no_load_friction_effort+c*abs(s):
            v=0.;tau=s;friction=u-s
        elif motor_pd is not None:
            sv=1 if u-s>0 else -1
            # Motor PD and dry friction are implicit in the same scalar solve.
            # Using the previous input velocity in an explicit D term can
            # oscillate despite positive gains, especially when contact sticks.
            radius=(p.input_effort_boundary+(1+c)*abs(s)+p.no_load_friction_effort)/(B-(c-1)*K*h)+1.
            left,right=(0.,radius) if sv>0 else (-radius,0.)
            for _ in range(48):
                middle=(left+right)/2;tt=s+K*h*middle
                residual=B*middle+tt+sv*(p.no_load_friction_effort+c*abs(tt))-motor_effort(middle)
                if residual>0:right=middle
                else:left=middle
            v=(left+right)/2;tau=s+K*h*v;u=motor_effort(v)
            friction=sv*(p.no_load_friction_effort+c*abs(tau))
        else:
            sv=1 if u-s>0 else -1
            candidates=[]
            for st in (-1,1):
                factor=1+sv*c*st
                vv=(u-factor*s-sv*p.no_load_friction_effort)/(B+factor*K*h)
                tt=s+K*h*vv
                if sv*vv>0 and st*tt>=-1e-12:
                    if not any(abs(vv-old[0])<1e-10 for old in candidates):candidates.append((vv,tt))
            if len(candidates)!=1:
                raise RuntimeError("Exact implicit motor friction inclusion did not have one branch")
            v,tau=candidates[0];friction=sv*(p.no_load_friction_effort+c*abs(tau))
        z=old_z+h*v
        if abs(K*(z-q))>p.transmission_effort_boundary:
            raise ValueError("Elastic transmission exceeds the finite development reference")
        self.pending={"q":q,"w":w,"z":old_z,"next_z":z,"v":v,"u":u,"h":h,
                      "friction":friction,"input_spring_effort":tau,"motor_pd":motor_pd}
        return {"stiffness":K,"damping":p.output_viscosity,"position_target":z,"velocity_target":0.,
                "max_effort":p.transmission_effort_boundary,"branch":"stick" if v==0 else "move"}

    def prepare_position(self,output_angle,output_velocity,position_command,dt,*,stiffness,damping):
        """Finite motor PD acting on the input angle referred to the output.

        Its command may come from position control or an outer force loop.
        The force loop never directly overwrites the elastic output angle.
        Gains and motor effort are development references, not motor ratings.
        """
        q,w,target,h,kp,kd=map(float,(output_angle,output_velocity,position_command,dt,stiffness,damping))
        if (self.integration!="passive_split" or not all(map(math.isfinite,(q,w,target,h,kp,kd)))
                or h<=0 or kp<=0 or kd<0):
            raise ValueError("Motor position control requires the passive split model and finite positive timestep/gain")
        if self.pending is not None:
            raise RuntimeError("Complete the previous physical step before preparing another command")
        return self._prepare_split(q,w,0.,h,motor_pd=(target,kp,kd))

    def _complete_split(self,q,w,observed,*,native_capped_drive=False):
        s=self.pending;self.pending=None;p=self.reference;K=p.transmission_stiffness
        v=s["v"];z=s["next_z"];h=s["h"];dz=z-s["z"];dq=q-s["q"]
        self.input_angle=z;self.input_velocity=v
        self.last_observed_effort=observed;self.previous_output_velocity=s["w"]
        endpoint=K*(z-q)-p.output_viscosity*w
        if native_capped_drive:
            if observed is not None:raise ValueError('A capped PD estimate must not be labelled a measured drive force')
            tau=max(-p.transmission_effort_boundary,min(p.transmission_effort_boundary,endpoint))
        else:
            tau=endpoint if observed is None else float(observed)
        motor=s["u"]*dz;friction_heat=s["friction"]*dz;input_heat=p.input_viscosity*v*v*h
        motor_spring_change=.5*K*((z-s["q"])**2-(s["z"]-s["q"])**2)
        split_loss=.5*K*dz*dz
        spring_change=.5*K*((z-q)**2-(s["z"]-s["q"])**2)
        output_heat=p.output_viscosity*dq*dq/h
        # Native efforts arrive as float32 projections of active and mimic
        # joint reactions. Preserve the readback, but allow their few-ULP
        # rounding at the unchanged finite drive cap. Analytic-only updates
        # retain the previous double-precision comparison tolerance.
        effort_tolerance=(8.*2.**-23*max(1.,p.transmission_effort_boundary)
                          if observed is not None else 1e-7)
        return {"input_angle":z,"input_velocity":v,"output_angle":q,"output_velocity":w,
                "input_effort":s["u"],"transmission_effort":tau,"friction_effort":s["friction"],
                "input_substep_spring_effort_nm":s["input_spring_effort"],
                "predicted_branch_consistent":True,
                "drive_saturation":abs(tau)>p.transmission_effort_boundary+effort_tolerance,
                "native_effort_readback_tolerance_nm":effort_tolerance,
                "observed_effort_at_finite_limit":observed is not None and abs(tau)>=p.transmission_effort_boundary-effort_tolerance,
                "elastic_effort_boundary_exceeded":abs(K*(z-q))>p.transmission_effort_boundary,
                "spring_law_residual_nm":endpoint-tau,
                "motor_work_j":motor,"joint_work_j":tau*dq,"spring_energy_change_j":spring_change,
                "friction_heat_j":friction_heat,"damping_heat_j":output_heat,"input_viscous_heat_j":input_heat,
                "backward_euler_numerical_loss_j":split_loss,
                "motor_substep_energy_residual_j":motor_spring_change+friction_heat+input_heat+split_loss-motor,
                "port_energy_residual_j":spring_change+tau*dq+output_heat+input_heat+friction_heat+split_loss-motor,
                "output_integration_residual_rad":dq-h*w,"observed_drive_effort_nm":observed,
                "drive_effort_source":("CAPPED_ENDPOINT_PD_ESTIMATE_NOT_SENSOR" if native_capped_drive else
                    "SUPPLIED_DRIVE_MEASUREMENT" if observed is not None else "ANALYTIC_ENDPOINT"),
                "output_drive_work_is_estimate":native_capped_drive,
                "native_drive_clipping_predicted":bool(native_capped_drive and abs(endpoint)>=p.transmission_effort_boundary),
                "moving_friction_law_error_nm":0.,"drive_formula_minus_observed_nm":None if observed is None else endpoint-tau,
                "motor_position_command_rad":None if s["motor_pd"] is None else s["motor_pd"][0],
                "motor_effort_at_boundary":abs(s["u"])>=p.input_effort_boundary-1e-9,
                "integration":self.integration,"friction_step":"exact_implicit_motor_inclusion"}

    def prepare(self, output_angle, output_velocity, input_effort, dt, *, predictor_inertia=None, predictor_load=None):
        """Return an implicit output drive for a predicted friction branch.

        In a moving branch, B*v + a*tau = u - sign(v)*T0, where
        a=1+sign(v)*c*sign(tau). Eliminating v gives one positive PD law
        that PhysX solves simultaneously with its other constraints.
        A post-step branch/power review is mandatory; it cannot be assumed
        that a branch predicted before a contact transition remains valid.
        """
        p = self.reference
        q,w,u,h=map(float,(output_angle,output_velocity,input_effort,dt))
        if not all(map(math.isfinite,(q,w,u,h))) or h<=0:
            raise ValueError("Finite state, effort and positive timestep required")
        if abs(u)>p.input_effort_boundary+1e-10:
            raise ValueError("Motor input effort exceeds the declared finite reference")
        if self.integration=="passive_split":
            return self._prepare_split(q,w,u,h)
        K,D,B,c=p.transmission_stiffness,p.transmission_damping,p.input_viscosity,p.load_friction_ratio
        A=K*h*(.5 if self.integration=="midpoint_native" else 1.)+D
        if B<=(c-1)*A:
            raise ValueError("Input regularization does not ensure a monotone implicit branch")
        tau0=K*(self.input_angle-q)-D*w
        predicted_branch=None
        predicted_torque=None
        if predictor_inertia is not None:
            inertia=float(predictor_inertia);load=float(predictor_load)
            if not math.isfinite(inertia) or inertia<=0 or not math.isfinite(load):
                raise ValueError("A finite positive robot inertia and finite measured-load prediction are required")
            inertial_slope=inertia/h*(2. if self.integration=="midpoint_native" else 1.)
            C=A*inertial_slope/(inertial_slope+A)
            sb=(K*(self.input_angle-q)-A*w-A/inertial_slope*load)*inertial_slope/(inertial_slope+A)
            magnitude=lambda t: math.hypot(t,p.load_zero_smoothing)
            if abs(u-sb)<=p.no_load_friction_effort+c*magnitude(sb)+1e-12:
                predicted_branch=(0,0)
            elif p.load_zero_smoothing>0:
                sv=1 if u-sb>0 else -1
                def residual(vv):
                    tt=sb+C*vv
                    return B*vv+tt+sv*(p.no_load_friction_effort+c*magnitude(tt))-u
                radius=(abs(u)+abs(sb)+p.no_load_friction_effort+c*magnitude(sb))/(B-(c-1)*C)+1.
                left,right=-radius,radius
                for _ in range(48):
                    middle=(left+right)/2
                    if residual(middle)>0:right=middle
                    else:left=middle
                predicted_torque=sb+C*(left+right)/2
                predicted_branch=(sv,0)
            else:
                candidates=[]
                for sv in (-1,1):
                    for st in (-1,1):
                        factor=1+sv*c*st
                        vv=(u-factor*sb-sv*p.no_load_friction_effort)/(B+factor*C)
                        tt=sb+C*vv
                        if sv*vv>1e-12 and st*tt>=-1e-12:
                            if not any(abs(vv-old[2])<1e-10 for old in candidates):candidates.append((sv,st,vv))
                if len(candidates)!=1:
                    raise ValueError("The robot-side predictor did not select one physical friction branch")
                predicted_branch=candidates[0][:2]
                predicted_torque=sb+C*candidates[0][2]
        cap=p.no_load_friction_effort+c*math.hypot(tau0,p.load_zero_smoothing)
        if (predicted_branch==(0,0)) or (predicted_branch is None and abs(u-tau0)<=cap+1e-12):
            sign_v=0;sign_tau=0;a=None
            friction_constant=0.
            kp,kd,target=K,D,self.input_angle
        else:
            sign_v=predicted_branch[0] if predicted_branch is not None else (1 if u-tau0>0 else -1)
            # tau0 is the hypothetical torque if the input stopped. During
            # motion it can have the opposite sign from the current load.
            # Use the actual previous input velocity to predict load sign.
            moving_torque=K*(self.input_angle-q)+D*(self.input_velocity-w)
            if self.integration=="midpoint_native" and self.last_observed_effort is not None:
                moving_torque=self.last_observed_effort
            sign_tau=predicted_branch[1] if predicted_branch is not None else (1 if (moving_torque if abs(moving_torque)>1e-10 else u)>0 else -1)
            if self.friction_step=="frozen_magnitude":
                # Freeze a nonnegative friction magnitude, not its tangent.
                # A tangent to |tau| (or its smooth approximation) can become
                # negative when the actual load crosses zero within this step.
                # Retain the true signed motion test and energy review below;
                # neither force nor dissipated work is clipped after the step.
                hint=predicted_torque if predicted_torque is not None else moving_torque
                a=1.
                friction_constant=sign_v*(p.no_load_friction_effort+c*math.hypot(hint,p.load_zero_smoothing))
            elif p.load_zero_smoothing>0:
                hint=predicted_torque if predicted_torque is not None else moving_torque
                magnitude=math.hypot(hint,p.load_zero_smoothing)
                slope=hint/magnitude
                a=1+sign_v*c*slope
                friction_constant=sign_v*(p.no_load_friction_effort+c*(magnitude-slope*hint))
            else:
                a=1+sign_v*c*sign_tau
                friction_constant=sign_v*p.no_load_friction_effort
            denominator=B+a*A
            kp,kd=K*B/denominator,D*B/denominator
            target=self.input_angle+A*(u-friction_constant)/(B*K)
        self.pending={"q":q,"w":w,"z":self.input_angle,"u":u,"h":h,"a":a,
                      "velocity_sign":sign_v,"torque_sign":sign_tau,"kp":kp,"kd":kd,"target":target,
                      "friction_constant":friction_constant}
        return {"stiffness":kp,"damping":kd,"position_target":target,"velocity_target":0.,
                "max_effort":p.transmission_effort_boundary,"branch":"stick" if sign_v==0 else "move"}

    def complete(self, output_angle, output_velocity, *, observed_drive_effort=None,native_capped_drive=False):
        if self.pending is None:
            raise RuntimeError("prepare must precede the physical step")
        if not all(map(math.isfinite,(output_angle,output_velocity))):
            raise ValueError('Finite measured output state required')
        if self.integration=="passive_split":
            return self._complete_split(float(output_angle),float(output_velocity),observed_drive_effort,
                native_capped_drive=native_capped_drive)
        if native_capped_drive:raise ValueError('Native capped-drive estimate is only used by passive-split integration')
        s=self.pending;self.pending=None;p=self.reference
        q,w=map(float,(output_angle,output_velocity))
        tau=s["kp"]*(s["target"]-q)-s["kd"]*w
        endpoint_formula=tau
        if self.integration=="midpoint_native":
            if observed_drive_effort is None:
                raise ValueError("Midpoint port update requires the reviewed native drive impulse")
            tau=float(observed_drive_effort)
        saturated=abs(tau)>p.transmission_effort_boundary+1e-7
        if s["velocity_sign"]==0:
            v=0.;friction=s["u"]-tau
            consistent=abs(friction)<=p.no_load_friction_effort+p.load_friction_ratio*math.hypot(tau,p.load_zero_smoothing)+1e-7
        else:
            v=(s["u"]-s["friction_constant"]-s["a"]*tau)/p.input_viscosity
            friction=s["u"]-tau-p.input_viscosity*v
            consistent=(s["velocity_sign"]*v>=-1e-7 and
                        (self.friction_step=="frozen_magnitude" or p.load_zero_smoothing>0 or s["torque_sign"]*tau>=-1e-7))
        self.input_angle=s["z"]+s["h"]*v
        self.input_velocity=v
        self.last_observed_effort=observed_drive_effort
        self.previous_output_velocity=s["w"]
        used_w=(q-s["q"])/s["h"] if self.integration=="midpoint_native" else w
        old_extension=s["z"]-s["q"];extension=self.input_angle-q
        used_extension=(old_extension+extension)/2 if self.integration=="midpoint_native" else extension
        reconstructed=p.transmission_stiffness*used_extension+p.transmission_damping*(v-used_w)
        motor_work=s["h"]*s["u"]*v
        joint_work=s["h"]*tau*used_w
        damping_heat=s["h"]*p.transmission_damping*(v-used_w)**2
        input_heat=s["h"]*p.input_viscosity*v*v
        friction_heat=s["h"]*friction*v
        spring_energy_change=.5*p.transmission_stiffness*(extension**2-old_extension**2)
        numerical_loss=0. if self.integration=="midpoint_native" else .5*p.transmission_stiffness*(extension-old_extension)**2
        energy_residual=(spring_energy_change+joint_work+damping_heat+input_heat+friction_heat+numerical_loss-motor_work)
        return {"input_angle":self.input_angle,"input_velocity":v,"output_angle":q,"output_velocity":w,
                "input_effort":s["u"],"transmission_effort":tau,"friction_effort":friction,
                "predicted_branch_consistent":bool(consistent),"drive_saturation":bool(saturated),
                "spring_law_residual_nm":reconstructed-tau,
                "motor_work_j":motor_work,"joint_work_j":joint_work,"spring_energy_change_j":spring_energy_change,
                "friction_heat_j":friction_heat,"damping_heat_j":damping_heat,"input_viscous_heat_j":input_heat,
                "backward_euler_numerical_loss_j":numerical_loss,"port_energy_residual_j":energy_residual,
                "output_integration_residual_rad":q-s["q"]-s["h"]*w,
                "observed_drive_effort_nm":observed_drive_effort,
                "moving_friction_law_error_nm":0. if s["velocity_sign"]==0 else friction-s["velocity_sign"]*(p.no_load_friction_effort+p.load_friction_ratio*math.hypot(tau,p.load_zero_smoothing)),
                "drive_formula_minus_observed_nm":None if observed_drive_effort is None else endpoint_formula-float(observed_drive_effort),
                "integration":self.integration,"friction_step":self.friction_step}
def finger_output_posture_targets(mechanism, output_positions, reference, output_goal, dt,
                                  *, allow_closing=False):
    """Finite motor references for an observed physical finger opening."""
    import numpy as np
    positions=np.asarray(output_positions,dtype=float)
    target=np.asarray(reference,dtype=float).copy();goal=np.asarray(output_goal,dtype=float)
    if positions.shape!=(3,) or target.shape!=(4,) or goal.shape!=(4,):
        raise ValueError('Three observed finger outputs and four hand references required')
    rows=[]
    for i,name in enumerate(('f1j2','f2j1','f3j2'),start=1):
        error=float(goal[i]-positions[i-1])
        speed=0. if abs(error)<.0002 else float(np.clip(4.*error,-.15,.15 if allow_closing else 0.))
        low,high=mechanism.setup['intervals'][name]
        target[i],row=position_reference_for_input_velocity(mechanism.drives[name],positions[i-1],speed,dt,
            mechanism.settings['motor_position_kp'],mechanism.settings['motor_position_kd'],low,high,
            clip_to_feasible=True)
        rows.append(row)
    return target,rows
