"""Short original-controller force-following check with declared bench geometry.

Only robot encoders and the wrist signal are online inputs. The known fixture
pose is a test initial condition, not a successful visual robot alignment.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.collision_roster import load_authoritative_collision_link_roster
from kcg_connector.grasp.carts_v2.models import _build_verified_robot_model
from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
from te_body_nut_rotation import interface_wrench_from_wrist


class LocalInterfaceFollowing:
    def __init__(self, repository, q0, nut_world, body_world, socket_world, dt, *, grip_recipe=None,
                 finger_mechanism_path=None):
        repo=Path(repository)
        hc=load_carts_hand_contract('src/kcg_connector/config/carts_hand_contact_v1.yaml',repository_root=repo)
        roster=load_authoritative_collision_link_roster('src/kcg_connector/config/carts_collision_roster_v1.yaml',repository_root=repo)
        self.model=_build_verified_robot_model(hc,roster,finger_mechanism_path=finger_mechanism_path)
        self.finger_mechanism_id='LEGACY_LINEAR_MIMIC'
        observer_options={}
        if finger_mechanism_path is not None:
            from kcg_connector.grasp.robust.finger_fourbar import load_finger_fourbars
            mechanism,_=load_finger_fourbars(finger_mechanism_path)
            self.finger_mechanism_id=mechanism['mechanism_id']
            if not grip_recipe or not grip_recipe.get('source_geometry_plan'):
                raise ValueError('Four-bar force following requires a matching source geometry plan')
            geometry_path=Path(grip_recipe['source_geometry_plan'])
            if not geometry_path.is_absolute():geometry_path=repo/geometry_path
            geometry=json.loads(geometry_path.read_text())
            if geometry.get('finger_mechanism_id')!=self.finger_mechanism_id:
                raise ValueError('Force-following grasp geometry belongs to a different finger mechanism')
            observer_options['sensor_semantics']='BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O'
        reference=repo/'src/kcg_connector/config/visual_assembly_v1_rotation_reference.json'
        self.settings=json.loads(reference.read_text())['settings'];self.source=str(reference);self.dt=dt
        if grip_recipe and 'maximum_arm_speed_rad_s' in grip_recipe:
            speed=float(grip_recipe['maximum_arm_speed_rad_s'])
            source_limit=min(self.model.joints[f'iiwa_joint_{i}'].limit.velocity for i in range(1,8))
            if not 0<speed<=source_limit:raise ValueError('Local comparison arm speed exceeds a source URDF joint velocity limit')
            self.settings['maximum_arm_speed_rad_s']=speed
        if grip_recipe and 'maximum_axial_speed_m_s' in grip_recipe:
            speed=float(grip_recipe['maximum_axial_speed_m_s'])
            if not 0<speed<=.005:raise ValueError('Local axial following speed must be finite and at most5mm/s')
            self.settings['maximum_axial_speed_m_s']=speed
        if grip_recipe and 'maximum_axial_travel_m' in grip_recipe:
            travel=float(grip_recipe['maximum_axial_travel_m'])
            if not 0<travel<=.015:raise ValueError('Axial travel must remain inside the declared assembly envelope')
            self.settings['maximum_axial_travel_m']=travel
        if grip_recipe and 'maximum_planar_offset_m' in grip_recipe:
            offset=float(grip_recipe['maximum_planar_offset_m'])
            speed=float(grip_recipe['maximum_planar_speed_m_s'])
            if not 0<offset<=.003 or not 0<speed<=.002:raise ValueError('Planar following must retain a finite local travel/speed range')
            self.settings['planar_force_admittance']['maximum_offset_m']=offset
            self.settings['planar_force_admittance']['maximum_speed_m_s']=speed
        q0=np.asarray(q0);H=self.model.forward_kinematics(q0,enforce_limits=False)['handbase_link']
        self.pivot0=np.asarray(nut_world);self.hand_from_pivot=np.linalg.inv(H)@self.pivot0
        self.com_hand=(np.linalg.inv(H)@body_world@np.r_[self.settings['payload_com_body_m'],1.])[:3]
        self.axes=np.asarray(socket_world)[:3,:3];self.axis=self.axes[:,2]
        self.arm=q0[:7].copy();self.offset=np.zeros(3);self.bias_samples=[];self.bias=None;self.filtered=None
        self.angular_follow=np.eye(3)
        # Reuse the source URDF inertial reader. Force-estimation geometry is
        # not used to choose the hand trajectory or load targets here.
        inertial_geometry=(grip_recipe['source_geometry_plan'] if finger_mechanism_path is not None else
            repo/'artifacts/kcg_connector/robot_assembly_20260911/source_surface_plan_01/geometry_plan.json')
        inertial_reader=ThreeFingerWrenchObserver(repo,self.model,inertial_geometry,**observer_options)
        self.inertials=inertial_reader.inertials
        self.lower=np.array([self.model.joints[f'iiwa_joint_{i}'].limit.lower for i in range(1,8)])
        self.upper=np.array([self.model.joints[f'iiwa_joint_{i}'].limit.upper for i in range(1,8)])
        self.records=[];self.grip_recipe=grip_recipe;self.grip_tare_q=[];self.grip_tare_reaction=[]
        self.position_error_gain=float(grip_recipe.get('pivot_position_error_gain_s_inv',3.)) if grip_recipe else 3.
        if not 0<self.position_error_gain<=30.:raise ValueError('Pivot position feedback gain must be finite and at most30/s')
        self.entry_axial_push=float(grip_recipe.get('entry_axial_push_n',0.)) if grip_recipe else 0.
        if not 0<=self.entry_axial_push<=3.0400615:raise ValueError('Entry push must remain within the demonstrated3.04N bench input')
        self.hand_position_stiffness=float(grip_recipe.get('hand_position_stiffness_nm_rad',12.)) if grip_recipe else 12.
        self.turn_start=float(grip_recipe.get('preindex_start_s',3.0)) if grip_recipe else 1.6
        self.side_start=float(grip_recipe.get('sidewall_start_s',4.0)) if grip_recipe else 4.
        self.side_ramp=float(grip_recipe.get('sidewall_ramp_s',1.5)) if grip_recipe else 1.5
        self.side_sequence=bool(grip_recipe and grip_recipe.get('sidewall_sequence'))
        self.side_grip=None
        if grip_recipe:
            self.grip_observer=ThreeFingerWrenchObserver(repo,self.model,grip_recipe['source_geometry_plan'],**observer_options)
            self.normal_target=np.asarray(grip_recipe['source_normal_targets_n'],float)
            self.effort_levers=np.asarray(grip_recipe['coupled_normal_effort_levers_m'],float)
            geometry_path=Path(grip_recipe['source_geometry_plan'])
            if not geometry_path.is_absolute():geometry_path=repo/geometry_path
            geometry=json.loads(geometry_path.read_text())
            self.contact_q=np.asarray(geometry['first_contact_hand_positions_rad'])
            self.hand_goal=self.contact_q.copy();self.hand_goal[1:]+=.002
            self.filtered_normal=None;self.last_grip=None

    def update(self, q, raw_wrist, elapsed, angle_rad, angular_rate, *, projected_reactions=None,
               grip_elapsed=None, grip_enabled=True, force_follow_enabled=True,
               arm_speed_override=None, entry_assist_enabled=True):
        q=np.asarray(q);fk=self.model.forward_kinematics(q,enforce_limits=False);H=fk['handbase_link'];R=H[:3,:3]
        gravity=np.zeros(6)
        for name,(mass,com) in self.inertials.items():
            T=fk[name];f=np.array([0.,0.,-9.81*mass]);gravity+=np.r_[f,np.cross(T[:3,:3]@com+T[:3,3]-H[:3,3],f)]
        canonical=-np.r_[R@raw_wrist[:3],R@raw_wrist[3:]]
        if .05 <= elapsed < .18:
            self.bias_samples.append(np.r_[R.T@(canonical-gravity)[:3],R.T@(canonical-gravity)[3:]])
            if self.grip_recipe:
                self.grip_tare_q.append(q.copy());self.grip_tare_reaction.append(np.asarray(projected_reactions).copy())
        if elapsed < 1.0:return None
        if self.bias is None:
            if len(self.bias_samples)<20:raise RuntimeError('Missing declared open-hand wrist zero')
            self.bias=np.mean(self.bias_samples,axis=0)
            if self.grip_recipe:self.grip_observer.calibrate_free_space(self.grip_tare_q,self.grip_tare_reaction)
        measured=canonical-gravity-np.r_[R@self.bias[:3],R@self.bias[3:]]
        grip_time=elapsed if grip_elapsed is None else grip_elapsed
        if grip_enabled and self.grip_recipe and self.grip_recipe.get('root_moment_control'):
            joint_gravity=self.grip_observer._system(q,fk=fk)[2]
            reaction=(np.asarray(projected_reactions)-self.grip_observer.tare_reaction
                -(joint_gravity-self.grip_observer.tare_gravity))
            u=np.clip((grip_time-1.)/1.5,0.,1.);u=10*u**3-15*u**4+6*u**5
            desired=u*np.asarray(self.grip_recipe['root_moment_targets_nm'])
            change=self.dt/(1/6+self.dt)*(desired-reaction)/self.hand_position_stiffness
            hold_positions=bool(self.grip_recipe.get('hold_grip_positions_during_turn') and grip_time>=self.turn_start)
            if not hold_positions:
                self.hand_goal[1:]+=np.clip(change,-.15*self.dt,.15*self.dt)
            self.hand_goal[1:]=np.clip(self.hand_goal[1:],self.contact_q[1:]-.04,self.contact_q[1:]+.24)
            self.last_grip={'control_quantity':('BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O' if self.model.fourbar_couplings else
                                               'DIRECT_PROXIMAL_JOINT_MOMENT_NOT_FINGERTIP_FORCE'),
                'measured_load_moment_nm':reaction.tolist(),'target_load_moment_nm':desired.tolist(),
                'hand_goal_rad':self.hand_goal.tolist(),'legacy_normal15N_reference_is_not_hardware_capacity':True,
                'finite_effort_position_hold_during_turn':hold_positions}
        if grip_enabled and self.grip_recipe and not self.grip_recipe.get('root_moment_control'):
            estimate=self.grip_observer.estimate(q,projected_reactions,measured,fk=fk)
            normal=estimate['source_surface_normal_force_n']
            alpha_n=self.dt/(.05+self.dt)
            self.filtered_normal=normal.copy() if self.filtered_normal is None else self.filtered_normal+alpha_n*(normal-self.filtered_normal)
            fraction=np.clip((elapsed-1.)/1.5,0.,1.);fraction=10*fraction**3-15*fraction**4+6*fraction**5
            desired=fraction*self.normal_target
            if self.grip_recipe.get('resultant_force_control'):
                force_norm=np.linalg.norm(estimate['force_world_n'],axis=1)
                self.filtered_force_norm=(force_norm.copy() if not hasattr(self,'filtered_force_norm') else
                    self.filtered_force_norm+alpha_n*(force_norm-self.filtered_force_norm))
                force_gains=[]
                for i,name in enumerate(self.grip_observer.links):
                    direction=(estimate['force_world_n'][i]/force_norm[i] if force_norm[i]>.5 else
                               estimate['source_surface_normals_world'][i])
                    Ji=self.model.geometric_jacobian(name,tuple(q),point_local_m=self.grip_observer.points_local[name],enforce_limits=False)
                    force_gains.append(-float(direction@Ji[:3,8+i]))
                change=self.dt/(1/6+self.dt)*np.asarray(force_gains)*(desired-self.filtered_force_norm)/self.hand_position_stiffness
            else:
                change=self.dt/(1/6+self.dt)*self.effort_levers*(desired-self.filtered_normal)/self.hand_position_stiffness
            self.hand_goal[1:]+=np.clip(change,-.15*self.dt,.15*self.dt)
            self.hand_goal[1:]=np.clip(self.hand_goal[1:],self.contact_q[1:]-.04,self.contact_q[1:]+.24)
            self.last_grip={'source_normal_n':normal.tolist(),'filtered_source_normal_n':self.filtered_normal.tolist(),
                'radial_resultant_n':estimate['radial_resultant_force_n'].tolist(),'reference_source_normal_n':desired.tolist(),
                'condition':estimate['normalized_condition'],'hand_goal_rad':self.hand_goal.tolist()}
            if self.grip_recipe.get('resultant_force_control'):
                self.last_grip.update(resultant_force_n=force_norm.tolist(),
                    filtered_resultant_force_n=self.filtered_force_norm.tolist(),
                    commanded_force_quantity='RESULTANT_NOT_LOCAL_NORMAL_SUM')
            if self.side_sequence and elapsed>=self.side_start:
                B=H@self.grip_observer.hand_from_body;side_normal=B[:3,:3]@np.array([1.,0.,0.])
                side_force=float(estimate['force_world_n'][2]@side_normal)
                blend=np.clip((elapsed-self.side_start)/self.side_ramp,0.,1.);blend=10*blend**3-15*blend**4+6*blend**5
                reference=self.normal_target+blend*(np.asarray(self.grip_recipe['sidewall_targets_n'])-self.normal_target)
                # Undo the ordinary positive-closing update for this step;
                # the useful sidewall force grows when finger3 opens.
                self.hand_goal[1:]-=np.clip(change,-.15*self.dt,.15*self.dt)
                if self.grip_recipe.get('neutralize_third_preload') and self.side_grip is None:
                    self.hand_goal[3]=float(q[10])
                J=self.model.geometric_jacobian('f3Link3',tuple(q),point_local_m=self.grip_observer.points_local['f3Link3'],enforce_limits=False)
                signed_lever=-float(J[:3,10]@side_normal)
                self.filtered_side=(side_force if not hasattr(self,'filtered_side') else
                    self.filtered_side+alpha_n*(side_force-self.filtered_side))
                if self.grip_recipe.get('sidewall_measured_balance'):
                    balanced=np.asarray(self.grip_recipe['sidewall_other_finger_ratios'])*max(.5,self.filtered_side)
                    balance_fraction=1. if self.grip_recipe.get('immediate_measured_balance') else blend
                    reference[:2]=(1.-balance_fraction)*self.normal_target[:2]+balance_fraction*np.maximum(balanced,.5)
                errors=reference-self.filtered_normal;errors[2]=reference[2]-self.filtered_side
                gains=self.effort_levers.copy();gains[2]=signed_lever
                correction=self.dt/(1/6+self.dt)*gains*errors/self.hand_position_stiffness
                correction[2]=np.clip(correction[2],-.012*self.dt,.012*self.dt)
                if self.grip_recipe.get('coupled_contact_forces'):
                    tangent=B[:3,:3]@np.array([0.,1.,0.])
                    tangent_force=float(estimate['force_world_n'][2]@tangent)
                    self.filtered_tangent=(tangent_force if not hasattr(self,'filtered_tangent') else
                        self.filtered_tangent+alpha_n*(tangent_force-self.filtered_tangent))
                    force_error=np.r_[reference[:2]-self.filtered_normal[:2],reference[2]-self.filtered_side,
                        self.grip_recipe['sidewall_tangent_ratio']*reference[2]-self.filtered_tangent]
                    contact_J=[]
                    for i,name in enumerate(self.grip_observer.links[:2]):
                        Ji=self.model.geometric_jacobian(name,tuple(q),point_local_m=self.grip_observer.points_local[name],enforce_limits=False)
                        ni=estimate['source_surface_normals_world'][i]
                        contact_J.append(ni@Ji[:3,7:11])
                    contact_J.extend([side_normal@J[:3,7:11],tangent@J[:3,7:11]])
                    contact_J=np.asarray(contact_J)
                    delta=-self.dt/(1/6+self.dt)*(contact_J.T@force_error)/self.hand_position_stiffness
                    self.hand_goal+=np.clip(delta,-.15*self.dt,.15*self.dt)
                    if abs(self.hand_goal[0]-self.contact_q[0])>np.deg2rad(5.):
                        raise RuntimeError('Bounded five-degree palm-layout correction exhausted')
                else:
                    self.hand_goal[1:]+=np.clip(correction,-.15*self.dt,.15*self.dt)
                    if self.filtered_normal[2]>5. and self.filtered_side>2.:
                        self.hand_goal[3]-=.003*self.dt
                self.hand_goal[1:]=np.clip(self.hand_goal[1:],self.contact_q[1:]-.04,self.contact_q[1:]+.24)
                self.side_grip={'side_force_estimate_n':side_force,'filtered_side_force_n':self.filtered_side,
                    'flat_projection_n':float(self.filtered_normal[2]),'signed_control_lever_m':signed_lever,
                    'force_targets_n':reference.tolist(),'third_actuator_cap_nm':
                        float(self.grip_recipe.get('finite_motor_cap_nm',2.7))}
                if self.grip_recipe.get('coupled_contact_forces'):
                    self.side_grip.update(tangent_force_estimate_n=tangent_force,
                        filtered_tangent_force_n=self.filtered_tangent,
                        tangent_force_reference_n=self.grip_recipe['sidewall_tangent_ratio']*reference[2],
                        contact_control_jacobian=contact_J.tolist(),palm_layout_target_deg=float(np.degrees(self.hand_goal[0])))
                self.last_grip.update(sidewall_mode=self.side_grip,hand_goal_rad=self.hand_goal.tolist())
                if elapsed>self.side_start+self.side_ramp and self.filtered_side<-.5:
                    raise RuntimeError('Sidewall contact force estimate outside its bounded range')
            # A released flat face can have a negative force projection while
            # the active sidewall has a positive load. Its own check is above.
            positive_reference=self.filtered_normal[:2] if self.side_sequence and elapsed>=self.side_start else self.filtered_normal
            if (not np.isfinite(self.filtered_normal).all() or estimate['normalized_condition']>150.
                    or (elapsed>2.5 and np.min(positive_reference)<-.5)):
                raise RuntimeError('Source-CAD force estimator observation stop; 15N normal reference is not calibrated hardware capacity')
        pivot=H@self.hand_from_pivot;com=R@self.com_hand+H[:3,3]
        fixed=interface_wrench_from_wrist(measured,H[:3,3],self.pivot0[:3,3],com,
            self.settings['payload_mass_kg'],9.81,self.axes)
        alpha=self.dt/(self.settings['contact_estimate_filter_time_constant_s']+self.dt)
        self.filtered=fixed.copy() if self.filtered is None else self.filtered+alpha*(fixed-self.filtered)
        wrench=self.filtered.copy()
        omega_follow=np.zeros(3)
        if force_follow_enabled and self.grip_recipe and self.grip_recipe.get('angular_admittance_rad_per_nm_s') and elapsed>=2.5:
            offset_local=self.axes.T@Rotation.from_matrix(self.angular_follow).as_rotvec()
            restoring=float(self.grip_recipe.get('angular_stiffness_nm_rad',0.))*offset_local[:2]
            omega_follow=self.axes@np.r_[wrench[3:5]-restoring,0.]*float(self.grip_recipe['angular_admittance_rad_per_nm_s'])
            max_rate=np.deg2rad(float(self.grip_recipe.get('maximum_angular_follow_speed_deg_s',2.)))
            omega_follow*=min(1.,max_rate/max(np.linalg.norm(omega_follow),1e-15))
            previous=self.angular_follow.copy()
            self.angular_follow=Rotation.from_rotvec(omega_follow*self.dt).as_matrix()@previous
            follow_vector=Rotation.from_matrix(self.angular_follow).as_rotvec()
            max_offset=np.deg2rad(float(self.grip_recipe.get('maximum_angular_follow_offset_deg',.75)))
            if np.linalg.norm(follow_vector)>max_offset:
                if not self.grip_recipe.get('bounded_angular_compliance'):
                    raise RuntimeError('Declared angular-following travel boundary')
                self.angular_follow=Rotation.from_rotvec(follow_vector*max_offset/np.linalg.norm(follow_vector)).as_matrix()
                omega_follow=Rotation.from_matrix(self.angular_follow@previous.T).as_rotvec()/self.dt
        wrench[3:]+=self.axes.T@np.cross(self.pivot0[:3,3]-pivot[:3,3],self.axes@wrench[:3])
        u=np.clip((elapsed-self.turn_start)/self.settings['turn_force_reference_ramp_duration_s'],0.,1.)
        g=10*u**3-15*u**4+6*u**5
        ref=self.settings['axial_force_reference_n']+g*(self.settings['turn_axial_force_reference_n']-self.settings['axial_force_reference_n'])
        # Successful isolated mating includes this finite axial help before
        # thread capture. Here it is produced by robot wrist-force control;
        # no force is applied directly to a connector actor.
        entry_ramp=np.clip((elapsed-self.turn_start)/.2,0.,1.)
        entry_ramp=10*entry_ramp**3-15*entry_ramp**4+6*entry_ramp**5
        entry_fade=np.clip((np.degrees(angle_rad)-40.)/10.,0.,1.)
        entry_fade=10*entry_fade**3-15*entry_fade**4+6*entry_fade**5
        entry_assist=self.entry_axial_push*entry_ramp*(1.-entry_fade) if entry_assist_enabled else 0.
        ref+=entry_assist
        vz=float(np.clip((wrench[2]-ref)*self.settings['axial_admittance_m_per_n_s'],
            -self.settings['maximum_axial_speed_m_s'],self.settings['maximum_axial_speed_m_s']))
        planar=self.settings['planar_force_admittance'];vxy=planar['admittance_m_per_n_s']*wrench[:2]
        if self.grip_recipe and self.grip_recipe.get('planar_compliance_m_per_n'):
            wanted=float(self.grip_recipe['planar_compliance_m_per_n'])*wrench[:2]
            wanted*=min(1.,planar['maximum_offset_m']/max(np.linalg.norm(wanted),1e-15))
            vxy=(wanted-self.offset[:2])/float(self.grip_recipe.get('planar_compliance_time_constant_s',.1))
        vxy*=min(1.,planar['maximum_speed_m_s']/max(np.linalg.norm(vxy),1e-15))
        # The original sequence completes the grip before enabling arm force
        # following. During a force ramp, hold the planned palm reference.
        if self.grip_recipe and elapsed<self.turn_start:
            vz=0.;vxy=np.zeros(2)
        if self.side_sequence or (self.grip_recipe and self.grip_recipe.get('planar_hold')):
            # Retain the planned lateral grasp while the finger changes faces.
            # The existing force limit replaces the unused zero-force travel
            # objective; axial following remains active during rotation.
            vxy=np.zeros(2)
        if not force_follow_enabled:
            # Open-hand reindexing retains the achieved axial/lateral pose.
            # The absent payload must not drive the insertion admittance.
            vz=0.;vxy=np.zeros(2)
        if self.grip_recipe:
            limits={'force_n':float(self.grip_recipe.get('planned_wrist_force_limit_n',3.0400615)),
                    'bending_nm':float(self.grip_recipe.get('planned_wrist_bending_observation_nm',.2)),
                    'twisting_nm':float(self.grip_recipe.get('planned_wrist_twisting_stop_nm',1.))}
            if not 0<limits['twisting_nm']<=4.6:raise ValueError('Coupling torque observation cannot exceed the public4.6Nm envelope')
            if not 0<limits['bending_nm']<=.4:raise ValueError('Local bending observation must be finite and at most0.4Nm')
            measured_limits={'force_n':float(np.linalg.norm(wrench[:3])),
                             'bending_nm':float(np.linalg.norm(wrench[3:5])),'twisting_nm':float(abs(wrench[5]))}
            record_only=set(self.grip_recipe.get('wrist_record_only_quantities',[]))
            if record_only-{'force_n','bending_nm'}:raise ValueError('Only uncalibrated force/bending observations may be record-only')
            self.wrist_observation_maxima={k:max(getattr(self,'wrist_observation_maxima',{}).get(k,0.),measured_limits[k]) for k in limits}
            exceeded=[k for k in limits if measured_limits[k]>limits[k] and k not in record_only]
            if exceeded:
                self.stop_sample={'time_s':elapsed,'wrench':wrench.tolist(),'observed':measured_limits,'limits':limits,'exceeded':exceeded}
                raise RuntimeError('WRIST_OBSERVATION_BOUND: '+','.join(exceeded))
        self.offset+=np.r_[vxy,vz]*self.dt
        if np.linalg.norm(self.offset[:2])>planar['maximum_offset_m'] or abs(self.offset[2])>self.settings['maximum_axial_travel_m']:
            raise RuntimeError('Original force-following travel boundary')
        target=self.pivot0.copy();target[:3,3]+=self.axes@self.offset
        target[:3,:3]=self.angular_follow@Rotation.from_rotvec(-self.axis*angle_rad).as_matrix()@self.pivot0[:3,:3]
        J=np.asarray(self.model.geometric_jacobian('handbase_link',tuple(q)))[:,:7]
        radius=pivot[:3,3]-H[:3,3];x,y,z=radius;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        P=np.vstack([J[:3]-skew@J[3:],.05*J[3:]])
        error=np.r_[target[:3,3]-pivot[:3,3],.05*Rotation.from_matrix(target[:3,:3]@pivot[:3,:3].T).as_rotvec()]
        twist=np.r_[self.axes@np.r_[vxy,vz],.05*(omega_follow-self.angular_follow@self.axis*angular_rate)]
        feedback=3.*error;feedback[:3]=self.position_error_gain*error[:3]
        requested_twist=twist+feedback
        speed=self.settings['maximum_arm_speed_rad_s']
        if arm_speed_override is not None:
            if not 0<arm_speed_override<=1.6:raise ValueError('Open-hand indexing speed reference exceeded')
            speed=float(arm_speed_override)
        velocity_tracking_residual=None
        if self.grip_recipe and self.grip_recipe.get('joint_limit_aware_velocity'):
            # Joint-space velocity bounds leave room to brake before reaching
            # a joint stop. Solve the task with the remaining arm redundancy
            # rather than clipping each component of an unconstrained result.
            low=np.maximum(-speed,2.*(self.lower+.005-self.arm))
            high=np.minimum(speed,2.*(self.upper-.005-self.arm))
            if np.any(low>=high):raise RuntimeError('Original arm joint velocity feasibility boundary')
            A=np.vstack([P,.0005*np.eye(7)])
            solved=lsq_linear(A,np.r_[requested_twist,np.zeros(7)],bounds=(low,high),
                              method='bvls',tol=1e-9,max_iter=30)
            if not solved.success:raise RuntimeError('Joint-limit-aware velocity solve did not converge')
            dq=solved.x;velocity_tracking_residual=P@dq-requested_twist
        else:
            dq=P.T@np.linalg.solve(P@P.T+.0005**2*np.eye(6),requested_twist)
            dq=np.clip(dq,-speed,speed)
        self.arm+=dq*self.dt
        if np.any(self.arm<self.lower) or np.any(self.arm>self.upper):raise RuntimeError('Original arm joint limit')
        self.records.append({'time_s':elapsed,'interface_wrench':wrench.tolist(),'force_reference_n':float(ref),
            'entry_axial_assist_reference_n':float(entry_assist),
            'offset_socket_m':self.offset.tolist(),'commanded_angle_deg':float(np.degrees(angle_rad)),
            'angular_follow_offset_deg':Rotation.from_matrix(self.angular_follow).as_rotvec(degrees=True).tolist(),
            'encoder_pivot_world_m':pivot[:3,3].tolist(),'nominal_arm_target_rad':self.arm.tolist(),
            'pivot_position_tracking_error_m':error[:3].tolist(),
            'bounded_velocity_task_residual':None if velocity_tracking_residual is None else velocity_tracking_residual.tolist(),
            'grip_estimate':self.last_grip})
        return self.arm.copy()

    def report(self):
        return {'source_successful_controller':self.source,'restored_components':['payload-compensated wrist wrench',
            '50ms causal force filtering','axial force admittance and bounded lateral admittance','bounded Cartesian pivot tracking'],
            'scope':'DECLARED_LOCAL_FIXTURE_INITIAL_POSE_NOT_VISUAL_ALIGNMENT_OR_FULL_ROBOT_ASSEMBLY',
            'finger_mechanism_id':self.finger_mechanism_id,
            'mechanism_candidate':bool(self.model.fourbar_couplings),
            'finger_sensor_semantics':('BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O' if self.model.fourbar_couplings else
                                       'LEGACY_PROXIMAL_JOINT_REACTION'),
            'finger_self_lock_and_complete_assembly_verified':False if self.model.fourbar_couplings else None,
            'online_object_or_contact_truth_used':False,'source_pose_gains_and_motion_speed_limits_retained':True,
            'controller_period_s':self.dt,
            'pivot_position_error_gain_s_inv':self.position_error_gain,
            'effective_motion_settings':self.settings,
            'stop_sample':getattr(self,'stop_sample',None),
            'wrist_observation_maxima':getattr(self,'wrist_observation_maxima',None),
            'free_space_tare_sample_count':len(self.bias_samples),'source_grip_regulation_not_changed_by_this_mode':self.grip_recipe is None,
            'grip_recipe':self.grip_recipe,'source_normal_estimate_is_not_an_instantaneous_contact_sum_bound':True}
