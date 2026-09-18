"""Refresh palm feedback while retaining the original held-Body controller."""
import math
import numpy as np

def refresh(session,dynamic):
    from te_foundationpose_handoff_runtime import _execute_held_plug_path
    runtime=session.runtime;world=session.world;stepper=session.stepper
    ft=runtime['nail_body_ft_auditor'];dt=float(dynamic['physics_dt_s'])
    probe={'authorization':{'simulation_only':True,'hardware_authorized':False},
           'motion':{'maximum_transport_joint_speed_rad_s':.15}}
    def consume_pending():
        steps=max(1,math.ceil((session.pending.available_time_s-float(world.current_time))/dt)+1)
        arm=np.asarray(ft.samples[-1]['active_targets_rad'][:7])
        def states():
            for _ in range(steps):
                session.service(request_new=False)
                yield arm
        result=_execute_held_plug_path(world,stepper,ft,runtime['grasp_result'],dynamic,states(),probe,
                                      phase='key_probe_palm_feedback_wait')
        if not result['completed']:raise RuntimeError('Original protection stopped the palm feedback hold')
        session.service(request_new=False)
    if session.pending is not None:consume_pending()
    previous=session.consumed_count
    session.next_request=float(world.current_time)
    session.service()
    if session.pending is not None:consume_pending()
    if session.consumed_count<=previous:raise RuntimeError('A fresh palm observation was not consumed')


def hold(session,dynamic,seconds,*,phase):
    from te_foundationpose_handoff_runtime import _execute_held_plug_path
    runtime=session.runtime;ft=runtime['nail_body_ft_auditor'];dt=float(dynamic['physics_dt_s'])
    arm=np.asarray(ft.samples[-1]['active_targets_rad'][:7])
    probe={'authorization':{'simulation_only':True,'hardware_authorized':False},
           'motion':{'maximum_transport_joint_speed_rad_s':.15}}
    def states():
        for _ in range(max(1,math.ceil(seconds/dt))):
            session.service()
            yield arm
    result=_execute_held_plug_path(session.world,session.stepper,ft,runtime['grasp_result'],dynamic,states(),probe,phase=phase)
    if not result['completed']:raise RuntimeError('Original protection stopped the planning hold')
    return result
