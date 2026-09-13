"""Recovery preserves history, commands only the remainder, and rejects hard faults."""
from pathlib import Path
import json,sys,types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))


@pytest.mark.parametrize('hard_fault',[False,True])
def test_recovery_keeps_failed_attempt_and_does_not_restart_full_angle(monkeypatch,tmp_path,hard_fault):
    import te_body_nut_rotation as rotation
    import te_body_nut_reindex as reindex
    import te_body_nut_regrasp as regrasp
    from te_body_nut_continuation import run_nut_rotation_with_recovery
    (tmp_path/'control.yaml').write_text('nut_reindex: {}\n')
    ft=types.SimpleNamespace(samples=[{'phase':'key_probe_nut_grip_hold'}])
    stepper=types.SimpleNamespace(step_index=0,abort_reason=None)
    runtime={'body_assembly_control_config':'control.yaml','nail_body_ft_auditor':ft,
        'nut_regrasp_collision_context':(None,None)}
    angles=[];recovery_calls=[]
    def run(*args,**kwargs):
        settings=args[6];output=Path(args[7]);output.mkdir(parents=True)
        angle=settings['rotation_about_socket_plus_z_deg'];angles.append(angle)
        command=-30. if len(angles)==1 else angle
        path=output/'nut_rotation_control_samples.jsonl'
        path.write_text(json.dumps({'step':stepper.step_index,'elapsed_s':0.,'commanded_rotation_deg':command})+'\n')
        stepper.step_index+=1;ft.samples.append({'phase':'key_probe_nut_rotation_turn'})
        done=len(angles)>1
        return dict(completed=done,first_step=stepper.step_index-1,last_step=stepper.step_index,
            failure_reason=None if done else ('Finite hand transmission boundary: f3j2' if hard_fault else 'recoverable nut turn stop: LOADED_POSE_ERROR'),
            control_samples_file=str(path),outer_abort_reason=None,initial_arm_encoder_rad=[0.]*7,
            final_arm_target_rad=[0.]*7,final_hand_target_rad=[.2]*4)
    def release(*args,**kwargs):
        recovery_calls.append('release');stepper.step_index+=1;ft.samples.append({'phase':'nut_index_free_final_hold'})
        return {'completed':True,'final_observation':{'current_image':True}}
    def grip_again(*args,**kwargs):
        recovery_calls.append('grip');return {'completed':True,'new_current_grip':True}
    monkeypatch.setattr(rotation,'_run_body_nut_rotation_interval',run)
    monkeypatch.setattr(reindex,'run_nut_release_and_reindex',release)
    monkeypatch.setattr(regrasp,'run_body_nut_regrasp',grip_again)
    original={'completed':True,'original_grip':True}
    result=run_nut_rotation_with_recovery(tmp_path,runtime,stepper,{'physics_dt_s':1/960},original,np.eye(4),
        {'rotation_about_socket_plus_z_deg':-90.,'recovery':{'enabled':True,'maximum_regrasps':1}},tmp_path/'run')
    assert original=={'completed':True,'original_grip':True}
    assert result['attempts'][0]['rotation']['completed'] is False
    if hard_fault:
        assert angles==[-90.] and recovery_calls==[] and not result['completed']
    else:
        assert angles==[-90.,-60.] and recovery_calls==['release','grip']
        assert result['completed'] and result['executed_loaded_command_deg']==-90.
        assert result['continuation_grip']['new_current_grip']
        assert result['recovery_count']==1
