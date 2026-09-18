import unittest
from types import SimpleNamespace
from perception_latency import DelayedObservation,execute_computation_delay


class PerceptionLatencyTests(unittest.TestCase):
    def test_planning_result_waits_for_controlled_physics_and_rounds_delay_up(self):
        world=SimpleNamespace(current_time=1.,playing=False)
        world.play=lambda:setattr(world,'playing',True)
        stepper=SimpleNamespace(step_index=20)
        calls=[]
        def protected_hold():
            self.assertTrue(world.playing)
            calls.append(stepper.step_index)
            stepper.step_index+=1;world.current_time+=.01
        result=execute_computation_delay(world,stepper,.01,.025,protected_hold)
        self.assertEqual(calls,[20,21,22])
        self.assertEqual(result['intervening_physics_steps'],3)
        self.assertAlmostEqual(result['available_time_s'],1.025)
        self.assertGreaterEqual(result['consumed_time_s'],result['available_time_s'])

    def test_a_paused_callback_cannot_claim_a_physical_delay(self):
        world=SimpleNamespace(current_time=1.,play=lambda:None)
        with self.assertRaisesRegex(RuntimeError,'did not advance physical time'):
            execute_computation_delay(world,SimpleNamespace(step_index=20),.01,.025,lambda:None)

    def test_original_control_stop_propagates_out_of_the_wait(self):
        world=SimpleNamespace(current_time=1.,play=lambda:None)
        def stopped():raise RuntimeError('original force stop')
        with self.assertRaisesRegex(RuntimeError,'original force stop'):
            execute_computation_delay(world,SimpleNamespace(step_index=20),.01,.025,stopped)

    def test_invalid_planning_delay_is_rejected(self):
        world=SimpleNamespace(current_time=1.,play=lambda:None)
        for delay in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                execute_computation_delay(world,SimpleNamespace(step_index=20),.01,delay,lambda:None)

    def test_computation_finished_in_wall_time_does_not_make_the_frame_available_early(self):
        result=DelayedObservation(2.,.05,.075,{'position':[1,2,3]})
        self.assertFalse(result.ready(2.))
        self.assertFalse(result.ready(2.124))
        self.assertTrue(result.ready(2.125))
        self.assertEqual(result.available_time_s,2.125)

    def test_bad_latency_cannot_become_an_instantaneous_measurement(self):
        for delay in (-.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):DelayedObservation(0.,0.,delay,{})


if __name__=='__main__':unittest.main()
