import unittest
import numpy as np
from te_nut_motion import bounded_release_hold_velocity


class ReleaseBrakingTests(unittest.TestCase):
    def test_recorded_early_return_speed_does_not_demand_excess_braking(self):
        q=np.zeros(7);v=np.array([.05640263,-.01171649,-.10958559,-.01320958,.05561836,-.00797224,.6320546865463257])
        g=np.array([0.,-43.31505,1.15407,21.96435,-.37459,-.03976,-.000063])
        ref,audit=bounded_release_hold_velocity(q,q,v,g,stiffness=2500.,damping=160.,effort_limit=100.)
        self.assertGreater(abs(audit['zero_velocity_request_effort_nm'][-1]),100.)
        self.assertLessEqual(max(abs(np.asarray(audit['predicted_drive_effort_nm']))),99.+1e-10)
        self.assertGreater(ref[-1],0.)
        self.assertLess(ref[-1],v[-1])

    def test_a_stopped_feasible_hold_keeps_zero_reference(self):
        q=np.zeros(7);g=np.array([0.,-43.,1.,22.,0.,0.,0.])
        ref,_=bounded_release_hold_velocity(q,q,q,g,stiffness=2500.,damping=160.,effort_limit=100.)
        np.testing.assert_array_equal(ref,q)

    def test_an_infeasible_stationary_hold_is_not_hidden_by_acceleration(self):
        q=np.zeros(7);g=np.array([0.,-143.,1.,22.,0.,0.,0.])
        with self.assertRaises(ValueError):
            bounded_release_hold_velocity(q,q,q,g,stiffness=2500.,damping=160.,effort_limit=100.)


if __name__=='__main__':unittest.main()
