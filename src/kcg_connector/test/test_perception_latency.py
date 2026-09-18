import unittest
from perception_latency import DelayedObservation


class PerceptionLatencyTests(unittest.TestCase):
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
