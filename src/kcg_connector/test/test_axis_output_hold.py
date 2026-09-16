import unittest
from types import SimpleNamespace
from te_worm_drive import WormDrive,WormReference,finger_output_posture_targets


class AxisOutputHoldTests(unittest.TestCase):
    def test_finite_motor_input_compensates_both_directions_of_output_drift(self):
        names=('f1j2','f2j1','f3j2')
        drives={n:WormDrive(.67,reference=WormReference(transmission_damping=0.,output_viscosity=2.),integration='passive_split') for n in names}
        for d in drives.values():d.input_angle=.695
        mechanism=SimpleNamespace(drives=drives,settings={'motor_position_kp':264.,'motor_position_kd':4.4},
            setup={'intervals':{n:(.5,.85) for n in names}})
        actual=[.669,.671,.67];reference=[1.047,.695,.695,.695];goal=[1.047,.67,.67,.67]
        targets,records=finger_output_posture_targets(mechanism,actual,reference,goal,1/960,allow_closing=True)
        for i,n in enumerate(names):
            drives[n].prepare_position(actual[i],0.,targets[i+1],1/960,stiffness=264.,damping=4.4)
            if i==0:self.assertGreater(drives[n].pending['v'],0.)
            elif i==1:self.assertLess(drives[n].pending['v'],0.)
            else:self.assertEqual(drives[n].pending['v'],0.)
            self.assertLessEqual(abs(records[i]['requested_motor_effort_nm']),7.7)
        self.assertEqual(actual,[.669,.671,.67])


if __name__=='__main__':unittest.main()
