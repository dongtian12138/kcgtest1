import unittest
from te_nut_motion import remaining_command_stroke, source_probe_rotation_degrees


class RemainingCommandTests(unittest.TestCase):
    def test_measured_early_stop_deficit_is_added_to_last_stroke(self):
        before=180.+85.4446251572038
        angle=remaining_command_stroke(90.,before,360.,final_interval=True)
        self.assertAlmostEqual(angle,94.5553748427962)
        self.assertAlmostEqual(before+angle,360.)

    def test_normal_intermediate_stroke_keeps_its_angle(self):
        self.assertEqual(remaining_command_stroke(90.,180.,360.,final_interval=False),90.)

    def test_last_stroke_cannot_exceed_its_geometric_angle_limit(self):
        self.assertEqual(remaining_command_stroke(90.,239.95,360.,final_interval=True),120.)

    def test_finished_budget_does_not_create_a_tiny_extra_turn(self):
        self.assertEqual(remaining_command_stroke(90.,359.995,360.,final_interval=True),0.)

    def test_invalid_budget_is_not_sent_to_a_controller(self):
        for before in (float('nan'),-1.):
            with self.assertRaises(ValueError):
                remaining_command_stroke(90.,before,360.,final_interval=True)

    def test_diagnostic_accepts_the_actual_compensated_final_stroke(self):
        angle=remaining_command_stroke(90.,265.4446251572038,360.,final_interval=True)
        self.assertEqual(source_probe_rotation_degrees({'rotation_degrees':angle}),angle)

    def test_diagnostic_rejects_invalid_angles_before_physics(self):
        for value in (0.,-1.,120.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                source_probe_rotation_degrees({'rotation_degrees':value})
        self.assertIsNone(source_probe_rotation_degrees({}))


if __name__=='__main__':unittest.main()
