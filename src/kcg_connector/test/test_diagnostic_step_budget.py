"""Diagnostic stops must return before issuing another physical command."""
from pathlib import Path
import tempfile
import unittest

from carts_v2.controller import JointSignalStepper


class DiagnosticStopTests(unittest.TestCase):
    def stepper(self):
        value=object.__new__(JointSignalStepper)
        value.abort_reason=None;value.latest=object();value.step_index=1920
        return value

    def test_step_budget_stops_without_needing_a_robot_or_world(self):
        value=self.stepper();value.diagnostic_step_limit=1920
        self.assertIs(value.advance('key_probe_nut_transfer',None,None),value.latest)
        self.assertEqual(value.abort_reason,'DIAGNOSTIC_STEP_BUDGET_REACHED')

    def test_stop_request_is_seen_in_every_motion_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'STOP_REQUEST';path.touch()
            for phase in ('key_probe_nut_open_hold','key_probe_nut_transfer',
                          'key_probe_nut_grip_hold','key_probe_nut_rotation_turn'):
                with self.subTest(phase=phase):
                    value=self.stepper();value.diagnostic_stop_request_path=str(path)
                    self.assertIs(value.advance(phase,None,None),value.latest)
                    self.assertEqual(value.abort_reason,'DIAGNOSTIC_STOP_REQUESTED')

    def test_original_failure_reason_is_preserved(self):
        value=self.stepper();value.abort_reason='ORIGINAL_FORCE_LIMIT'
        value.diagnostic_step_limit=0
        self.assertIs(value.advance('turn',None,None),value.latest)
        self.assertEqual(value.abort_reason,'ORIGINAL_FORCE_LIMIT')


if __name__=='__main__':unittest.main()
