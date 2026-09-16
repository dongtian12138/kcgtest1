"""Native-only diagnostics retain supplied points and mark unobserved channels."""
import copy
import unittest
from carts_v2.evaluate_run import TruthAuditRecorder


class NativeContactAuditTests(unittest.TestCase):
    def recorder(self):
        row={'paths':('/object/Body','/table','/object/Body/shape','/table/shape'),
             'records':2,'contact_data_offset':0,'contacts':[
                 {'position_m':[.1,.2,.3],'normal':[0.,0.,1.],
                  'impulse_n_s':[0.,0.,.02],'separation_m':-.000001},
                 {'position_m':[.2,.3,.4],'normal':[0.,0.,1.],
                  'impulse_n_s':[0.,0.,0.],'separation_m':.00001}]}
        r=object.__new__(TruthAuditRecorder);r.contact_audit_mode='native-report'
        r.tensor_contact_sensor_paths=['/object/Body'];r._physics_step_reports=[[copy.deepcopy(row)]]
        r._event_headers=[(row['paths'],2)]
        r.roots={'object':'/object','robot':'/robot','table':'/table','fixture':'/fixture'}
        def forbidden():raise AssertionError('an omitted diagnostic getter was called')
        r._tensor_contact_rows=forbidden;r._tensor_friction_rows=forbidden
        return r,row

    def test_native_points_are_retained_including_zero_impulse_points(self):
        r,row=self.recorder();result=r._contact_counts()
        self.assertEqual(result['poll_headers'][0]['contacts'],row['contacts'])
        self.assertEqual(result['poll_contact_data_count'],2)
        self.assertEqual(result['object_table_positive_normal_impulse_n_s'],.02)
        self.assertTrue(result['contact_report_channels_agree'])
        self.assertFalse(result['tensor_contacts_available'])
        self.assertFalse(result['friction_data_available'])
        self.assertEqual(result['uncollected_contact_channels'],['tensor_raw_contact','tensor_friction'])

    def test_nonfinite_native_evidence_is_rejected_without_tensor_validation(self):
        r,_=self.recorder()
        r._physics_step_reports[0][0]['contacts'][0]['impulse_n_s'][0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'not finite'):r._contact_counts()

    def test_missing_native_callback_is_not_mistaken_for_no_contact(self):
        r,_=self.recorder();r._physics_step_reports=[]
        with self.assertRaisesRegex(RuntimeError,'did not provide'):r._contact_counts()


if __name__=='__main__':unittest.main()
