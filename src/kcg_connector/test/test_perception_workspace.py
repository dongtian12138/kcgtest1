import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from perception_workspace import select_detections


class PerceptionWorkspaceTests(unittest.TestCase):
    def fixture(self,folder):
        records=[{'score':.9,'bbox':[0,0,20,20],'segmentation':{'source':'left'}},
                 {'score':.5,'bbox':[20,0,20,20],'segmentation':{'source':'right'}}]
        mask=np.zeros((2,20,40));mask[0,:,:20]=1;mask[1,:,20:]=1
        source=folder/'detections.json';source.write_text(json.dumps(records))
        archive=folder/'detections.npz'
        np.savez(archive,segmentation=mask,score=np.array([.9,.5]),bbox=np.array([x['bbox'] for x in records]))
        return records,source,archive

    def test_higher_scoring_wrong_region_is_excluded_without_changing_scores_or_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);records,source,archive=self.fixture(p)
            result=select_detections(source,archive,np.ones((20,40)),[[100,0,20],[0,100,10],[0,0,1]],
                np.eye(4),{'minimum':[0,-.2,.9],'maximum':[.25,.2,1.1]},p/'selected.json',p/'summary.json')
            self.assertEqual(json.loads((p/'selected.json').read_text()),[records[1]])
            self.assertEqual(json.loads(source.read_text()),records)
            self.assertFalse(result['object_or_contact_truth_used'])

    def test_no_depth_support_does_not_fall_back_to_wrong_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);records,source,archive=self.fixture(p)
            with self.assertRaises(RuntimeError):
                select_detections(source,archive,np.full((20,40),np.nan),[[100,0,20],[0,100,10],[0,0,1]],
                    np.eye(4),{'minimum':[0,-.2,.9],'maximum':[.25,.2,1.1]},p/'selected.json',p/'summary.json')
            self.assertFalse((p/'selected.json').exists())
            self.assertEqual(json.loads((p/'summary.json').read_text())['kept_count'],0)


if __name__=='__main__':unittest.main()
