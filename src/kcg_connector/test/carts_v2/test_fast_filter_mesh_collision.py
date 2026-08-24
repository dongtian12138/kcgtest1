"""Regressions for sampled full-hand mesh rejection in the V2 fast gate."""

from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from kcg_connector.grasp.carts_v2 import fast_filter
from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.models import load_v2_inputs


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_grasp_v2.yaml"
OBJECT_A = "current_d38999_26kj61sn_public_spec"


class FastFilterMeshCollisionTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
        seed = next(row for row in generate_raw_candidates(cls.inputs)
                    if row.source_sample_index == 50)
        cls.prediction = SequentialClosurePredictor(cls.inputs).predict(seed)
        assert cls.prediction.status == "CLOSURE_SURVIVE"

    def test_exact_pad_exclusion_exposes_candidate_11_nonpad_contact(self) -> None:
        surfaces = fast_filter._exact_nonpad_surfaces(self.inputs)
        self.assertEqual(
            {len(surfaces[pad.link_name]) for pad in self.inputs.hand_contract.pads},
            {11713},
        )
        scene = fast_filter._prepare_fcl_scene(self.inputs)
        self.assertEqual(len(scene[3]), 28)
        reason = fast_filter._first_state_collision(
            self.inputs,
            fast_filter._sampled_hand_states(self.inputs, self.prediction),
            scene,
        )
        self.assertEqual(
            reason,
            "NONPAD_HAND_OBJECT_COLLISION:FINGER_1_CLOSURE_0248:f1Link3",
        )

    def test_missing_fcl_fails_closed(self) -> None:
        pose = self.prediction.seed.object_from_hand_matrix()
        world_pose = self.inputs.frozen_world_from_object @ pose
        world_pose[2, 3] += 0.5  # Isolate availability from the known table sweep.
        pose = np.linalg.inv(self.inputs.frozen_world_from_object) @ world_pose
        seed = replace(
            self.prediction.seed,
            candidate_id="fcl_missing",
            object_from_hand=tuple(float(value) for value in pose.reshape(-1)),
        )
        with mock.patch.object(fast_filter, "fcl", None):
            result = fast_filter.fast_filter_predictions(
                self.inputs, (replace(self.prediction, seed=seed),)
            )[0]
        self.assertEqual(result.reasons, ("FCL_MESH_COLLISION_BACKEND_UNAVAILABLE",))
        self.assertEqual(result.status, "FAST_REJECT")
        self.assertTrue(result.sequential_closure_sweep_pass)
