"""Regressions for sampled full-hand mesh rejection in the V2 fast gate."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from kcg_connector.grasp.carts_v2 import fast_filter
from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.models import (
    joint_positions_for_phases,
    load_v2_inputs,
)


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

    def _translated_seed(self, identifier: str, delta_world) -> object:
        pose = self.prediction.seed.object_from_hand_matrix()
        world_pose = self.inputs.frozen_world_from_object @ pose
        world_pose[:3, 3] += np.asarray(delta_world, dtype=np.float64)
        pose = np.linalg.inv(self.inputs.frozen_world_from_object) @ world_pose
        return replace(self.prediction.seed, candidate_id=identifier,
                       object_from_hand=tuple(float(value) for value in pose.reshape(-1)))

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

    def test_pregrasp_batch_reuses_scene_and_bounds_each_joint_step(self) -> None:
        far = self._translated_seed("far", (0.5, 0.0, 0.5))
        exhausted = replace(far, candidate_id="exhausted")
        with mock.patch.object(
            fast_filter, "_prepare_fcl_scene", wraps=fast_filter._prepare_fcl_scene
        ) as prepare, mock.patch.object(
            fast_filter, "_precontact_mesh_report",
            wraps=fast_filter._precontact_mesh_report,
        ) as inspect:
            safe, safe_second, no_remaining = fast_filter.fast_filter_pregrasp_paths(
                self.inputs,
                ((far, (0.1, 0.1, 0.1)), (far, (0.2, 0.2, 0.2)),
                 (exhausted, (0.75, 0.75, 0.75))),
            )
        prepare.assert_called_once_with(self.inputs)
        self.assertTrue(safe["accepted"])
        self.assertTrue(safe_second["accepted"])
        self.assertLessEqual(safe["maximum_joint_increment_rad"], 0.0015 + 1e-12)
        self.assertGreater(safe["checked_state_count"], 5)
        states = fast_filter._bounded_pregrasp_states(
            self.inputs, far, (0.1, 0.1, 0.1))
        second_states = fast_filter._bounded_pregrasp_states(
            self.inputs, far, (0.2, 0.2, 0.2))
        shared = sum(state[0].startswith("PALM_FAR_") for state in states)
        self.assertEqual(inspect.call_count, len(states) + len(second_states) - shared)
        self.assertEqual(safe["reused_identical_state_count"], 0)
        self.assertEqual(safe_second["reused_identical_state_count"], shared)
        self.assertEqual(safe_second["physical_query_state_count"],
                         len(second_states) - shared)
        self.assertAlmostEqual(states[0][2][0], 0.0)
        self.assertAlmostEqual(
            states[-1][2][0], far.pregrasp_joint_positions_rad[0])
        self.assertEqual(
            safe["unresolved_checks"],
            (
                "FCL_SURFACE_QUERY_CANNOT_PROVE_CONTAINMENT",
                "HAND_APPROACH_SAMPLED_NOT_CONTINUOUS",
            ),
        )
        self.assertEqual(
            no_remaining["reasons"],
            ("NO_POSITIVE_THREE_FINGER_REMAINING_CLOSURE",),
        )
        snapshot = fast_filter.fast_filter_pregrasp_paths(
            self.inputs, ((far, (0.0, 0.0, 0.0)),), budget_probe_only=True)[0]
        self.assertEqual(snapshot["checked_state_count"], 1)
        self.assertEqual(snapshot["physical_query_state_count"], 1)
        self.assertEqual(snapshot["unresolved_checks"][0],
                         "BUDGET_PROBE_PREGRASP_SNAPSHOT_ONLY")

    def test_remote_preshape_table_penetration_fails_closed(self) -> None:
        low = self._translated_seed("low", (0.0, 0.0, -0.2))
        result = fast_filter.fast_filter_pregrasp_paths(
            self.inputs, ((low, (0.1, 0.1, 0.1)),)
        )[0]
        self.assertFalse(result["accepted"])
        self.assertTrue(result["reasons"][0].startswith(
            "HAND_TABLE_PENETRATION:PALM_FAR_0000:"))
        self.assertLess(result["minimum_table_clearance_m"], 0.0)

    def test_target_pregrasp_pad_must_keep_initial_clearance(self) -> None:
        isolated = replace(self.inputs, table_top_z_m=-1.0)
        seed = self.prediction.seed
        phases = (0.1, 0.1, 0.1)
        base = isolated.frozen_world_from_object @ seed.object_from_hand_matrix()
        joints = joint_positions_for_phases(
            isolated, phases,
            reference_joint_positions_rad=seed.pregrasp_joint_positions_rad,
        )
        pad = isolated.hand_contract.pads[0]
        transform = isolated.hand_model.forward_kinematics(
            joints, base_transform=base
        )[pad.link_name]
        scene = fast_filter._prepare_fcl_scene(isolated)
        pad_object = fast_filter.fcl.CollisionObject(
            fast_filter.build_fcl_bvh_model(pad.points_local_m, pad.faces)
        )
        pad_object.setTransform(fast_filter.fcl.Transform(
            transform[:3, :3], transform[:3, 3]))
        witness = fast_filter.fcl.DistanceResult()
        fast_filter.fcl.distance(
            pad_object, scene[2],
            fast_filter.fcl.DistanceRequest(enable_nearest_points=True), witness,
        )
        pad_point, object_point = map(np.asarray, witness.nearest_points)
        direction = object_point - pad_point
        shift = direction * ((np.linalg.norm(direction) - 0.0005)
                             / np.linalg.norm(direction))
        near = self._translated_seed("near_pad", shift)
        result = fast_filter.fast_filter_pregrasp_paths(
            isolated, ((near, phases),)
        )[0]
        self.assertEqual(
            result["reasons"],
            ("PAD_OBJECT_PRECONTACT_DISTANCE:PREGRASP:finger_1_pad",),
        )
        self.assertEqual(len(result["pregrasp_pad_clearance_by_name_m"]), 3)
        self.assertLessEqual(
            result["pregrasp_pad_clearance_by_name_m"]["finger_1_pad"], 0.001)

    def test_task_grip_surface_intersection_is_not_hidden_from_path_gate(self) -> None:
        vertices = np.asarray(((0.0, 0.0, 0.0), (0.01, 0.0, 0.0),
                               (0.0, 0.01, 0.0)))
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        object_collision = fast_filter.fcl.CollisionObject(
            fast_filter.build_fcl_bvh_model(vertices, faces))
        contact_collision = fast_filter.fcl.CollisionObject(
            fast_filter.build_fcl_bvh_model(vertices, faces))
        hand = SimpleNamespace(forward_kinematics=lambda *_args, **_kwargs: {
            "tip": np.eye(4)
        })
        inputs = SimpleNamespace(
            hand_model=hand, task_grip_surfaces={"finger_1_pad": object()})
        scene = ({}, {}, object_collision, (), {
            "finger_1_pad": ("tip", contact_collision)
        })
        reason = fast_filter._first_state_collision(
            inputs, (("APPROACH_00", np.eye(4), np.zeros(1)),), scene)
        self.assertEqual(
            reason,
            "TASK_GRIP_SURFACE_OBJECT_INTERSECTION:APPROACH_00:finger_1_pad",
        )
