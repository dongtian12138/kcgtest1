from __future__ import annotations

import csv
import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_j599_25_35_assets",
    ROOT / "src" / "build_j599_25_35_assets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class J599ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = MODULE.load_contract()
        cls.contacts = MODULE.load_contacts()

    def test_identity_matches_user_parts(self):
        identity = self.contract["identity"]
        self.assertEqual(
            identity["plug"]["part_number"], "J599/26FJ35PN"
        )
        self.assertEqual(identity["plug"]["contact_style"], "pin")
        self.assertEqual(
            identity["receptacle"]["part_number"], "J599/20FJ35SN"
        )
        self.assertEqual(identity["receptacle"]["contact_style"], "socket")
        self.assertEqual(identity["insert_arrangement"], "25-35")
        self.assertEqual(identity["contact_count"], 128)

    def test_contact_ids_and_controlling_coordinates(self):
        self.assertEqual(
            [item.contact_id for item in self.contacts], list(range(1, 129))
        )
        self.assertAlmostEqual(
            max(math.hypot(item.x_m, item.y_m) for item in self.contacts),
            0.555 * 0.0254,
        )
        with (
            ROOT / "data" / "contact_positions_25_35.csv"
        ).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row, item in zip(rows, self.contacts):
            self.assertLessEqual(
                abs(item.x_m * 1000.0 - float(row["x_mm"])), 0.015
            )
            self.assertLessEqual(
                abs(item.y_m * 1000.0 - float(row["y_mm"])), 0.015
            )

    def test_five_n_keys_have_positive_nominal_clearance(self):
        features = MODULE.key_features(self.contract)
        self.assertEqual(
            [item.angle_deg for item in features],
            [0.0, 80.0, 142.0, 196.0, 293.0],
        )
        self.assertTrue(
            all(
                value > 0.0
                for value in MODULE.nominal_key_clearances_m(self.contract)
            )
        )

    def test_three_degree_wrong_yaw_is_an_interference_case(self):
        self.assertTrue(MODULE.wrong_yaw_interferes(self.contract, 3.0))

    def test_thread_relation_is_three_start_7p62_mm_per_turn(self):
        thread = self.contract["public_interface_geometry"]["thread"]
        self.assertEqual(thread["starts"], 3)
        self.assertAlmostEqual(thread["pitch_mm"], 2.54)
        self.assertAlmostEqual(thread["lead_mm_per_revolution"], 7.62)
        self.assertEqual(thread["start_phases_deg"], [0.0, 120.0, 240.0])

    def test_truth_and_isolation_flags_fail_closed(self):
        authorization = self.contract["authorization"]
        self.assertFalse(authorization["hardware_authorized"])
        self.assertFalse(authorization["hardware_exact_fidelity"])
        self.assertFalse(authorization["real_hardware_assembly_success_claimed"])
        self.assertFalse(
            self.contract["scope"]["old_model_modification_allowed"]
        )
        self.assertFalse(self.contract["scope"]["robot_or_hand_included"])


if __name__ == "__main__":
    unittest.main()
