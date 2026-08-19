"""Pure-CPU review tests for the public-spec keyed-v2 asset generator."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kcg_connector.d38999_keyed_public_spec_v2 import (
    EXPECTED_KEY_ANGLES_DEG,
    PAIR_MODEL_ID,
    PLUG_MODEL_ID,
    RECEPTACLE_MODEL_ID,
    RECOMMENDED_ASSET_NAME,
    ROOT_PRIM,
    load_keyed_public_spec_v2,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    PACKAGE_ROOT / "isaac/create_d38999_keyed_public_spec_v2_asset.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "d38999_keyed_public_spec_v2_asset_generator", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_import_is_lazy_and_does_not_require_isaac_runtime():
    script = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location(
    'keyed_v2_generator', {str(GENERATOR_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for name in ('isaacsim', 'omni', 'pxr'):
    assert name not in sys.modules, name
print(json.dumps({{'lazy_import': True}}))
"""
    environment = dict(os.environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"lazy_import": True}


def test_plan_uses_independent_root_ids_suffixes_and_recommended_output():
    generator = _load_generator()
    plan = generator.build_asset_plan(load_keyed_public_spec_v2())

    assert plan.root_path == ROOT_PRIM
    assert plan.fixed_path == ROOT_PRIM + "/FixedReceptacle"
    assert plan.loose_path == ROOT_PRIM + "/LoosePlug"
    assert plan.body_path == ROOT_PRIM + "/LoosePlug/BodyAssembly"
    assert plan.nut_path == ROOT_PRIM + "/LoosePlug/CouplingNut"
    assert plan.joint_path == ROOT_PRIM + "/LoosePlug/CouplingNutJoint"
    assert plan.pair_model_id == PAIR_MODEL_ID
    assert plan.plug_model_id == PLUG_MODEL_ID
    assert plan.receptacle_model_id == RECEPTACLE_MODEL_ID
    assert plan.recommended_asset_name == RECOMMENDED_ASSET_NAME
    assert generator.recommended_output_path().name == RECOMMENDED_ASSET_NAME
    assert "simulation_only" in plan.geometry_fidelity
    assert plan.plug_socket_visual_front_offset_m == pytest.approx(5.0e-5)
    assert plan.body_mass_properties.mass_kg == pytest.approx(0.23)
    assert plan.nut_mass_properties.mass_kg == pytest.approx(0.08)
    assert "not_public_spec" in plan.mass_property_source_kind


def test_plan_authors_five_n_keys_and_matching_collision_keyways():
    generator = _load_generator()
    plan = generator.build_asset_plan(load_keyed_public_spec_v2())

    assert tuple(feature.angle_deg for feature in plan.keys) == (
        EXPECTED_KEY_ANGLES_DEG
    )
    assert len(plan.keys) == 5
    assert plan.keys[0].name == "Main"
    assert plan.keys[0].key_width_m == pytest.approx(0.00254)
    assert plan.keys[0].keyway_width_m == pytest.approx(0.00320)
    assert all(
        feature.key_width_m < feature.keyway_width_m
        for feature in plan.keys
    )
    assert plan.plug_shell_outer_radius_m < plan.receptacle_bore_radius_m
    assert plan.plug_rear_body_radius_m == pytest.approx(0.02215)
    assert (
        plan.plug_key_outer_radius_m
        < plan.receptacle_keyway_outer_radius_m
    )
    assert generator.key_pattern_fits_at_yaw(plan, 0.0) is True
    assert generator.key_pattern_fits_at_yaw(plan, 180.0) is False


def test_wrong_c2_yaw_is_blocked_at_keyway_before_visual_contact_plane():
    generator = _load_generator()
    plan = generator.build_asset_plan(load_keyed_public_spec_v2())

    assert plan.polarization_collision_plane_z_m == 0.0
    assert (
        plan.polarization_collision_plane_z_m
        < plan.first_electrical_contact_plane_z_m
        < plan.receptacle_guide_length_m
    )
    assert plan.first_electrical_contact_plane_z_m == pytest.approx(0.012)
    assert plan.contact_collision_mode == "visual_only"
    assert plan.thread_collision_mode == "unmodeled"


def test_plan_keeps_all_61_public_coordinates_visual_only():
    generator = _load_generator()
    plan = generator.build_asset_plan(load_keyed_public_spec_v2())

    assert len(plan.contacts) == 61
    assert len({(item.x_m, item.y_m) for item in plan.contacts}) == 61
    by_label = {item.label: item for item in plan.contacts}
    assert (by_label["A"].x_m, by_label["A"].y_m) == pytest.approx(
        (0.0049784, 0.0127)
    )
    assert (by_label["F"].x_m, by_label["F"].y_m) == pytest.approx(
        (0.0136144, -0.000762)
    )
    assert (by_label["PP"].x_m, by_label["PP"].y_m) == (0.0, 0.0)


def test_source_has_explicit_key_keyway_colliders_and_mass_properties():
    source = GENERATOR_PATH.read_text(encoding="utf-8")

    assert 'body_path + "/CollisionKeys"' in source
    assert 'fixed_path + "/CollisionKeyways"' in source
    assert 'collision_role="polarization_key"' in source
    assert 'collision_role="polarization_keyway_sidewall"' in source
    assert 'collision_role="polarization_keyway_outer_wall"' in source
    assert 'plan.nut_path + "/CollisionGripShell"' in source
    assert 'collision_role="coupling_nut_continuous_grip"' in source
    assert '"continuous_cylinder_for_grasp_stability"' in source
    assert '"MIL-DTL-38999/26G_A4_Figure1_B_shell25"' in source
    assert '"kcg:collisionMode", "visual_only"' in source
    assert '"kcg:threadCollisionMode", plan.thread_collision_mode' in source
    assert "PhysxSchema.PhysxCollisionAPI.Apply" in source
    assert "CreateContactOffsetAttr" in source
    assert "CreateRestOffsetAttr" in source
    assert "CreateCenterOfMassAttr" in source
    assert "CreateDiagonalInertiaAttr" in source
    assert "CreatePrincipalAxesAttr" in source
    assert '"kcg:massPropertiesSourceAssetRevision"' in source
    assert "plan.plug_socket_visual_front_offset_m" in source
    assert "hashlib" not in source


def test_collision_offset_is_explicit_and_smaller_than_radial_clearance():
    generator = _load_generator()
    plan = generator.build_asset_plan(load_keyed_public_spec_v2())
    radial_clearance = (
        plan.receptacle_bore_radius_m - plan.plug_shell_outer_radius_m
    )

    assert generator.COLLISION_CONTACT_OFFSET_M_SIM_ASSUMPTION == pytest.approx(
        1.0e-5
    )
    assert generator.COLLISION_REST_OFFSET_M_SIM_ASSUMPTION == 0.0
    assert 2.0 * generator.COLLISION_CONTACT_OFFSET_M_SIM_ASSUMPTION < (
        radial_clearance
    )


def test_generator_fails_closed_after_simulation_app_start():
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    assert "passed = False" in source
    assert "except BaseException" in source
    assert "traceback.print_exc()" in source
    assert "exit_code=0 if passed else 1" in source
