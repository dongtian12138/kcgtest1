import math
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_keyed_public_spec_v2 import (
    EXPECTED_CONTACT_LABELS,
    EXPECTED_KEY_ANGLES_DEG,
    MASS_PROPERTY_SOURCE_KIND,
    PAIR_MODEL_ID,
    PLUG_MODEL_ID,
    RECEPTACLE_MODEL_ID,
    RECOMMENDED_ASSET_NAME,
    R4_MASS_PROPERTY_SOURCE_REVISION,
    keyed_pattern_fits_at_yaw,
    keyed_yaw_peak_to_peak_clearance_deg,
    load_keyed_public_spec_v2,
    rectangular_key_fits,
    safe_new_asset_output,
)


def test_public_spec_v2_is_a_new_simulation_only_identity():
    model = load_keyed_public_spec_v2()
    identity = model.document["identity"]
    authorization = model.document["authorization"]

    assert identity["pair_model_id"] == PAIR_MODEL_ID
    assert identity["loose_plug_model_id"] == PLUG_MODEL_ID
    assert identity["fixed_receptacle_model_id"] == RECEPTACLE_MODEL_ID
    assert identity["pair_model_id"] != "d38999_shell25j_61_pair_proxy_v1"
    assert identity["selected_class"] == "K"
    assert identity["space_grade_claimed"] is False
    assert identity["real_hardware_identity_claimed"] is False
    assert authorization["simulation_geometry_use_allowed"] is True
    assert authorization["simulation_insertion_control_authorized"] is False
    assert authorization["robot_control_authorized"] is False
    assert authorization["hardware_control_authorized"] is False
    assert authorization["real_assembly_success_claimed"] is False


def test_25_61_table_keeps_all_controlling_inch_coordinates():
    model = load_keyed_public_spec_v2()
    assert tuple(item.label for item in model.contacts) == EXPECTED_CONTACT_LABELS
    assert len(model.contacts) == 61
    assert len({(item.x_in, item.y_in) for item in model.contacts}) == 61

    by_label = {item.label: item for item in model.contacts}
    assert by_label["A"].pin_front_mm == pytest.approx((4.9784, 12.7))
    assert by_label["F"].pin_front_mm == pytest.approx((13.6144, -0.762))
    assert by_label["M"].pin_front_mm == pytest.approx((0.0, -13.6398))
    assert by_label["PP"].pin_front_mm == (0.0, 0.0)


def test_socket_front_screen_view_mirrors_x_once_only():
    model = load_keyed_public_spec_v2()
    by_label = {item.label: item for item in model.contacts}
    pin_x, pin_y = by_label["A"].pin_front_mm
    socket_x, socket_y = by_label["A"].socket_front_same_screen_axes_mm
    assert socket_x == -pin_x
    assert socket_y == pin_y
    assert model.document["contact_pattern"]["local_assembly_frame_rule"] == (
        "use_same_table_once_and_mirror_through_the_mating_transform"
    )


def test_n_polarization_is_five_key_and_not_c2():
    model = load_keyed_public_spec_v2()
    assert model.key_angles_deg == EXPECTED_KEY_ANGLES_DEG
    original = {round(angle % 360.0, 9) for angle in model.key_angles_deg}
    rotated_pi = {round((angle + 180.0) % 360.0, 9) for angle in model.key_angles_deg}
    assert original != rotated_pi

    keying = model.document["keying"]
    assert keying["polarization_must_precede_coupling_and_contact"] is True
    assert keying["plug_key_axial_length_min"] == 7.24


def test_tight_public_dimensions_leave_positive_keyway_clearance():
    model = load_keyed_public_spec_v2()
    keying = model.document["keying"]
    minor_key_max = (
        keying["plug_minor_key_width_nominal"]
        + keying["plug_minor_key_width_plus"]
    )
    minor_slot_min = (
        keying["receptacle_minor_keyway_width_nominal"]
        - keying["receptacle_minor_keyway_width_minus"]
    )
    assert minor_key_max == pytest.approx(1.35)
    assert minor_slot_min == pytest.approx(1.57)
    assert minor_slot_min - minor_key_max == pytest.approx(0.22)


def test_yaw_profiles_are_derived_and_ordered_not_invented_measurements():
    model = load_keyed_public_spec_v2()
    nominal = model.clearance_profile("nominal_centered")
    tight = model.clearance_profile("tight_size_centered")
    stress = model.clearance_profile("adversarial_gdt_stress")

    assert nominal.peak_to_peak_deg == pytest.approx(0.8509706248885368)
    assert tight.peak_to_peak_deg == pytest.approx(0.6661267828482448)
    assert stress.peak_to_peak_deg == pytest.approx(0.060550934851961585)
    assert nominal.peak_to_peak_deg > tight.peak_to_peak_deg > stress.peak_to_peak_deg
    assert stress.required_p95_deg == pytest.approx(0.030275467425980793)
    assert stress.derivation_kind == "project_adversarial_gdt_stress_assumption"
    assert stress.drawing_specified_clearance is False
    assert model.simulation_acceptance_profile == "adversarial_gdt_stress"
    profiles = model.document["yaw_clearance_profiles"]
    assert profiles["source_kind"] == (
        "SPEC_TOLERANCE_DERIVED_CONSERVATIVE_SIMULATION_ONLY"
    )
    assert profiles["real_measured_clearance_deg"] is None


def test_cross_section_accepts_zero_and_rejects_beyond_each_half_window():
    model = load_keyed_public_spec_v2()
    for profile in model.clearance_profiles:
        half = 0.5 * profile.peak_to_peak_deg
        kwargs = {
            "slot_width_mm": profile.slot_width_mm,
            "key_width_mm": profile.key_width_mm,
            "radius_mm": profile.radius_mm,
        }
        assert rectangular_key_fits(yaw_deg=0.0, **kwargs)
        assert rectangular_key_fits(yaw_deg=half - 1.0e-8, **kwargs)
        assert not rectangular_key_fits(yaw_deg=half + 1.0e-8, **kwargs)
        assert not rectangular_key_fits(yaw_deg=180.0, **kwargs)
        assert keyed_yaw_peak_to_peak_clearance_deg(**kwargs) == pytest.approx(
            profile.peak_to_peak_deg, abs=1.0e-12
        )


def test_complete_n_pattern_rejects_c2_and_alternate_polarizations():
    assert keyed_pattern_fits_at_yaw(0.0)
    assert keyed_pattern_fits_at_yaw(0.42)
    assert not keyed_pattern_fits_at_yaw(0.43)
    assert not keyed_pattern_fits_at_yaw(180.0)

    alternate_minor_angles = {
        "A": (135.0, 170.0, 200.0, 310.0),
        "B": (49.0, 169.0, 200.0, 244.0),
        "C": (66.0, 140.0, 200.0, 257.0),
        "D": (62.0, 145.0, 180.0, 280.0),
        "E": (79.0, 153.0, 197.0, 272.0),
    }
    for minor_angles in alternate_minor_angles.values():
        assert not keyed_pattern_fits_at_yaw(
            0.0,
            receptacle_keyway_angles_deg=(0.0, *minor_angles),
        )


def test_contact_face_fits_inside_public_bore_with_margin():
    model = load_keyed_public_spec_v2()
    max_radius_mm = max(math.hypot(*item.pin_front_mm) for item in model.contacts)
    bore_radius_min_mm = 0.5 * model.document["interface_dimensions_mm"][
        "receptacle"
    ]["h_bore_diameter_min"]
    pin_radius_max_mm = 0.5 * (0.040 + 0.001) * 25.4
    assert max_radius_mm + pin_radius_max_mm < bore_radius_min_mm


def test_mass_com_and_inertia_are_frozen_simulation_assumptions():
    model = load_keyed_public_spec_v2()
    assert model.mass_property_source_kind == MASS_PROPERTY_SOURCE_KIND
    assert (
        model.mass_property_source_asset_revision
        == R4_MASS_PROPERTY_SOURCE_REVISION
    )
    assert model.body_mass_properties.mass_kg == pytest.approx(0.23)
    assert model.nut_mass_properties.mass_kg == pytest.approx(0.08)
    assert model.body_mass_properties.center_of_mass_m == pytest.approx(
        (2.7594312541623367e-6, 1.5090811302798102e-6, -0.024185307323932648)
    )
    assert model.body_mass_properties.diagonal_inertia_kg_m2 == pytest.approx(
        (4.9906659114640206e-5, 4.993189941160381e-5, 5.680214235326275e-5)
    )
    assert model.nut_mass_properties.center_of_mass_m == pytest.approx(
        (0.0, 0.0, -0.019999999552965164)
    )
    section = model.document["simulation_mass_properties"]
    assert section["exact_hardware_mass_properties_available"] is False
    assert section["runtime_randomization_enabled_for_checkpoint_a"] is False


@pytest.mark.parametrize(
    "mutator",
    (
        lambda doc: doc["simulation_mass_properties"].update(
            source_asset_revision="wrong_revision"
        ),
        lambda doc: doc["simulation_mass_properties"].update(
            exact_hardware_mass_properties_available=True
        ),
        lambda doc: doc["simulation_mass_properties"]["body_assembly"].update(
            mass_kg=0.08
        ),
        lambda doc: doc["simulation_mass_properties"]["coupling_nut"].update(
            principal_axes_xyzw=[0.0, 0.0, 0.0, 2.0]
        ),
    ),
)
def test_mass_property_contract_fails_closed(tmp_path, mutator):
    source = load_keyed_public_spec_v2()
    document = deepcopy(source.document)
    mutator(document)
    path = tmp_path / "invalid_mass_properties.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_keyed_public_spec_v2(path)


def test_safe_output_refuses_legacy_names_and_overwrite(tmp_path: Path):
    for legacy_name in (
        "connector_pair.usda",
        "d38999_shell25j_61_pair_proxy_v1.usda",
        "d38999_insert_proxy_v2.usda",
    ):
        with pytest.raises(ValueError):
            safe_new_asset_output(tmp_path / legacy_name)

    output = tmp_path / RECOMMENDED_ASSET_NAME
    assert safe_new_asset_output(output) == output.resolve()
    output.touch()
    with pytest.raises(FileExistsError):
        safe_new_asset_output(output)
