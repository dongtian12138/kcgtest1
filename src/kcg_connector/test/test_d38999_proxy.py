"""Pure contract tests for the public-dimensional D38999 proxy."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.d38999_proxy import (
    DEFAULT_D38999_PROXY_CONFIG_PATH,
    D38999_PROXY_SCHEMA_VERSION,
    RECOMMENDED_D38999_ASSET_NAME,
    load_d38999_shell25j_proxy,
    require_safe_d38999_output,
    verify_public_source_files,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PACKAGE_ROOT / "isaac/create_d38999_proxy_asset.py"
VALIDATOR_PATH = PACKAGE_ROOT / "isaac/validate_d38999_proxy_asset.py"


def _document():
    return yaml.safe_load(
        DEFAULT_D38999_PROXY_CONFIG_PATH.read_text(encoding="utf-8")
    )


def _write(tmp_path, document):
    path = tmp_path / "proxy.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_shipped_proxy_selects_the_checked_shell25j_mating_pair():
    config = load_d38999_shell25j_proxy()
    assert config.schema_version == D38999_PROXY_SCHEMA_VERSION
    assert config.identity.loose_part_number == "D38999/26KJ61SN"
    assert config.identity.fixed_part_number == "D38999/20KJ61PN"
    assert config.identity.shell_size == 25
    assert config.identity.shell_size_code == "J"
    assert config.identity.insert_arrangement == 61
    assert config.identity.loose_contact_style == "socket"
    assert config.identity.fixed_contact_style == "pin"
    assert config.identity.certification_claim == "none"


def test_j_rows_are_transcribed_and_class_k_selects_unprimed_envelopes():
    config = load_d38999_shell25j_proxy()
    receptacle = config.receptacle_public_mm
    plug = config.plug_public_mm
    assert receptacle.s_nominal == pytest.approx(46.0)
    assert receptacle.r1 == pytest.approx(38.10)
    assert receptacle.r2 == pytest.approx(34.93)
    assert receptacle.v_nominal == pytest.approx(20.07)
    assert receptacle.z_max == pytest.approx(31.5)
    assert receptacle.z_prime_max == pytest.approx(32.0)
    assert plug.b_diameter_nominal == pytest.approx(44.3)
    assert plug.b_prime_diameter_max == pytest.approx(46.8)
    assert plug.h_max_rib_count == 28
    assert plug.k_max == pytest.approx(44.9)
    assert plug.s_diameter_max == pytest.approx(48.0)
    assert plug.z_max == pytest.approx(31.0)
    assert config.receptacle_source.selected_class == "K"
    assert config.plug_source.selected_class == "K"


def test_class_j_m_bending_row_is_not_a_selected_class_k_safety_limit():
    config = load_d38999_shell25j_proxy()
    for source, dimensions in (
        (config.receptacle_source, config.receptacle_public_mm),
        (config.plug_source, config.plug_public_mm),
    ):
        assert source.selected_class == "K"
        assert "apply only to classes J and M" in source.applicability_note
        assert "not applicable to selected class K" in (
            source.applicability_note
        )
        assert dimensions.class_j_m_external_bending_moment_minimum_nm == (
            pytest.approx(91.310)
        )
        assert not hasattr(
            dimensions, "external_bending_moment_minimum_nm"
        )


def test_proxy_envelopes_follow_public_rows_but_unknowns_remain_explicit():
    config = load_d38999_shell25j_proxy()
    assert config.plug_geometry_m.coupling_nut_outer_radius == pytest.approx(
        0.5 * config.plug_public_mm.s_diameter_max * 1.0e-3
    )
    assert config.receptacle_geometry_m.flange_side == pytest.approx(
        config.receptacle_public_mm.s_nominal * 1.0e-3
    )
    assert config.plug_geometry_m.contact_count == 61
    assert config.receptacle_geometry_m.contact_count == 61
    assert config.rules.thread_collision_mode == "none"
    assert config.rules.certified_geometry is False
    assert config.rules.space_qualified_claim is False
    assert "thread_profile_pitch_running_clearance_and_end_stops" in (
        config.unknowns
    )
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)


def test_both_public_dla_sources_match_recorded_sha256():
    paths = verify_public_source_files(
        load_d38999_shell25j_proxy(), PACKAGE_ROOT
    )
    assert [path.name for path in paths] == [
        "dtl38999ss20.pdf",
        "dtl38999ss26.pdf",
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda doc: doc.update(extra=True), "keys are invalid"),
        (
            lambda doc: doc["identity"].update(certification_claim="MIL"),
            "certification_claim",
        ),
        (
            lambda doc: doc["identity"].update(
                loose_part_number="D38999/26KJ61PN"
            ),
            "loose_part_number",
        ),
        (
            lambda doc: doc["public_dimensions_mm"]["plug"].update(
                s_diameter_max=49.0
            ),
            "public row differs",
        ),
        (
            lambda doc: doc["proxy_geometry_m"]["plug"].update(
                coupling_nut_outer_radius=0.025
            ),
            "exceeds public S",
        ),
        (
            lambda doc: doc["proxy_rules"].update(certified_geometry=True),
            "cannot claim fidelity",
        ),
        (
            lambda doc: doc["source_documents"]["plug"].update(
                local_path="../vendor.stp"
            ),
            "package-relative",
        ),
        (
            lambda doc: doc["source_documents"]["plug"].update(
                applicability_note=(
                    "The 91.310 N-m value is a selected class K limit."
                )
            ),
            "exclude the class J/M bending row",
        ),
        (
            lambda doc: doc["proxy_geometry_m"]["plug"].update(
                contact_count=60
            ),
            "exactly 61",
        ),
    ),
)
def test_loader_rejects_schema_drift_or_fidelity_overclaim(
    tmp_path, mutator, message
):
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(ValueError, match=message):
        load_d38999_shell25j_proxy(_write(tmp_path, document))


@pytest.mark.parametrize(
    "script_path",
    (GENERATOR_PATH, VALIDATOR_PATH),
)
def test_isaac_entrypoints_and_loader_import_without_runtime(script_path):
    script = f'''
import importlib.util
import json
import sys
from kcg_connector.d38999_proxy import load_d38999_shell25j_proxy
spec = importlib.util.spec_from_file_location(
    "d38999_entrypoint", {str(script_path)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_import": True}}))
'''
    environment = dict(__import__("os").environ)
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


def test_generator_and_validator_fail_closed_after_simulation_app_start():
    for path in (GENERATOR_PATH, VALIDATOR_PATH):
        source = path.read_text(encoding="utf-8")
        assert "passed = False" in source
        assert "except BaseException" in source
        assert "traceback.print_exc()" in source
        assert "exit_code=0 if passed else 1" in source


def test_output_guard_preserves_legacy_asset(tmp_path):
    with pytest.raises(ValueError, match="legacy connector_pair"):
        require_safe_d38999_output(tmp_path / "connector_pair.usda")
    safe = require_safe_d38999_output(
        tmp_path / RECOMMENDED_D38999_ASSET_NAME
    )
    assert safe.name == RECOMMENDED_D38999_ASSET_NAME
    with pytest.raises(ValueError, match=".usd"):
        require_safe_d38999_output(tmp_path / "proxy.obj")
