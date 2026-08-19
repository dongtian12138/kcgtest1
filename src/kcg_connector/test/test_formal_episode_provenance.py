'''Contract tests for the per-episode formal provenance payload.

Every formal episode must record the SHA-256 of every source file that
decides the formal lift PASS decision.  Round 026a added the two missing
control sources to the runner's metrics["provenance"] block:

* "grasp_stability_monitor_sha256" - the monitor that implements the
  8 N wrist-force / 0.30 N*m wrist-moment sensor gates;
* "physical_grasp_config_loader_sha256" - the strict loader that
  instantiates that monitor from the YAML (the YAML digest itself stays
  under the unchanged "physical_grasp_config_sha256" key).

These tests pin both keys to the per-episode provenance block, prove the
paths are built from repository_root (the files the runner actually
imports), pin the fail-closed hashing helper, and verify the recording is
evidence-only: the keys never feed control, randomization, staging, gate
thresholds, or truth boundaries.
'''

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from kcg_connector.grasp.single_finger_posthoc_audit import (
    PROVENANCE_DIGEST_KEYS,
    PROVENANCE_KEYS,
)

REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
)

NEW_PROVENANCE_KEYS = (
    "grasp_stability_monitor_sha256",
    "physical_grasp_config_loader_sha256",
    "three_finger_sequential_grasp_sha256",
)

NEW_SOURCE_RELATIVES = {
    "grasp_stability_monitor_sha256": (
        "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
    ),
    "physical_grasp_config_loader_sha256": (
        "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
    ),
    "three_finger_sequential_grasp_sha256": (
        "src/kcg_connector/kcg_connector/grasp/"
        "three_finger_sequential_grasp.py"
    ),
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _tree():
    return ast.parse(_source())


def _call_name(call):
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def _provenance_entries(tree):
    dictionary = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "metrics"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "provenance"
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(
                "metrics['provenance'] is not a dict literal"
            )
        dictionary = node.value
        break
    assert dictionary is not None, "metrics['provenance'] dict not found"
    entries = {}
    for key_node, value_node in zip(dictionary.keys, dictionary.values):
        assert isinstance(key_node, ast.Constant)
        entries[key_node.value] = value_node
    return entries


def _normalized(expression):
    return "".join(ast.unparse(expression).split())


def test_provenance_records_monitor_and_loader_source_hashes():
    entries = _provenance_entries(_tree())
    for key in NEW_PROVENANCE_KEYS:
        value = entries.get(key)
        assert value is not None, f"provenance key {key!r} missing"
        assert isinstance(value, ast.Call), key
        assert _call_name(value) == "_sha256_file", key
        assert len(value.args) == 1, key


def test_new_provenance_paths_are_built_from_repository_root():
    entries = _provenance_entries(_tree())
    for key, relative in NEW_SOURCE_RELATIVES.items():
        value = entries[key]
        directory, filename = relative.rsplit("/", 1)
        expected = f"repository_root/{directory!r}/{filename!r}"
        assert _normalized(value.args[0]) == expected, key


def test_provenance_sha256_helper_fails_closed():
    tree = _tree()
    helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sha256_file"
    ]
    assert len(helpers) == 1, "expected exactly one _sha256_file helper"
    function = helpers[0]
    assert any(
        isinstance(node, ast.Raise) for node in ast.walk(function)
    ), "_sha256_file must raise instead of recording a null/fake digest"
    assert any(
        isinstance(node, ast.Call) and _call_name(node) == "is_file"
        for node in ast.walk(function)
    ), "_sha256_file must check the source file exists first"
    assert not any(
        isinstance(node, ast.Try) for node in ast.walk(function)
    ), "_sha256_file must not swallow read errors"
    assert any(
        isinstance(node, ast.Call) and _call_name(node) == "hexdigest"
        for node in ast.walk(function)
    ), "_sha256_file must return a real digest"


def test_new_provenance_keys_are_recorded_once_and_never_consumed():
    source = _source()
    for key in NEW_PROVENANCE_KEYS:
        assert source.count(f'"{key}"') == 1, (
            f"{key!r} must appear exactly once in the runner "
            "(recorded, never read back into control)"
        )


def test_recorded_source_files_are_the_actually_imported_modules():
    import kcg_connector.grasp.grasp_stability_monitor as monitor_module
    import kcg_connector.grasp.physical_grasp_config as loader_module
    sequential_module = __import__(
        "kcg_connector.grasp.three_finger_sequential_grasp",
        fromlist=["three_finger_sequential_grasp"],
    )

    bindings = {
        "grasp_stability_monitor_sha256": monitor_module.__file__,
        "physical_grasp_config_loader_sha256": loader_module.__file__,
        "three_finger_sequential_grasp_sha256": sequential_module.__file__,
    }
    for key, module_file in bindings.items():
        recorded = (REPOSITORY / NEW_SOURCE_RELATIVES[key]).resolve()
        assert Path(module_file).resolve() == recorded, key
        digest = hashlib.sha256(recorded.read_bytes()).hexdigest()
        assert SHA256_PATTERN.fullmatch(digest), key


def test_yaml_config_digest_key_is_unchanged():
    entries = _provenance_entries(_tree())
    value = entries.get("physical_grasp_config_sha256")
    assert isinstance(value, ast.Call)
    assert _call_name(value) == "_sha256_file"
    assert len(value.args) == 1
    assert _normalized(value.args[0]) == "physical_grasp_path", (
        "the YAML digest must keep pointing at the config file path"
    )


def test_provenance_superset_of_offline_audit_comparator_contract():
    keys = set(_provenance_entries(_tree()))
    for required in PROVENANCE_KEYS:
        assert required in keys, required
    for digest_key in PROVENANCE_DIGEST_KEYS:
        assert digest_key in keys, digest_key
    for key in NEW_PROVENANCE_KEYS:
        assert key in keys, key
    assert "audit_mode" in keys


def test_gate_and_stage_thresholds_unchanged_in_config():
    document = CONFIG.read_text(encoding="utf-8")
    for expected in (
        "maximum_wrist_force_n: 8.0",
        "maximum_wrist_moment_nm: 0.30",
        "stage_increment_m: [0.002, 0.010, 0.040]",
        "stage_speed_m_s: [0.002, 0.004, 0.008]",
        "stage_hold_s: [0.50, 0.50, 2.00]",
        "maximum_root_torque_delta_nm: 2.0",
    ):
        assert expected in document, expected


def test_randomization_realization_still_called_exactly_once():
    tree = _tree()
    count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "realize_randomization"
    )
    assert count == 1, "randomization boundary must stay untouched"
