from __future__ import annotations

import importlib.util
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = (
    WORKSPACE_ROOT
    / "src/kcg_connector/isaac/build_d38999_multilayer_models.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_d38999_multilayer_models_v2_test", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_override_generates_one_deterministic_combined_assembly_asset() -> None:
    generator = _load_generator()
    contract_path = WORKSPACE_ROOT / generator.CONTRACT_RELATIVE_PATH
    physical_contract_path = (
        WORKSPACE_ROOT / generator.PHYSICAL_CONTRACT_RELATIVE_PATH
    )
    override_path = WORKSPACE_ROOT / generator.AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH
    output_root = WORKSPACE_ROOT / generator.OUTPUT_ROOT_RELATIVE_PATH

    contract = generator._load_and_validate_contract(contract_path)
    overrides = generator._load_and_validate_authorized_overrides_v2(override_path)
    contract_sha = generator._sha256(contract_path)
    physical_contract_sha = generator._sha256(physical_contract_path)
    physical_shoulder_recipe = (
        generator._load_and_validate_physical_shoulder_recipe(
            physical_contract_path, contract
        )
    )
    override_sha = generator._sha256(override_path)
    generator_sha = generator._sha256(GENERATOR_PATH)
    first, first_mapping = generator._build_documents(
        contract,
        output_root=output_root,
        contract_sha=contract_sha,
        physical_contract_sha=physical_contract_sha,
        physical_shoulder_recipe=physical_shoulder_recipe,
        generator_sha=generator_sha,
        authorized_overrides_v2=overrides,
        authorized_overrides_v2_sha256=override_sha,
    )
    second, second_mapping = generator._build_documents(
        contract,
        output_root=output_root,
        contract_sha=contract_sha,
        physical_contract_sha=physical_contract_sha,
        physical_shoulder_recipe=physical_shoulder_recipe,
        generator_sha=generator_sha,
        authorized_overrides_v2=overrides,
        authorized_overrides_v2_sha256=override_sha,
    )

    assert first == second
    assert first_mapping == second_mapping
    assembly = first["D38999_ASSEMBLY_CONTROL_V1.usda"]
    assert assembly.count("custom int kcg:eventOnsetProxyVersion = 2") == 1
    assert assembly.count('kcg:collisionRole = "coupling_nut_grasp_collision"') == 1
    assert "hard_socket_entries_61" not in assembly
    assert "pin_barriers_61" not in assembly
    assert "custom bool kcg:newBackshellEnabled = 0" in assembly
    assert assembly.count("PhysxCollisionAPI") == 278
    assert assembly.count("float physxCollision:contactOffset = 1e-05") == 278
    assert assembly.count("float physxCollision:restOffset = 0") == 278
    assert assembly.count("custom bool kcg:dynamicCollisionEnabled = 0") == 62
    assert assembly.count('kcg:collisionRole = "hard_nut_body_shoulder"') == 8
    assert assembly.count('def Cylinder "AnalyticCap"') == 2
    assert assembly.count('def Sphere "AnalyticSphere_') == 6
    current_assembly = (
        output_root / "D38999_ASSEMBLY_CONTROL_V1.usda"
    ).read_text(encoding="utf-8")
    assert assembly == current_assembly
    assert first_mapping["static_generation_checks"]["traceable_pair_count"] == 61
    assert first_mapping["static_generation_checks"]["cross_label_effect_count"] == 0

    for filename, expected_sha in generator.EXPECTED_PRESERVED_OUTPUT_SHA256.items():
        assert generator._sha256(output_root / filename) == expected_sha
