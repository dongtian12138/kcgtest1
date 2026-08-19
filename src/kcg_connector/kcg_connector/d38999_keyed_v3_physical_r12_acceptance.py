"""Pure-CPU acceptance loader for the deterministic r12 structural repair."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from kcg_connector.d38999_keyed_v2_physical_acceptance import (
    NOMINAL_R7_EVENT_ORDER,
    PhysicalAcceptanceMatrix,
    load_physical_acceptance_matrix,
)
from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
    R12_ACCEPTANCE_PATH,
    R12_ASSET_NAME,
    R12_SUCCESSOR_REVISION,
    load_r12_physical_model_contract,
)


R12_ACCEPTANCE_SCHEMA = "kcg_d38999_keyed_v3_physical_acceptance_r12_v1"


def build_r12_acceptance_document(
    r11_document: Mapping[str, Any],
) -> dict[str, Any]:
    document = deepcopy(dict(r11_document))
    document["schema_version"] = R12_ACCEPTANCE_SCHEMA
    document["model_contract"].update(
        {
            "path": "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml",
            "required_successor_revision": R12_SUCCESSOR_REVISION,
            "required_successor_asset": R12_ASSET_NAME,
        }
    )
    a2 = document["phase_release"]["A2_result_contract"]
    a2.update(
        {
            "schema_version": "kcg_d38999_keyed_physical_r12_resolved_readback_v1",
            "contract_revision": "d38999_keyed_v3_r12_family_algebra_v1",
        }
    )
    a2["expected_row_counts"]["collider_rows"] = 14761
    return document


def load_r12_physical_acceptance_matrix(
    path: Path | str = R12_ACCEPTANCE_PATH,
) -> PhysicalAcceptanceMatrix:
    acceptance_path = Path(path).expanduser().resolve()
    r11 = load_physical_acceptance_matrix()
    expected = build_r12_acceptance_document(r11.document)
    actual = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("r12 acceptance differs from frozen r11 thresholds plus r12 identity")
    model = load_r12_physical_model_contract()
    if actual["model_contract"] != {
        "path": "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml",
        "required_successor_revision": model.document["identity"]["successor_revision"],
        "required_successor_asset": model.document["identity"]["recommended_asset_name"],
    }:
        raise ValueError("r12 acceptance/model identity does not close")
    if actual["benches"] != r11.document["benches"]:
        raise ValueError("r12 P1-P14 thresholds or controllers changed")
    return PhysicalAcceptanceMatrix(path=acceptance_path, document=actual)


__all__ = [
    "NOMINAL_R7_EVENT_ORDER",
    "R12_ACCEPTANCE_SCHEMA",
    "build_r12_acceptance_document",
    "load_r12_physical_acceptance_matrix",
]
