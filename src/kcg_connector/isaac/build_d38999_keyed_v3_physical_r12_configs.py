#!/usr/bin/env python3

"""Materialize the deterministic r12 model and acceptance YAML documents."""

from __future__ import annotations

from pathlib import Path

import yaml

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    load_physical_model_contract,
)
from kcg_connector.d38999_keyed_v2_physical_acceptance import (
    load_physical_acceptance_matrix,
)
from kcg_connector.d38999_keyed_v3_physical_r12_acceptance import (
    build_r12_acceptance_document,
)
from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
    R12_ACCEPTANCE_PATH,
    R12_CONTRACT_PATH,
    build_r12_document,
)


def _dump(path: Path, document: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite r12 config: {path}")
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    r11 = load_physical_model_contract()
    _dump(R12_CONTRACT_PATH, build_r12_document(r11.document))
    r11_acceptance = load_physical_acceptance_matrix()
    _dump(
        R12_ACCEPTANCE_PATH,
        build_r12_acceptance_document(r11_acceptance.document),
    )
    print(f"created={R12_CONTRACT_PATH}")
    print(f"created={R12_ACCEPTANCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
