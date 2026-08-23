#!/usr/bin/env python3
"""Build deterministic static material-boundary evidence for both study objects.

This program is deliberately offline.  It reads mesh/CAD lineage and authored
solid-role metadata only; it does not import Isaac, load PhysX, read contacts,
or mutate any geometry source.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.material_boundary import (
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.object_contract import load_object_contract
from kcg_connector.grasp.robust.positive_solid_union import (
    bind_positive_usda_component_ids,
    certify_positive_solid_component_union,
    parse_usda_gprim_solid_roles,
)


SCHEMA_VERSION = "carts_object_material_boundary_certificate_v1"
CLAIM_SCOPE = "STATIC_MESH_LOCAL_MATERIAL_ONLY_NO_ROUTE_SCENE_OR_DYNAMICS"
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"
CURRENT_ROLE_KIND = "USD_GPRIM_KCG_POSITIVE_VOLUME_TRUE_V1"
TRANSFER_ROLE_KIND = "SUPPLIER_STEP_SINGLE_SOLID_V1"

_REFERENCE_PATH = re.compile(
    r'custom string kcg:referenceSourcePath\s*=\s*"([^"]+)"'
)
_REFERENCE_SHA256 = re.compile(
    r'custom string kcg:referenceSourceSha256\s*=\s*"([0-9a-f]{64})"'
)
_REFERENCE_TARGET = re.compile(
    r'prepend references\s*=\s*@([^@]+)@<([^>]+)>'
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--object-contract", type=Path, required=True)
    parser.add_argument("--current-output", type=Path, required=True)
    parser.add_argument("--transfer-output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_match(pattern: re.Pattern[str], text: str, label: str) -> str:
    values = pattern.findall(text)
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"{label} must occur exactly once")
    return values[0]


def _reference_target(text: str) -> tuple[str, str]:
    values = _REFERENCE_TARGET.findall(text)
    if len(values) != 1:
        raise ValueError("source-stage direct reference must occur exactly once")
    return str(values[0][0]), str(values[0][1])


def _replace_prefix(path: str, source: str, target: str) -> str:
    if not source.startswith("/") or not target.startswith("/"):
        raise ValueError("USD prefixes must be absolute prim paths")
    if path == source:
        return target
    if not path.startswith(source + "/"):
        raise ValueError(f"source component {path!r} is outside {source!r}")
    return target + path[len(source) :]


def _write_json(path: Path, document: Mapping[str, object]) -> str:
    payload = (
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return _sha256(path)


def _current_certificate(
    *,
    repository: Path,
    object_contract_path: Path,
) -> dict[str, object]:
    loaded = load_object_contract(
        object_contract_path,
        object_id=CURRENT_OBJECT,
        repository_root=repository,
    )
    geometry = loaded.geometry_contract
    mesh_path = (repository / str(geometry["path"])).resolve()
    stage_path = (repository / str(geometry["source_stage"])).resolve()
    stage_text = stage_path.read_text(encoding="utf-8")
    declared_source_path = _one_match(
        _REFERENCE_PATH, stage_text, "kcg:referenceSourcePath"
    )
    declared_source_sha256 = _one_match(
        _REFERENCE_SHA256, stage_text, "kcg:referenceSourceSha256"
    )
    reference_asset, source_prefix = _reference_target(stage_text)
    source_usda = (repository / declared_source_path).resolve()
    referenced_usda = (stage_path.parent / reference_asset).resolve()
    if source_usda != referenced_usda:
        raise ValueError("source-stage metadata and USD reference resolve differently")
    actual_source_sha256 = _sha256(source_usda)
    if actual_source_sha256 != declared_source_sha256:
        raise ValueError("source USDA SHA-256 differs from source-stage declaration")

    with np.load(mesh_path, allow_pickle=False) as archive:
        required = {"vertices_m", "faces", "source_prim_paths"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"current-object NPZ is missing arrays: {missing}")
        vertices = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        composed_component_ids = tuple(
            str(value) for value in archive["source_prim_paths"]
        )
    source_subtree = str(geometry["source_subtree"])
    if not source_subtree.endswith("/LoosePlug"):
        raise ValueError("current source subtree is not the loose-plug subtree")
    composed_prefix = source_subtree[: -len("/LoosePlug")]
    source_component_ids = tuple(
        _replace_prefix(value, composed_prefix, source_prefix)
        for value in composed_component_ids
    )
    role_inventory = parse_usda_gprim_solid_roles(source_usda)
    positive_ids = bind_positive_usda_component_ids(
        source_component_ids,
        role_inventory,
    )
    certificate = certify_positive_solid_component_union(
        vertices,
        faces,
        source_asset_sha256=loaded.model.provenance.source_sha256,
        source_component_ids=source_component_ids,
        positive_solid_component_ids=positive_ids,
        role_authority_kind=CURRENT_ROLE_KIND,
        role_authority_sha256=actual_source_sha256,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "object_id": CURRENT_OBJECT,
        "representation": "POSITIVE_SOLID_COMPONENT_UNION",
        "source_asset": {
            "path": str(mesh_path.relative_to(repository)),
            "sha256": loaded.model.provenance.source_sha256,
        },
        "role_authority": {
            "kind": CURRENT_ROLE_KIND,
            "path": str(source_usda.relative_to(repository)),
            "sha256": actual_source_sha256,
            "component_count": len(source_component_ids),
            "required_fields": [
                "kcg:positiveVolume=true",
                "kcg:closedManifold=true",
            ],
        },
        "physics_loaded": False,
        "collision_or_contact_truth_read": False,
        "source_modified": False,
        "formal_material_boundary_eligible": True,
        "certificate": certificate.as_dict(),
    }


def _transfer_certificate(
    *,
    repository: Path,
    object_contract_path: Path,
    raw_contract: Mapping[str, object],
) -> dict[str, object]:
    loaded = load_object_contract(
        object_contract_path,
        object_id=TRANSFER_OBJECT,
        repository_root=repository,
    )
    objects = raw_contract.get("objects")
    if not isinstance(objects, Mapping) or not isinstance(
        objects.get(TRANSFER_OBJECT), Mapping
    ):
        raise ValueError("transfer object document is missing")
    transfer = objects[TRANSFER_OBJECT]
    assert isinstance(transfer, Mapping)
    original = transfer.get("original_cad")
    if not isinstance(original, Mapping):
        raise ValueError("transfer original_cad contract is missing")
    original_path = (repository / str(original["path"])).resolve()
    audit_path = (repository / str(original["geometry_audit"])).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "kcg_te_j35_step_geometry_audit_v1"
        or audit.get("scope") != "supplier_step_topology_only"
        or audit.get("isaac_sim_loaded") is not False
    ):
        raise ValueError("TE STEP geometry audit contract changed")
    products = audit.get("products")
    plug = products.get("plug") if isinstance(products, Mapping) else None
    if not isinstance(plug, Mapping):
        raise ValueError("TE STEP plug audit is missing")
    if (
        int(plug.get("solid_count", -1)) != 1
        or Path(str(plug.get("path", ""))).resolve() != original_path
        or str(plug.get("sha256", "")) != _sha256(original_path)
    ):
        raise ValueError("TE STEP plug is not the one hash-bound supplier solid")
    certificate = certify_single_embedded_material_boundary(
        loaded.model.mesh.vertices_m,
        loaded.model.mesh.faces,
    )
    mesh_path = Path(loaded.model.provenance.source_path).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "object_id": TRANSFER_OBJECT,
        "representation": "SINGLE_EMBEDDED_MATERIAL_BOUNDARY",
        "source_asset": {
            "path": str(mesh_path.relative_to(repository)),
            "sha256": loaded.model.provenance.source_sha256,
        },
        "role_authority": {
            "kind": TRANSFER_ROLE_KIND,
            "path": str(audit_path.relative_to(repository)),
            "sha256": _sha256(audit_path),
            "original_step_path": str(original_path.relative_to(repository)),
            "original_step_sha256": _sha256(original_path),
            "solid_count": 1,
        },
        "physics_loaded": False,
        "collision_or_contact_truth_read": False,
        "source_modified": False,
        "formal_material_boundary_eligible": True,
        "certificate": asdict(certificate),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    repository = arguments.repository.resolve()
    contract_path = arguments.object_contract
    if not contract_path.is_absolute():
        contract_path = (repository / contract_path).resolve()
    raw_contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw_contract, Mapping):
        raise ValueError("object contract is not a mapping")
    current = _current_certificate(
        repository=repository,
        object_contract_path=contract_path,
    )
    current_sha256 = _write_json(arguments.current_output.resolve(), current)
    print(
        json.dumps(
            {
                "object_id": CURRENT_OBJECT,
                "output": str(arguments.current_output.resolve()),
                "output_sha256": current_sha256,
                "certificate_sha256": current["certificate"]["certificate_sha256"],  # type: ignore[index]
            },
            sort_keys=True,
        ),
        flush=True,
    )
    transfer = _transfer_certificate(
        repository=repository,
        object_contract_path=contract_path,
        raw_contract=raw_contract,
    )
    transfer_sha256 = _write_json(arguments.transfer_output.resolve(), transfer)
    print(
        json.dumps(
            {
                "object_id": TRANSFER_OBJECT,
                "output": str(arguments.transfer_output.resolve()),
                "output_sha256": transfer_sha256,
                "certificate_sha256": transfer["certificate"]["certificate_sha256"],  # type: ignore[index]
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
