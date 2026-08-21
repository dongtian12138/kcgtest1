#!/usr/bin/env python3

"""Build exact-source three-finger PAD arrays for CARTS-Grasp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from kcg_connector.grasp.robust.terminal_pad_source import (  # noqa: E402
    METHOD_ID,
    extract_exact_terminal_pad_source,
)


SCHEMA = "CARTS_EXACT_SOURCE_TERMINAL_PAD_V1"
SPECS = (
    (
        1,
        "f1Link3",
        "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
    ),
    (
        2,
        "f2Link2",
        "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
    ),
    (
        3,
        "f3Link3",
        "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
    ),
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(repository_root: Path, output_dir: Path) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)

    links: list[dict[str, object]] = []
    for finger_number, link_name, source_sha256 in SPECS:
        source = root / f"src/iiwa_description/meshes/hand/{link_name}.STL"
        pad = extract_exact_terminal_pad_source(
            link_name=link_name,
            source_stl_path=source,
            source_stl_sha256=source_sha256,
            expected_source_face_count=14192,
            expected_pad_face_count=2479,
            expected_pad_vertex_count=1250,
            expected_component_area_rank=1,
        )
        name = f"{link_name}_PAD_BODY_raw_source_local_m.npz"
        path = destination / name
        np.savez_compressed(
            path,
            points_local_m=np.asarray(pad.points_local_m, dtype="<f8"),
            faces=np.asarray(pad.faces, dtype="<i8"),
            source_face_indices=np.asarray(pad.source_face_indices, dtype="<i8"),
        )
        diagnostics = dict(pad.audit)
        diagnostics["source_path"] = str(source.relative_to(root))
        links.append(
            {
                "finger_number": finger_number,
                "link_name": link_name,
                "source_mesh": str(source.relative_to(root)),
                "source_mesh_sha256": source_sha256,
                "pad_source_arrays": name,
                "pad_source_arrays_sha256": _sha256(path),
                "diagnostics": {
                    **diagnostics,
                    "pad_component_is_winding_consistent": True,
                    "exact_source_face_ordinal_lineage_complete": True,
                    "dynamic_use_allowed": False,
                },
            }
        )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "status": "STATIC_EXACT_SOURCE_PAD_LINEAGE_VERIFIED",
        "method_id": METHOD_ID,
        "source_authority": "AUTHORED_HAND_STL_GEOMETRY",
        "semantic_authority": "USER_CONFIRMED_HAND_GEOMETRY_SEMANTICS",
        "local_points_unit": "metre",
        "dynamic_use_allowed": False,
        "online_control_role_truth_allowed": False,
        "coordinate_tolerance_used": False,
        "source_vertex_changed": False,
        "links": links,
    }
    _write_json(destination / "TERMINAL_PAD_SOURCE_MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    manifest = build(
        Path(arguments.repository_root),
        Path(arguments.output_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
