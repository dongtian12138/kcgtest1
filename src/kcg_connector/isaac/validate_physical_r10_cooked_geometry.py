#!/usr/bin/env python3

"""Fail closed on PhysX-cooked geometry drift in the physical-r10 asset.

The validator asks the installed PhysX cooker for every non-analytic connector
collider and compares the cooked hull to the authored millimetre-local hull.
It also records any thickness-adjustment warning emitted during cooking.  It
writes no artifact and computes no file fingerprint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Sequence


PASS_BANNER = "ISAAC PHYSICAL R10 COOKED GEOMETRY PASSED"
FAIL_BANNER = "ISAAC PHYSICAL R10 COOKED GEOMETRY FAILED"
ADJUSTED_THICKNESS_TEXT = "adjusted the thickness of a very thin or very small mesh"
CPU_FALLBACK_TEXT = "ConvexMeshCookingTask: failed to cook GPU-compatible mesh"


def _emit(value: Any) -> None:
    text = value if isinstance(value, str) else json.dumps(
        value, allow_nan=False, sort_keys=True
    )
    os.write(1, (text + "\n").encode("utf-8"))


def _bounds(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(min(float(point[axis]) for point in points) for axis in range(3)),
        tuple(max(float(point[axis]) for point in points) for axis in range(3)),
    )


def _maximum_bound_error(
    left: tuple[tuple[float, ...], tuple[float, ...]],
    right: tuple[tuple[float, ...], tuple[float, ...]],
) -> float:
    return max(
        abs(left[side][axis] - right[side][axis])
        for side in range(2)
        for axis in range(3)
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _bidirectional_vertex_error(
    authored: Sequence[Sequence[float]], cooked: Sequence[Sequence[float]]
) -> float:
    return max(
        max(min(_distance(point, other) for other in cooked) for point in authored),
        max(min(_distance(point, other) for other in authored) for point in cooked),
    )


def _run() -> dict[str, Any]:
    import carb.logging
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdUtils

    from kcg_connector.d38999_keyed_v2_a2_readback_result import (
        _authorized_asset_path,
        _trusted_collider_inventory,
        validate_a2_composed_asset_release,
    )
    from kcg_connector.d38999_keyed_v2_physical_model_contract import (
        load_physical_model_contract,
    )

    model = load_physical_model_contract()
    asset_path = _authorized_asset_path(model)
    static_release = validate_a2_composed_asset_release(asset_path, model=model)
    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError("authorized r10 asset could not be opened")
    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    expected = _trusted_collider_inventory(model)
    mesh_paths = sorted(
        value["prim_path"]
        for value in expected.values()
        if value["typeName"] == "Mesh"
    )
    expected_mesh_count = static_release.collider_row_count - sum(
        value["typeName"] != "Mesh" for value in expected.values()
    )
    if len(mesh_paths) != expected_mesh_count:
        raise RuntimeError("trusted r10 mesh inventory does not close")

    contract = model.document["convex_cooking_representation"]
    scale = float(contract["mesh_uniform_scale_xyz"][0])
    bound_tolerance_m = float(contract["cooked_bounds_abs_tolerance_m"])
    vertex_tolerance_m = bound_tolerance_m
    adjusted_messages: list[str] = []
    cpu_fallback_paths: list[str] = []
    logging = carb.logging.acquire_logging()

    def on_log(
        source: str,
        level: int,
        filename: str,
        line_number: int,
        message: str,
    ) -> None:
        del source, level, filename, line_number
        if ADJUSTED_THICKNESS_TEXT in message:
            adjusted_messages.append(str(message))
        if CPU_FALLBACK_TEXT in message and " Prim " in message:
            cpu_fallback_paths.append(str(message).rsplit(" Prim ", 1)[1].strip())

    logger_handle = logging.add_logger(on_log)
    cooking = get_physx_cooking_interface()
    cooking.release_local_mesh_cache()
    failed: list[dict[str, Any]] = []
    maximum_bound_error_m = 0.0
    maximum_vertex_error_m = 0.0
    cooked_vertex_count_mismatch_count = 0
    result_failure_count = 0
    try:
        for index, path in enumerate(mesh_paths, start=1):
            prim = stage.GetPrimAtPath(path)
            mesh = UsdGeom.Mesh(prim)
            authored = [
                tuple(float(value) for value in point)
                for point in mesh.GetPointsAttr().Get()
            ]
            callback: dict[str, Any] = {}

            def on_result(result: Any, convexes: list[Any]) -> None:
                callback["result"] = result
                callback["convexes"] = convexes

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
                run_asynchronously=False,
                on_result=on_result,
            )
            valid = callback.get("result") == PhysxCollisionRepresentationResult.RESULT_VALID
            convexes = callback.get("convexes", [])
            if not valid or len(convexes) != 1:
                result_failure_count += 1
                if len(failed) < 20:
                    failed.append(
                        {
                            "prim_path": path,
                            "failure": "invalid_cooking_result_or_convex_count",
                            "convex_count": len(convexes),
                        }
                    )
                continue
            cooked = [
                (float(vertex.x), float(vertex.y), float(vertex.z))
                for vertex in convexes[0].vertices
            ]
            bound_error_m = _maximum_bound_error(
                _bounds(authored), _bounds(cooked)
            ) * scale
            vertex_error_m = _bidirectional_vertex_error(authored, cooked) * scale
            vertex_count_matches = len(authored) == len(cooked)
            maximum_bound_error_m = max(maximum_bound_error_m, bound_error_m)
            maximum_vertex_error_m = max(maximum_vertex_error_m, vertex_error_m)
            cooked_vertex_count_mismatch_count += int(not vertex_count_matches)
            if (
                bound_error_m > bound_tolerance_m
                or vertex_error_m > vertex_tolerance_m
                or not vertex_count_matches
            ) and len(failed) < 20:
                failed.append(
                    {
                        "prim_path": path,
                        "failure": "cooked_hull_differs_from_authored_hull",
                        "authored_vertex_count": len(authored),
                        "cooked_vertex_count": len(cooked),
                        "cooked_polygon_count": len(convexes[0].polygons),
                        "maximum_bound_error_m": bound_error_m,
                        "maximum_bidirectional_vertex_error_m": vertex_error_m,
                    }
                )
            if index % 1000 == 0:
                os.write(
                    2,
                    f"cooked_geometry_progress={index}/{len(mesh_paths)}\n".encode(),
                )
    finally:
        logging.remove_logger(logger_handle)

    failed_collider_count = sum(
        1
        for row in failed
        if row["failure"] != "invalid_cooking_result_or_convex_count"
    )
    # The retained list is capped; the exact independent counters below carry
    # the fail-closed verdict even if more than 20 examples fail.
    expected_cpu_fallback_paths = sorted(
        contract["cpu_collision_fallback_allowed_only_for_oblong_keyway_prisms"]
    )
    resolved_cpu_fallback_paths = sorted(cpu_fallback_paths)
    passed = bool(
        result_failure_count == 0
        and cooked_vertex_count_mismatch_count == 0
        and maximum_bound_error_m <= bound_tolerance_m
        and maximum_vertex_error_m <= vertex_tolerance_m
        and len(adjusted_messages)
        <= int(contract["adjusted_thickness_warning_count_allowed"])
        and resolved_cpu_fallback_paths == expected_cpu_fallback_paths
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "contract_revision": model.document["identity"]["successor_revision"],
        "asset_path": str(Path(asset_path)),
        "mesh_collider_count": len(mesh_paths),
        "cooking_result_failure_count": result_failure_count,
        "cooked_vertex_count_mismatch_count": cooked_vertex_count_mismatch_count,
        "maximum_cooked_bound_error_m": maximum_bound_error_m,
        "maximum_bidirectional_cooked_vertex_error_m": maximum_vertex_error_m,
        "bound_tolerance_m": bound_tolerance_m,
        "adjusted_thickness_warning_count": len(adjusted_messages),
        "adjusted_thickness_warning_examples": adjusted_messages[:20],
        "expected_cpu_collision_fallback_paths": expected_cpu_fallback_paths,
        "resolved_cpu_collision_fallback_paths": resolved_cpu_fallback_paths,
        "unexpected_cpu_collision_fallback_paths": sorted(
            set(resolved_cpu_fallback_paths) - set(expected_cpu_fallback_paths)
        ),
        "missing_cpu_collision_fallback_paths": sorted(
            set(expected_cpu_fallback_paths) - set(resolved_cpu_fallback_paths)
        ),
        "retained_failure_example_count": len(failed),
        "retained_failure_examples": failed,
        "failed_collider_example_count": failed_collider_count,
        "static_a2_collider_row_count": static_release.collider_row_count,
        "file_fingerprints_computed": False,
        "passed": passed,
    }


def main() -> int:
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    try:
        report = _run()
        status = 0 if report["passed"] else 1
    except BaseException as error:
        traceback.print_exc()
        report = {
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "file_fingerprints_computed": False,
            "passed": False,
        }
    _emit(report)
    _emit(PASS_BANNER if status == 0 else FAIL_BANNER)
    application.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
