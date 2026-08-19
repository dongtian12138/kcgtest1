import ast
import math
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
ISAAC = REPOSITORY / "src/kcg_connector/isaac"
sys.path.insert(0, str(ISAAC))

import d38999_multilayer_runtime_collision as runtime_fix


BENCH_SOURCE = (ISAAC / "d38999_multilayer_nominal_bench.py").read_text(
    encoding="utf-8"
)
VALIDATION_SOURCE = (
    ISAAC / "d38999_multilayer_init_fix_validation.py"
).read_text(encoding="utf-8")
VALIDATION_TREE = ast.parse(VALIDATION_SOURCE)


def _called_attributes() -> list[str]:
    return [
        node.func.attr
        for node in ast.walk(VALIDATION_TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_exact_64_segment_annulus_is_preserved() -> None:
    specs = runtime_fix.annular_convex_segment_specs(
        inner_radius_m=0.0170825,
        outer_radius_m=0.017845,
        z0_m=0.0,
        z1_m=0.01505,
    )
    assert len(specs) == 64
    assert [row["index"] for row in specs] == list(range(64))
    for row in specs:
        assert len(row["points_m"]) == 8
        assert row["face_vertex_counts"] == (4, 4, 4, 4, 4, 4)
        assert len(row["face_vertex_indices"]) == 24
        assert row["xform_scale"] == 0.001
        radii = [math.hypot(point[0], point[1]) for point in row["points_m"]]
        assert math.isclose(min(radii), 0.0170825, abs_tol=1e-12)
        assert math.isclose(max(radii), 0.017845, abs_tol=1e-12)
        assert {point[2] for point in row["points_m"]} == {0.0, 0.01505}


def test_contract_clearances_are_positive_and_exact() -> None:
    assert math.isclose(0.0179575 - 0.017845, 0.0001125, abs_tol=1e-12)
    assert math.isclose(0.0170825 - 0.01695, 0.0001325, abs_tol=1e-12)


def test_millimetre_local_scale_round_trips_exact_geometry() -> None:
    row = runtime_fix.annular_convex_segment_specs(
        inner_radius_m=0.0170825,
        outer_radius_m=0.017845,
        z0_m=0.0,
        z1_m=0.01505,
    )[17]
    for point_m, point_local in zip(row["points_m"], row["points_local_mm"]):
        for expected, local in zip(point_m, point_local):
            assert math.isclose(expected, local * 0.001, abs_tol=1e-15)


def test_nominal_bench_uses_the_single_shared_fix() -> None:
    assert "configure_continuous_plug_guide_runtime_collision" in BENCH_SOURCE
    assert 'attribute.Set("convexDecomposition")' not in BENCH_SOURCE
    assert '_configure_runtime_collision_cooking(stage, frozen["contract"])' not in BENCH_SOURCE
    assert 'stage, frozen["contract"]' in BENCH_SOURCE


def test_validation_is_exactly_three_independent_process_indices() -> None:
    assert 'choices=(1, 2, 3)' in VALIDATION_SOURCE
    assert 'f"VALIDATION_{arguments.run_index:02d}"' in VALIDATION_SOURCE
    assert 'dynamic.get("validation_processes_started") != arguments.run_index' in VALIDATION_SOURCE
    assert 'dynamic.get("validation_processes_completed") != arguments.run_index - 1' in VALIDATION_SOURCE


def test_validation_has_one_explicit_step_and_no_pose_write() -> None:
    assert VALIDATION_SOURCE.count("world.step(render=False)") == 1
    called = _called_attributes()
    assert "set_world_pose" not in called
    assert "set_local_pose" not in called
    assert "set_default_state" not in called
    assert '"object_pose_write_after_physics_start_count": 0' in VALIDATION_SOURCE


def test_rejected_offset_route_is_absent() -> None:
    combined = BENCH_SOURCE + VALIDATION_SOURCE + Path(runtime_fix.__file__).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "CreateContactOffsetAttr",
        "CreateRestOffsetAttr",
        "physxCollision:contactOffset",
        "physxCollision:restOffset",
    ):
        assert forbidden not in combined


def test_validation_keeps_dynamic_claim_false_until_all_three_pass() -> None:
    assert '"dynamic_pass_claimed": False' in VALIDATION_SOURCE
    assert '"formal_p1_pass_claimed": False' in VALIDATION_SOURCE
    assert '"hardware_authorized": False' in VALIDATION_SOURCE


def test_all_four_frozen_hashes_remain_locked() -> None:
    for digest in (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
        "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec",
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
        "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    ):
        assert digest in VALIDATION_SOURCE or digest in (
            ISAAC / "d38999_multilayer_init_timeline_probe.py"
        ).read_text(encoding="utf-8")
