import json
import numpy as np
import pytest

from kcg_connector.d38999_cad_registration import (
    fixed_camera_model,
    proxy_cad_points,
    render_points,
    transform_points,
)
from kcg_connector.d38999_inhand_multiview import (
    c2_action_state12,
    matrix_pose,
    pose_matrix,
)
from kcg_connector.postgrasp_shadow_estimator import (
    ALL_THRESHOLDS_CANDIDATE,
    _ResidualProblem,
    FormalArchiveError,
    FormalView,
    estimate_grouped_views,
    estimate_joint_c2,
    estimate_postgrasp_T_HP,
    load_formal_archive,
    write_formal_archive,
)


def _camera_pose(camera):
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rows.T
    matrix[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return matrix


def _synthetic_views(state, count=2):
    plug_cad, receptacle_cad = proxy_cad_points()
    hp, rp = state[:6], state[6:]
    views = []
    eyes = ((0.55, -0.85, 0.72), (0.30, -0.85, 0.72))
    for index, eye in enumerate(eyes[:count]):
        camera = fixed_camera_model(
            eye=eye,
            target=(0.535, -0.0125, 0.231),
            resolution=(320, 180),
        )
        T_WC = _camera_pose(camera)
        T_WH = np.eye(4)
        T_WP = T_WH @ pose_matrix(hp)
        T_WR = T_WP @ np.linalg.inv(pose_matrix(rp))
        observation = render_points(
            camera,
            (
                transform_points(plug_cad, matrix_pose(T_WP)),
                transform_points(receptacle_cad, matrix_pose(T_WR)),
            ),
        )
        views.append(
            FormalView(
                view_id=f"V{index}",
                timestamp_utc="2026-08-15T00:00:00Z",
                rgb=observation["rgb"],
                depth=observation["depth"],
                camera=camera,
                T_WH=T_WH,
                T_HC=np.linalg.inv(T_WH) @ T_WC,
                T_WC=T_WC,
            )
        )
    return views


def test_semantic_permutation_does_not_change_estimator(tmp_path):
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    first = estimate_joint_c2(views, state)
    archive = tmp_path / "archive"
    write_formal_archive(archive, views)
    # Pollute every formal view directory with posthoc semantic sidecars.
    for label in ("plug", "receptacle", "random"):
        (archive / "V0" / f"posthoc_{label}_semantic.npy").write_bytes(b"fake")
        (archive / "V1" / f"posthoc_{label}_semantic.npy").write_bytes(b"fake")
    loaded = load_formal_archive(archive)
    second = estimate_joint_c2(loaded, state)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["shadow_authorized"] is False
    assert first["control_authorized"] is False


def test_archive_roundtrip_rejects_wrong_role(tmp_path):
    views = _synthetic_views(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    )
    archive = tmp_path / "archive"
    write_formal_archive(archive, views)
    manifest_path = archive / "formal_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["role"] = "truth_restore"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FormalArchiveError):
        load_formal_archive(archive)


def test_c2_nonzero_rx_ry_is_matrix_right_action():
    from scipy.spatial.transform import Rotation

    state = np.array([0.1, -0.2, 0.3, 0.3, -0.2, 0.5, 0.01, 0.02, -0.03, -0.15, 0.1, 0.2])
    acted = c2_action_state12(state)
    rz_pi = Rotation.from_euler("z", np.pi).as_matrix()
    for offset in (0, 6):
        expected = pose_matrix(state[offset:offset + 6])
        expected[:3, :3] = expected[:3, :3] @ rz_pi
        actual = pose_matrix(acted[offset:offset + 6])
        assert np.max(np.abs(actual - expected)) < 1.0e-10


def test_c2_projection_invariance_cost_nonzero_tilt():
    plug_cad, receptacle_cad = proxy_cad_points()
    state = np.array([0.0, 0.0, 0.0, 0.3, -0.2, 0.4, 0.0, 0.0, -0.02, -0.1, 0.2, 0.3])
    acted = c2_action_state12(state)
    views = _synthetic_views(state)
    problem_a = _ResidualProblem(
        views, plug_cad, receptacle_cad, state, include_prior=False
    )
    problem_b = _ResidualProblem(
        views, plug_cad, receptacle_cad, acted, include_prior=False
    )
    cost_a = float(np.sum(problem_a.residual(state) ** 2))
    cost_b = float(np.sum(problem_b.residual(acted) ** 2))
    assert abs(cost_a - cost_b) < 1.0e-9


def test_fixed_world_view_archive_roundtrip_and_two_independent_views(tmp_path):
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    fixed = views[1]
    fixed = FormalView(
        view_id="FIXED_WORLD_CAMERA_V0",
        timestamp_utc=fixed.timestamp_utc,
        rgb=fixed.rgb,
        depth=fixed.depth,
        camera=fixed.camera,
        T_WH=fixed.T_WH,
        T_WC=fixed.T_WC,
        T_HC=None,
        group="fixed_world_camera_views",
        extrinsic_source="fixed_camera_config_T_WC",
    )
    views[1] = fixed
    archive = tmp_path / "archive"
    write_formal_archive(archive, views)
    loaded = load_formal_archive(archive)
    assert loaded[1].T_HC is None
    assert loaded[1].extrinsic_source == "fixed_camera_config_T_WC"
    result = estimate_postgrasp_T_HP(views, state)
    assert result["T_HP_independent_view_count"] == 2
    assert result["T_HP_multiview"] is True
    assert result["control_authorized"] is False
    assert result["covariance_calibration_status"] == "UNVALIDATED"


def test_unknown_independent_group_fails_closed():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    views[0] = FormalView(
        view_id="BAD",
        timestamp_utc=views[0].timestamp_utc,
        rgb=views[0].rgb,
        depth=views[0].depth,
        camera=views[0].camera,
        T_WH=views[0].T_WH,
        T_WC=views[0].T_WC,
        T_HC=views[0].T_HC,
        group="unknown_group",
    )
    with pytest.raises(FormalArchiveError):
        estimate_postgrasp_T_HP(views, state)


def test_comoving_wrist_views_are_not_counted_as_independent_t_hp_views():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    result = estimate_postgrasp_T_HP(views, state)
    assert result["T_HP_independent_view_count"] == 1
    assert result["T_HP_multiview"] is False
    assert len(result["T_HP_ignored_comoving_view_ids"]) == 1


def test_missing_surface_gets_bounded_support_not_unbounded_depth():
    state = np.array(
        [0.52, -0.21, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0]
    )
    camera = fixed_camera_model(
        eye=(0.55, -0.21, 0.35),
        target=(0.52, -0.21, 0.25),
        resolution=(320, 180),
    )
    plug_cad, receptacle_cad = proxy_cad_points()
    observation = render_points(
        camera,
        (transform_points(plug_cad, matrix_pose(pose_matrix(state[:6]))),),
    )
    punched = observation["depth"].copy()
    punched[130:, 60:260] += 0.05  # ring's lower arc now 50 mm deeper (a bore)
    view = FormalView(
        view_id="V0",
        timestamp_utc="2026-08-15T00:00:00Z",
        rgb=observation["rgb"],
        depth=punched,
        camera=camera,
        T_WH=np.eye(4),
        T_WC=_camera_pose(camera),
        T_HC=_camera_pose(camera),
    )
    problem = _ResidualProblem(
        [view],
        plug_cad,
        receptacle_cad,
        state,
        endpoints=("plug",),
        missing_surface_margin_m=0.015,
        missing_surface_support=2.0,
    )
    residual = problem.residual(state)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    # Without the missing-surface path these points would contribute
    # -0.05/0.00075 ~= -66 residual units each and blow the RMS up.
    problem_unbounded = _ResidualProblem(
        [view],
        plug_cad,
        receptacle_cad,
        state,
        endpoints=("plug",),
        missing_surface_margin_m=1.0e9,
        missing_surface_support=2.0,
    )
    rms_unbounded = float(
        np.sqrt(np.mean(problem_unbounded.residual(state) ** 2))
    )
    assert rms < 0.5 * rms_unbounded
    diagnostics = problem.last_plug_support[0]
    assert diagnostics["missing_surface_fraction"] > 0.0
    assert diagnostics["missing_surface_fraction"] < 1.0


def test_second_inhand_camera_group_counts_as_independent_t_hp_view():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    secondary = views[1]
    views[1] = FormalView(
        view_id="PALM_SECONDARY_V0",
        timestamp_utc=secondary.timestamp_utc,
        rgb=secondary.rgb,
        depth=secondary.depth,
        camera=secondary.camera,
        T_WH=secondary.T_WH,
        T_WC=secondary.T_WC,
        T_HC=secondary.T_HC,
        group="postgrasp_second_inhand_camera_views",
        extrinsic_source="T_HC_calibrated",
    )
    result = estimate_postgrasp_T_HP(views, state)
    assert result["T_HP_independent_view_count"] == 2
    assert result["T_HP_multiview"] is True
    assert result["T_HP_ignored_comoving_view_ids"] == []


def test_second_inhand_camera_duplicates_collapse_to_one_representative():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    for index, view in enumerate(views):
        views[index] = FormalView(
            view_id=f"PALM_SECONDARY_V{index}",
            timestamp_utc=view.timestamp_utc,
            rgb=view.rgb,
            depth=view.depth,
            camera=view.camera,
            T_WH=view.T_WH,
            T_WC=view.T_WC,
            T_HC=view.T_HC,
            group="postgrasp_second_inhand_camera_views",
            extrinsic_source="T_HC_calibrated",
        )
    result = estimate_postgrasp_T_HP(views, state)
    assert result["T_HP_independent_view_count"] == 1
    assert result["T_HP_multiview"] is False
    assert len(result["T_HP_ignored_comoving_view_ids"]) == 1


def test_hp_only_c2_branches_are_matrix_right_pairs_nonzero_tilt():
    from scipy.spatial.transform import Rotation

    state = np.array([0.01, -0.02, 0.03, 0.3, -0.2, 0.5, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    result = estimate_grouped_views(
        postgrasp_inhand_views=views,
        final_preinsert_views=(),
        initial_state=state,
    )
    hypotheses = result["c2"]["hypotheses"]
    t0 = pose_matrix(np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]))
    t1 = pose_matrix(np.asarray(hypotheses[1]["T_hand_plug_xyz_rpy"]))
    expected_t1 = t0.copy()
    expected_t1[:3, :3] = t0[:3, :3] @ Rotation.from_euler(
        "z", np.pi
    ).as_matrix()
    assert np.max(np.abs(t1 - expected_t1)) < 1.0e-10
    plug_cad, receptacle_cad = proxy_cad_points()
    cost0 = float(
        np.sum(
            _ResidualProblem(
                views,
                plug_cad,
                receptacle_cad,
                np.concatenate(
                    (
                        np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]),
                        state[6:],
                    )
                ),
                include_prior=False,
                endpoints=("plug",),
            ).residual(
                np.concatenate(
                    (
                        np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]),
                        state[6:],
                    )
                )
            )
            ** 2
        )
    )
    cost1 = float(
        np.sum(
            _ResidualProblem(
                views,
                plug_cad,
                receptacle_cad,
                np.concatenate(
                    (
                        np.asarray(hypotheses[1]["T_hand_plug_xyz_rpy"]),
                        state[6:],
                    )
                ),
                include_prior=False,
                endpoints=("plug",),
            ).residual(
                np.concatenate(
                    (
                        np.asarray(hypotheses[1]["T_hand_plug_xyz_rpy"]),
                        state[6:],
                    )
                )
            )
            ** 2
        )
    )
    assert abs(cost0 - cost1) < 1.0e-9


def _occluded_view():
    camera = fixed_camera_model(
        eye=(0.1, 0.0, 1.0),
        target=(0.0, 0.0, 0.0),
        resolution=(64, 36),
    )
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    t_wc = np.eye(4)
    t_wc[:3, :3] = rows.T
    t_wc[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return FormalView(
        view_id="OCCLUDED",
        timestamp_utc="2026-08-15T00:00:00Z",
        rgb=np.zeros((36, 64, 3), dtype=np.uint8),
        depth=np.full((36, 64), 0.5, dtype=np.float32),
        camera=camera,
        T_WH=np.eye(4),
        T_WC=t_wc,
        T_HC=np.eye(4),
    )


def test_foreground_occluded_samples_are_ignored_and_support_gate_fails():
    plug_cad, receptacle_cad = proxy_cad_points()
    state = np.zeros(12)
    view = _occluded_view()
    baseline = _ResidualProblem(
        [view], plug_cad, receptacle_cad, state, include_prior=False,
        endpoints=("plug",), occlusion_policy="baseline"
    )
    ignored = _ResidualProblem(
        [view], plug_cad, receptacle_cad, state, include_prior=False,
        endpoints=("plug",), occlusion_policy="ignore_foreground_occluded"
    )
    baseline_residual = baseline.residual(state)
    ignored_residual = ignored.residual(state)
    assert float(np.sum(np.abs(baseline_residual))) > 1.0
    assert float(np.sum(np.abs(ignored_residual))) == 0.0
    support = ignored.last_plug_support[0]
    assert support["foreground_occluded_fraction"] > 0.9
    assert support["visible_depth_support_fraction"] == 0.0
    assert support["visible_depth_support_fraction"] < 0.05


def test_shell25j_profile_c2_and_local_rz_observability():
    from scipy.spatial import cKDTree
    from kcg_connector.d38999_cad_registration import (
        fixed_camera_model,
        render_points,
        shell25j_cad_profile_metadata,
        shell25j_plug_cad_profile,
        transform_points,
    )
    from kcg_connector.d38999_inhand_multiview import c2_action_pose6

    profile = shell25j_plug_cad_profile().plug_mating
    acted = transform_points(profile, c2_action_pose6(np.zeros(6)))
    tree = cKDTree(acted.xyz)
    distance, _ = tree.query(profile.xyz)
    assert float(np.max(distance)) < 1.0e-8
    camera = fixed_camera_model(
        eye=(0.05, 0.0, 0.08),
        target=(0.0, 0.0, 0.005),
        resolution=(320, 180),
    )
    import kcg_connector.d38999_cad_registration as cad_module

    old_plug = cad_module._legacy_axisymmetric_plug_profile()
    new_plug = profile

    def image_diff(cad):
        def render(rz_rad):
            return render_points(
                camera, (transform_points(cad, [0, 0, 0, 0, 0, rz_rad]),)
            )["rgb"].astype(int)
        return float(np.mean(np.abs(render(0.0) - render(np.radians(2.3)))))

    assert image_diff(old_plug) < 1.0e-9
    assert image_diff(new_plug) > 0.1
    metadata = shell25j_cad_profile_metadata()
    assert metadata["asset_sha256"] == (
        "6f716b6e40129f98e5914b5597005c575617d5f55cd0f2c0c8df067ee6788740"
    )
    assert metadata["socket_count"] == 61
    assert metadata["mating_shell_segment_count"] == 20


def test_physical_central_jacobian_has_separate_translation_rotation_scale():
    plug_cad, receptacle_cad = proxy_cad_points()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    problem = _ResidualProblem(
        views,
        plug_cad,
        receptacle_cad,
        state,
        frozen_mask=(False,) * 6 + (True,) * 6,
        endpoints=("plug",),
    )
    jac, normalized_steps = problem._physical_normalized_jacobian()
    evaluated = jac(np.zeros(12))
    assert evaluated.shape[1] == 12
    assert np.all(np.isfinite(evaluated))
    assert np.all(evaluated[:, :6].any(axis=0))
    assert np.allclose(evaluated[:, 6:], 0.0)
    assert normalized_steps[0] == pytest.approx(0.1)
    assert normalized_steps[3] == pytest.approx(0.002)


def test_deterministic_multistart_is_truth_free_and_bounded():
    from kcg_connector.postgrasp_shadow_estimator import _deterministic_hp_starts

    nominal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    starts = _deterministic_hp_starts(nominal, 9)
    assert len(starts) == 9
    assert np.array_equal(starts[0], nominal)
    assert all(
        abs(float(start[index])) <= (0.002 if index < 3 else 0.10471975511965977)
        for start in starts
        for index in range(6)
    )
    starts_again = _deterministic_hp_starts(nominal, 9)
    assert all(np.array_equal(a, b) for a, b in zip(starts, starts_again))


def test_depth_gated_edge_ignores_background_depth_edges_and_low_support_fails():
    plug_cad, receptacle_cad = proxy_cad_points()
    state = np.zeros(12)
    view = _occluded_view()
    global_edge = _ResidualProblem(
        [view], plug_cad, receptacle_cad, state, include_prior=False,
        endpoints=("plug",), occlusion_policy="baseline",
        edge_policy="global",
    )
    gated_edge = _ResidualProblem(
        [view], plug_cad, receptacle_cad, state, include_prior=False,
        endpoints=("plug",), occlusion_policy="baseline",
        edge_policy="depth_gated",
    )
    global_residual = global_edge.residual(state)
    gated_residual = gated_edge.residual(state)
    edge_length = global_edge.group_sizes[0]
    assert float(np.sum(np.abs(global_residual[:edge_length]))) > 0.0
    assert float(np.sum(np.abs(gated_residual[:edge_length]))) == 0.0
    support = gated_edge.last_plug_support[0]
    assert support["depth_gated_edge_support_fraction"] == 0.0
    assert support["depth_gated_edge_support_fraction"] < 0.02


def test_valid_5dof_c2_unresolved_and_grouped_api():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    views = _synthetic_views(state)
    result = estimate_joint_c2(views, state)
    assert result["success"] is True
    assert result["status"] == "VALID_5DOF_C2_UNRESOLVED"
    assert result["c2"]["resolution"] == "C2_UNRESOLVED"
    assert result["c2"]["observable_dofs"] == 5
    assert len(result["c2"]["hypotheses"]) == 2
    assert result["c2"]["averaged"] is False
    assert result["control_authorized"] is False
    hp_only = estimate_grouped_views(
        postgrasp_inhand_views=views,
        final_preinsert_views=(),
        initial_state=state,
    )
    assert hp_only["status"] == "REJECTED_T_HP_POSE_INVALID"
    assert hp_only["success"] is False
    assert hp_only["optimizer_converged"] is True
    assert hp_only["reject_reason"] is not None
    assert hp_only["T_receptacle_plug_status"] == "UNAVAILABLE_NO_RECEPTACLE_VIEWS"
    assert len(hp_only["c2"]["hypotheses"]) == 2
    assert hp_only["c2"]["averaged"] is False
    assert hp_only["c2"]["rz_status"] == "C2_UNRESOLVED"
    assert hp_only["optimizer_converged"] is True
    assert hp_only["pose_valid"] is False
    assert hp_only["real_keying_modeled"] is False
    assert hp_only["keying_model_id"] is None
    assert hp_only["c2"]["selected_for_shadow"] is None
    key_branch = hp_only["key_branch_selection"]
    assert key_branch["status"] == "KEYED_GEOMETRY_UNAVAILABLE"
    assert key_branch["current_model_id"] == "d38999_shell25j_proxy_v1"
    assert key_branch["selected_for_shadow"] is None
    assert key_branch["shadow_selected_hypothesis_id"] is None
    assert key_branch["control_authorized"] is False
    assert "selected_for_control" not in key_branch
    assert "T_hand_plug_xyz_rpy" not in hp_only
    # Unobservable directions must not leak Infinity/NaN into the evidence
    # JSON, because the runtime writes with allow_nan=False.
    json.dumps(hp_only, allow_nan=False, sort_keys=True)
    for hypothesis in hp_only["c2"]["hypotheses"]:
        assert hypothesis["probability"] is None
        assert hypothesis["probability_status"] == "UNVALIDATED"
        assert hypothesis["condition_number"] is None or np.isfinite(
            hypothesis["condition_number"]
        )
    rp_only = estimate_grouped_views(
        postgrasp_inhand_views=(),
        final_preinsert_views=views,
        initial_state=state,
        T_hand_plug_fixed=state[:6],
    )
    assert rp_only["status"] == "VALID_T_RP_ONLY"


def test_no_t_hp_view_still_reports_unkeyed_geometry_blocker():
    result = estimate_postgrasp_T_HP([], np.zeros(12, dtype=np.float64))

    assert result["status"] == "REJECTED_T_HP_POSE_INVALID"
    assert result["key_branch_selection"]["status"] == (
        "KEYED_GEOMETRY_UNAVAILABLE"
    )
    assert result["key_branch_selection"]["selected_for_shadow"] is None
    assert result["control_authorized"] is False


def test_candidate_rz_search_covers_observed_postgrasp_envelope():
    # The measured sequential-grasp design envelope reaches about 5.89 deg.
    # These remain SIM_TUNING_ONLY candidates, never authorization gates.
    required = np.deg2rad(6.0)
    assert ALL_THRESHOLDS_CANDIDATE["hp_rz_search_half_width_rad"] >= required
    assert ALL_THRESHOLDS_CANDIDATE["rp_rz_search_half_width_rad"] >= required
