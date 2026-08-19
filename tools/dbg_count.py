import numpy as np
from kcg_connector.postgrasp_shadow_estimator import FormalView, estimate_postgrasp_T_HP
from kcg_connector.d38999_cad_registration import (
    fixed_camera_model, proxy_cad_points, render_points, transform_points,
)
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix


def _camera_pose(camera):
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    m = np.eye(4); m[:3,:3] = rows.T; m[:3,3] = np.asarray(camera.position_world)
    return m


def _synthetic_views(state, count=2):
    plug_cad, receptacle_cad = proxy_cad_points()
    hp, rp = state[:6], state[6:]
    views = []
    eyes = ((0.55, -0.85, 0.72), (0.30, -0.85, 0.72))
    for index, eye in enumerate(eyes[:count]):
        camera = fixed_camera_model(eye=eye, target=(0.535, -0.0125, 0.231), resolution=(320, 180))
        T_WC = _camera_pose(camera)
        T_WH = np.eye(4)
        T_WP = T_WH @ pose_matrix(hp)
        T_WR = T_WP @ np.linalg.inv(pose_matrix(rp))
        obs = render_points(camera, (transform_points(plug_cad, matrix_pose(T_WP)),
                                     transform_points(receptacle_cad, matrix_pose(T_WR))))
        views.append(FormalView(view_id=f"V{index}", timestamp_utc="t", rgb=obs["rgb"],
                                depth=obs["depth"], camera=camera, T_WH=T_WH,
                                T_HC=np.linalg.inv(T_WH) @ T_WC, T_WC=T_WC))
    return views


state = np.array([0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,-0.02,0.0,0.0,0.0])
views = _synthetic_views(state)
print("n views:", len(views), "groups:", [v.group for v in views], "ids:", [v.view_id for v in views])
views[1] = FormalView(view_id="PALM_SECONDARY_V0", timestamp_utc=views[1].timestamp_utc,
                      rgb=views[1].rgb, depth=views[1].depth, camera=views[1].camera,
                      T_WH=views[1].T_WH, T_WC=views[1].T_WC, T_HC=views[1].T_HC,
                      group="postgrasp_second_inhand_camera_views",
                      extrinsic_source="T_HC_calibrated")
result = estimate_postgrasp_T_HP(views, state)
print("count:", result["T_HP_independent_view_count"], "ignored:", result["T_HP_ignored_comoving_view_ids"])
