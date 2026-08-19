#!/usr/bin/env python3
import sys
from pathlib import Path

path = Path("/home/noob/WorkPlace/kcgtest1/src/kcg_connector/isaac/d38999_postgrasp_fresh_replay_smoke.py")
text = path.read_text(encoding="utf-8")

anchor = 'capture_views.append(("WRIST_H0", "wrist", None))'
start_anchor = "            actual_arm_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).ravel()[:7]"
end_anchor = '            metrics["wrist_h0_capture"] = {"status":"GPU_PASS","view_id":"WRIST_H0","control_authorized":False,"formal_estimator_input":True}'

ai = text.find(anchor)
si = text.find(start_anchor, ai)
ei = text.find(end_anchor, ai)
if ai < 0 or si < 0 or ei < 0:
    print("anchors not found", ai, si, ei); sys.exit(1)
ei_end = ei + len(end_anchor)
print("replacing chars", si, "..", ei_end)

lines = [
"            actual_arm_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).ravel()[:7]",
"            tcp_fk = np.asarray(iiwa14_grasp_tcp_transform(tuple(actual_arm_q)))",
"            t_wh = tcp_fk @ tcp_from_handbase",
"            camera_path = \"/World/PostgraspShadowWristRgbdCamera\"",
"            camera_prim = stage.GetPrimAtPath(camera_path)",
"            if camera_prim is None or not camera_prim.IsValid():",
"                camera_prim = UsdGeom.Camera.Define(stage, camera_path)",
"            identity = Gf.Matrix4d(1.0); identity.SetTranslateOnly(Gf.Vec3d(0,0,0))",
"            UsdGeom.Xformable(camera_prim).ClearXformOpOrder(); UsdGeom.Xformable(camera_prim).AddTransformOp().Set(identity)",
"            camera_prim.CreateFocalLengthAttr(24.0); camera_prim.CreateHorizontalApertureAttr(20.955)",
"            camera_prim.CreateVerticalApertureAttr(20.955*720/1280); camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1,10.0))",
"            view_records = []",
"            for view_id, kind, palm_eye in capture_views:",
"                if kind == \"palm\":",
"                    palm_model = fixed_camera_model(",
"                        eye=palm_eye,",
"                        target=(0.001, 0.0, 0.0),",
"                        resolution=(1280, 720),",
"                    )",
"                    palm_rows = np.asarray(",
"                        palm_model.world_to_camera, dtype=np.float64",
"                    )",
"                    camera_in_plug = np.eye(4, dtype=np.float64)",
"                    camera_in_plug[:3, :3] = palm_rows.T",
"                    camera_in_plug[:3, 3] = np.asarray(",
"                        palm_model.position_world, dtype=np.float64",
"                    )",
"                    t_hc = nominal_hand_to_plug @ camera_in_plug",
"                else:",
"                    t_hc = _calibrated_hand_camera_from_nominal_plug(",
"                        nominal_hand_to_plug, (0.120, 0.0, 0.060), (0.0, 0.0, 0.006), (1280, 720)",
"                    )",
"                t_wc = t_wh @ t_hc",
"                eye_world = tuple(float(v) for v in t_wc[:3, 3])",
"                target_world = tuple(float(v) for v in _camera_target_from_t_wc(t_wc))",
"                direction = np.asarray(target_world) - np.asarray(eye_world)",
"                direction = direction / np.linalg.norm(direction)",
"                camera_rotation = Gf.Rotation(Gf.Vec3d(0,0,-1), Gf.Vec3d(*direction))",
"                camera_matrix = Gf.Matrix4d(1.0); camera_matrix.SetRotate(camera_rotation); camera_matrix.SetTranslateOnly(Gf.Vec3d(*eye_world))",
"                UsdGeom.Xformable(camera_prim).ClearXformOpOrder(); UsdGeom.Xformable(camera_prim).AddTransformOp().Set(camera_matrix)",
"                wrist_rgbd = replace(rgbd_base, camera=replace(rgbd_base.camera, prim_path=camera_path, frame_id=\"postgrasp_wrist_rgbd_camera_optical\", eye_m=eye_world, target_m=target_world, resolution=(1280,720)))",
"                view_dir = output_root / \"formal_views\" / view_id",
"                capture = capture_d38999_rgbd_raw_formal(",
"                    bindings={\"Camera\":Camera,\"Gf\":Gf,\"Image\":Image,\"Usd\":Usd,\"UsdGeom\":UsdGeom,\"UsdLux\":UsdLux,\"rep\":rep},",
"                    simulation_app=simulation_app, world=world, stage=stage, tabletop=tabletop,",
"                    rgbd=wrist_rgbd, output_dir=view_dir, camera_clipping_range_m=(0.1,10.0),",
"                )",
"                if capture.passed is not True:",
"                    raise RuntimeError(f\"raw capture failed for view {view_id}\")",
"                (view_dir/\"fk.json\").write_text(json.dumps({\"arm_q_actual_rad\":actual_arm_q.tolist(),\"tcp_pose_4x4\":tcp_fk.tolist(),\"T_WH_4x4\":t_wh.tolist(),\"T_WC_4x4\":t_wc.tolist()},indent=2)+\"\\n\")",
"                camera_record = {\"prim_path\":camera_path,\"frame_id\":\"postgrasp_wrist_rgbd_camera_optical\",\"eye_m\":list(eye_world),\"target_m\":list(target_world),\"intrinsics\":capture.metrics[\"camera\"][\"intrinsics\"]}",
"                if kind == \"palm\":",
"                    camera_record[\"palm_eye_plug_m\"] = list(palm_eye)",
"                    camera_record[\"palm_target_plug_m\"] = [0.001, 0.0, 0.0]",
"                    camera_record[\"T_HC_4x4\"] = t_hc.tolist()",
"                (view_dir/\"camera.json\").write_text(json.dumps(camera_record,indent=2)+\"\\n\")",
"                view_records.append({\"view_id\":view_id,\"output_directory\":str(view_dir)})",
"                metrics[f\"capture_{view_id}\"] = {\"status\":\"GPU_PASS\",\"view_id\":view_id,\"control_authorized\":False,\"formal_estimator_input\":True}",
"            (output_root/\"formal_manifest.json\").write_text(json.dumps({\"schema_version\":\"kcg_d38999_wrist_h0_v1\",\"role\":\"formal_raw_observation\",\"formal_estimator_input\":True,\"estimator_run\":False,\"control_authorized\":False,\"object_truth_present\":False,\"contact_report_present\":False,\"views\":view_records},indent=2)+\"\\n\")",
"            if arguments.wrist_h0_capture and not arguments.palm_h0_capture:",
"                metrics[\"wrist_h0_capture\"] = {\"status\":\"GPU_PASS\",\"view_id\":\"WRIST_H0\",\"control_authorized\":False,\"formal_estimator_input\":True}",
]
new_block = "\n".join(lines) + "\n"
text = text[:si] + new_block + text[ei_end:]
path.write_text(text, encoding="utf-8")
print("done")
