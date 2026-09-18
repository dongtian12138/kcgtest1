"""Localize the socket from the same fixed Global1 frame used for pickup.

The image-only worker can run alongside pickup. Its result has a simulation
availability time; a fast wall-clock return never makes the observation early.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import json
import numpy as np


def _estimate(repository, image_directory, output, sample_time, camera, intrinsic, workspace):
    from te_body_socket_observation import SOCKET_CAD_MM, SOCKET_TEMPLATES
    from te_foundationpose_handoff_runtime import _run_sam6d_frame, _json_ready
    from te_runtime_paths import sam6d_runtime
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    calibration = output / 'camera.json'
    calibration.write_text(json.dumps({'cam_K': intrinsic.ravel().tolist(), 'depth_scale': 1}))
    sam_root, sam_python = sam6d_runtime(repository)
    seed = _run_sam6d_frame(repository=repository, sam6d_root=sam_root, sam6d_python=sam_python,
        templates=repository / SOCKET_TEMPLATES, cad_mm=repository / SOCKET_CAD_MM,
        rgb=image_directory / 'rgb.png', depth_m=image_directory / 'depth_m.npy',
        depth_mm=image_directory / 'depth_mm.png', camera_json=calibration,
        output_dir=output / 'perception', workspace_world_aabb_m=workspace,
        workspace_world_from_camera=camera)
    elapsed = perf_counter() - started
    record = _json_ready({'scope': 'FIXED_GLOBAL1_COARSE_SOCKET_ONLY',
        'same_frame_as_initial_plug_localization': True, 'rgbd_directory': str(image_directory),
        'sample_time_s': sample_time, 'estimation_wall_s': elapsed,
        'nominal_sensor_delay_s': .05, 'available_time_s': sample_time + .05 + elapsed,
        'world_from_camera_cv': camera, 'intrinsics_3x3': intrinsic, 'coarse_estimate': seed,
        'world_from_socket_coarse': camera @ np.asarray(seed['camera_from_object']),
        'socket_key_yaw_measured': False, 'online_object_or_contact_truth_used': False,
        'receptacle_cad_mm': str(repository / SOCKET_CAD_MM),
        'hardware_latency_calibrated': False})
    (output / 'camera_and_estimate.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def start(repository, runtime, image_directory, sample_time):
    from four_camera_rig import configuration, camera_spec, intrinsics
    root = Path(repository)
    rig = configuration(root, runtime)
    if rig is None or runtime.get('global1_socket_job') is not None:
        return
    sources = json.loads((root / 'src/kcg_connector/config/te_same_reset_rgbd_observe_v1.json').read_text())['frozen_sources']
    template = json.loads((root / sources['provider_input_template']).read_text())
    workspace = template['frozen_endpoint_workspaces_world_aabb_m']['receptacle']
    spec, camera = camera_spec(root, rig, 'global_1')
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='global1_image_only')
    runtime['global1_socket_executor'] = executor
    runtime['global1_socket_job'] = executor.submit(_estimate, root, Path(image_directory),
        Path(runtime['output_directory']) / 'global1_socket', sample_time, camera, intrinsics(spec), workspace)
