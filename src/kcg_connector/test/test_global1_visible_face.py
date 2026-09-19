from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import global1_visible_face as module
from te_visual_body_start import body_grasp_frame


def setup(monkeypatch, *, yaw=0., tilt=0., height=.2, sign_unique=True):
    measured=np.eye(4)
    measured[:3,:3]=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])@Rotation.from_euler('z',yaw).as_matrix()
    measured[:3,3]=[.52,-.21,height]
    inputs=SimpleNamespace(depth_m=np.ones((2,2)),intrinsics=np.eye(3),world_from_camera=np.eye(4),
        manifest={'known_static_scene_geometry':{'table':{'center_world_m':[0.,0.,.15],'size_m':[1.,1.,.1]}},
            'frozen_endpoint_workspaces_world_aabb_m':{'plug':{'minimum':[.47,-.26,.19],'maximum':[.57,-.16,.245]}}})
    monkeypatch.setattr(module,'_foreground_points',lambda *_:{'plug':{'pixel_v':[0,1],'pixel_u':[0,1]}})
    monkeypatch.setattr(module,'estimate_plug_rear_circle_from_float_depth',lambda **_: {'camera_from_object':measured,'metrics':{}})
    legacy={'transport_grasp_pose':{'quaternion_xyzw':None,'derivation':{
        'axis_sign':{'sign_unique':sign_unique,'supplier_plus_z_world_sign':-1}}}}
    return inputs,legacy


def test_arbitrary_face_transverse_basis_does_not_rotate_initial_grasp(monkeypatch):
    first=module.estimate(*setup(monkeypatch,yaw=0.),'CAD')
    second=module.estimate(*setup(monkeypatch,yaw=1.3),'CAD')
    np.testing.assert_allclose(body_grasp_frame(first['transport_grasp_pose']),body_grasp_frame(second['transport_grasp_pose']))
    assert first['key_yaw_measured'] is False
    assert second['transport_grasp_pose']['quaternion_xyzw'] is None


@pytest.mark.parametrize('changes',[{'tilt':6.},{'height':.204},{'sign_unique':False}])
def test_original_support_and_axis_guards_remain_fail_closed(monkeypatch,changes):
    with pytest.raises(ValueError):module.estimate(*setup(monkeypatch,**changes),'CAD')
