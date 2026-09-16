"""Free-space calibration must remain zero when only hand orientation changes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_three_finger_wrench_observer import FingerRootMomentObserver


def test_pose_gravity_is_removed_without_erasing_applied_finger_load(tmp_path):
    links=set(sum((list(v) for v in FingerRootMomentObserver.subtrees),[]))
    p=tmp_path/'src/iiwa_description/urdf/hand.xacro';p.parent.mkdir(parents=True)
    p.write_text('<robot>'+''.join(f'<link name="{n}"><inertial><mass value="0.2"/><origin xyz="0.1 0 0"/></inertial></link>' for n in links)+'</robot>')
    def fk(q,**kwargs):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',q[0]).as_matrix()
        return {n:T for n in links|{'base'}}
    joint=lambda:SimpleNamespace(parent_link='base',axis=np.array([0.,1.,0.]),origin_transform=lambda:np.eye(4))
    model=SimpleNamespace(joints={n:joint() for n in FingerRootMomentObserver.joints},forward_kinematics=fk)
    observer=FingerRootMomentObserver(tmp_path,model);q0=np.zeros(11);q1=q0.copy();q1[0]=np.pi/2
    bias=np.array([.01,-.02,.03]);initial=observer.gravity_moments(q0)+bias
    observer.calibrate_free_space([q0,q0],[initial,initial])
    rotated=observer.gravity_moments(q1)+bias
    assert np.max(np.abs(rotated-initial))>.3
    assert np.allclose(observer.observe(q1,rotated),0.,atol=1e-12)
    load=np.array([.1,.2,.3])
    assert np.allclose(observer.observe(q1,rotated+load),load,atol=1e-12)
    assert np.allclose(observer.control_moments(q1,rotated,10,.01),0.,atol=1e-12)
    filtered=observer.control_moments(q1,rotated+load,11,.01)
    assert np.allclose(filtered,load/6,atol=1e-12)
    assert np.array_equal(observer.control_moments(q1,rotated+load,11,.01),filtered)
