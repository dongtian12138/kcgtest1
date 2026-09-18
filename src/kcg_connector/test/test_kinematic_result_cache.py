import unittest
import numpy as np
from types import MappingProxyType
from kinematic_result_cache import install_fk_cache


class FakeModel:
    def __init__(self):
        self.joints={};self.joint_order=('joint',);self.base_link='base'
        self.fourbar_couplings={};self.independent_joint_names=('joint',)
        self.calls=0
    def forward_kinematics(self,positions,*,base_transform=None,enforce_limits=True):
        self.calls+=1
        if enforce_limits and positions[0]>1:raise ValueError('limit')
        a=np.eye(4);a[0,3]=positions[0]
        return MappingProxyType({'base':a})


class CacheTests(unittest.TestCase):
    def test_values_mutation_isolation_limit_mode_and_invalidation(self):
        model=FakeModel();r=install_fk_cache(model,capacity=2)
        first=model.forward_kinematics((.25,));expected=first['base'].tobytes()
        first['base'][0,3]=99
        second=model.forward_kinematics((.25,));self.assertEqual(second['base'].tobytes(),expected)
        second['base'][0,3]=88
        self.assertEqual(model.forward_kinematics((.25,))['base'].tobytes(),expected)
        self.assertEqual(model.calls,1)
        model.forward_kinematics((2.,),enforce_limits=False)
        with self.assertRaises(ValueError):model.forward_kinematics((2.,))
        model.joints={};model.forward_kinematics((.25,))
        self.assertEqual(r['invalidations'],1)
    def test_signed_zero_bypass_and_bounded_eviction(self):
        model=FakeModel();r=install_fk_cache(model,capacity=1)
        a=model.forward_kinematics((-0.,));b=model.forward_kinematics((0.,))
        self.assertNotEqual(a['base'].tobytes(),b['base'].tobytes())
        model.forward_kinematics((-0.,));self.assertEqual(model.calls,3)
        model.forward_kinematics([0.]);self.assertEqual(r['bypassed'],1)
