import numpy as np
import pytest
from global_socket_coarse_geometry import depth_component_from_image_seed


def data():
    depth=np.full((40,60),2.)
    depth[5:25,5:25]=1.
    depth[5:25,35:55]=1.
    background=np.full(depth.shape,2.)
    k=np.array([[100.,0.,30.],[0.,100.,20.],[0.,0.,1.]])
    workspace={'minimum':[-1.,-1.,.8],'maximum':[1.,1.,1.2]}
    return depth,background,k,workspace


def test_inner_seed_recovers_only_its_connected_measured_object():
    depth,background,k,workspace=data();seed=np.zeros(depth.shape,bool);seed[7:23,7:23]=True
    original=seed.copy()
    mask,audit=depth_component_from_image_seed(depth,background,seed,k,np.eye(4),workspace)
    assert mask.sum()==400 and mask[5:25,5:25].all() and not mask[:,35:].any()
    np.testing.assert_array_equal(seed,original)
    assert audit['overlap_pixels']==256 and not audit['object_or_contact_truth_used']


def test_ambiguous_two_object_seed_is_rejected():
    depth,background,k,workspace=data();seed=depth<2.
    with pytest.raises(ValueError):depth_component_from_image_seed(depth,background,seed,k,np.eye(4),workspace)


def test_background_or_outside_workspace_cannot_supply_missing_rim():
    depth,background,k,workspace=data();seed=np.ones(depth.shape,bool)
    with pytest.raises(ValueError):depth_component_from_image_seed(depth,background,seed,k,np.eye(4),workspace)
    seed=np.zeros(depth.shape,bool);seed[5:25,5:25]=True
    with pytest.raises(ValueError):depth_component_from_image_seed(depth,depth,seed,k,np.eye(4),workspace)
