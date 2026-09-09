"""Read the native SDF at saved source-geometry points, never for control."""
import json,sys
from pathlib import Path
import numpy as np


def query_pin_contact_sdf(stage,physics_view,query_file,*,body_path,local_pin_paths):
    from pxr import UsdGeom
    repository=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(repository/'artifacts/kcg_connector/model_delivery_20260908/src'))
    from contact_sdf_probe import _frontend_array,_host
    query=json.load(Path(query_file).open());groups={}
    for record in query['records']:
        path=body_path+'/SocketRigidCoreCollision'
        for candidate in local_pin_paths:
            prim=stage.GetPrimAtPath(candidate);a=prim.GetAttribute('kcg:sourcePinIndices')
            indices=list(a.Get()) if a and a.HasAuthoredValueOpinion() else [int(candidate.rsplit('_',1)[-1])]
            if record['pin_index'] in indices:path=candidate;break
        if local_pin_paths and path.endswith('SocketRigidCoreCollision'):raise ValueError('source pin not found in SDF groups')
        groups.setdefault(path,[]).append(record)
    result={'scope':'READ_ONLY_NATIVE_SDF_AT_SAVED_CONTACT_POINTS; SOURCE TRUTH IS NOT USED BY CONTROL','status':'AVAILABLE','records':[]}
    for path,records in groups.items():
        local=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()).T
        if not np.allclose(local[:3,:3],np.eye(3),rtol=0,atol=1e-9):raise ValueError('expected source-metre SDF coordinates')
        points=np.array([r['point_body_m'] for r in records])-local[:3,3]
        view=physics_view.create_sdf_shape_view(path,len(points))
        if view.count!=1:raise RuntimeError('SDF view did not bind one shape')
        # This tensor layout was independently checked in the earlier source
        # SDF diagnostics: gradient XYZ followed by signed distance.
        native=_host(view.get_sdf_and_gradients(_frontend_array(physics_view,points[None],'cpu')))[0]
        if native.shape!=(len(points),4) or not np.isfinite(native).all():raise RuntimeError('invalid SDF readback')
        for record,value in zip(records,native):
            gradient=value[:3].astype(float);norm=np.linalg.norm(gradient);reference=np.array(record['exact_nearest_outward_gradient_body'])
            result['records'].append({**record,'sdf_path':path,'native_gradient_mesh':gradient.tolist(),
                'native_signed_distance_m':float(value[3]),'signed_distance_error_m':float(value[3]-record['exact_signed_distance_m']),
                'gradient_dot_exact':float(gradient@reference/max(norm,1e-30)),
                'minus_gradient_dot_recorded_contact_normal':float(-gradient@record['native_normal_body']/max(norm,1e-30)),
                'gradient_caveat':'Nearest-surface gradients can be nonunique at edges; both adjacent recorded poses and raw values are retained.'})
    return result
