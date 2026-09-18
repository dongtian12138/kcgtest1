"""Bounded reuse of deterministic FK within one immutable runtime model.

Every caller still receives independent writable matrices. Only finite tuples
of independent coordinates with the default base are cached; all other calls
retain the original validation path. Keys preserve signed zero and the limit
checking mode. A change of any model contract object invalidates the cache.
"""
from collections import OrderedDict
from collections.abc import Mapping
import math
import struct
from types import MappingProxyType, MethodType


def install_fk_cache(model, capacity=16):
    if type(capacity) is not int or not 1<=capacity<=32:
        raise ValueError('A small bounded FK cache is required')
    if hasattr(model,'_performance_fk_cache_report'):
        return model._performance_fk_cache_report
    original=model.forward_kinematics
    cache=OrderedDict()
    report={'capacity':capacity,'hits':0,'misses':0,'bypassed':0,'invalidations':0,
            'independent_result_arrays':True,'physics_or_controller_parameters_changed':False}
    previous=[None]
    def signature(self):
        return tuple(id(getattr(self,name)) for name in
                     ('joints','joint_order','base_link','fourbar_couplings','independent_joint_names'))
    def cached(self,positions,*,base_transform=None,enforce_limits=True):
        if (not isinstance(positions,tuple) or base_transform is not None
                or len(positions)!=len(self.independent_joint_names)
                or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in positions)
                or type(enforce_limits) is not bool):
            report['bypassed']+=1
            return original(positions,base_transform=base_transform,enforce_limits=enforce_limits)
        current=signature(self)
        if current!=previous[0]:
            if previous[0] is not None:report['invalidations']+=1
            cache.clear();previous[0]=current
        key=(struct.pack('<'+'d'*len(positions),*positions),enforce_limits)
        value=cache.get(key)
        if value is None:
            report['misses']+=1
            value=original(positions,enforce_limits=enforce_limits)
            if not isinstance(value,Mapping):raise TypeError('FK must return named transforms')
            saved={name:matrix.copy() for name,matrix in value.items()}
            for matrix in saved.values():matrix.flags.writeable=False
            cache[key]=saved
            if len(cache)>capacity:cache.popitem(last=False)
            return value
        report['hits']+=1;cache.move_to_end(key)
        return MappingProxyType({name:matrix.copy() for name,matrix in value.items()})
    model.forward_kinematics=MethodType(cached,model)
    model._performance_fk_cache_report=report
    return report
