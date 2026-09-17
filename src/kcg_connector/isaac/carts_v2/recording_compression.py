"""Explicit gzip-compatible recording backends; no changes to observations."""
from functools import lru_cache
from pathlib import Path
import gzip
import sys


@lru_cache(maxsize=2)
def gzip_backend(name='gzip'):
    if name == 'gzip':
        return gzip
    if name != 'isal':
        raise ValueError('unsupported recording compression backend')
    try:
        from isal import igzip
    except ModuleNotFoundError:
        local = Path(__file__).resolve().parents[4]/'.deps/recording'
        if not (local/'isal').is_dir():
            raise RuntimeError('Install isal==1.8.0 in the declared recording environment')
        sys.path.insert(0,str(local))
        from isal import igzip
    return igzip
