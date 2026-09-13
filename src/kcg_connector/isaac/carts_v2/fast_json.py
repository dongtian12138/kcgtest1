"""C JSON codec for numeric evidence; retain NaN/Inf failure values as JSON did."""
import ujson
from pathlib import Path


def _default(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return ujson.dumps(value,ensure_ascii=False,escape_forward_slashes=False,default=_default)


def loads(value):
    return ujson.loads(value)


def dump_array(stream,rows):
    """Stream the existing array format without copying an entire long trace."""
    stream.write('[')
    for i,row in enumerate(rows):
        if i:stream.write(',')
        stream.write(dumps(row))
    stream.write(']')
