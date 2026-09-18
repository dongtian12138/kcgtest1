"""Lossless contact-point blocks with the existing per-point read interface.

Extension42 stores float64; extension43 preserves the SDK's native float32.
Both reconstruct the same Python float values and float64 review arrays.
All original fields and ordering can be reconstructed; no points are dropped.
"""
from collections.abc import Sequence
import struct
import sys
import msgpack
import numpy as np

CONTACT_EXT_CODE = 42
POINT = struct.Struct('<10d')


class PackedContactPoints(Sequence):
    __slots__ = ('payload', 'native_finite_verified')
    point_struct = POINT
    numpy_dtype = '<f8'
    extension_code = CONTACT_EXT_CODE

    def __init__(self, payload, *, native_finite_verified=False):
        if sys.byteorder != 'little' or not isinstance(payload, bytes) or len(payload) % self.point_struct.size:
            raise ValueError('invalid little-endian contact block')
        self.payload = payload
        self.native_finite_verified = bool(native_finite_verified)

    def __len__(self):
        return len(self.payload)//self.point_struct.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        p = self.point_struct.unpack_from(self.payload, index*self.point_struct.size)
        return {'position_m':list(p[:3]),'normal':list(p[3:6]),
                'impulse_n_s':list(p[6:9]),'separation_m':p[9]}

    def tolist(self):
        return list(self)

    def as_array(self):
        return np.frombuffer(self.payload,dtype=self.numpy_dtype).reshape(-1,10).astype(np.float64,copy=False)


class PackedNativeFloat32ContactPoints(PackedContactPoints):
    """Lossless only for values copied directly from the checked float32 SDK ABI."""
    __slots__ = ()
    point_struct = struct.Struct('<10f')
    numpy_dtype = '<f4'
    extension_code = 43


def encode_extension(value):
    if isinstance(value, PackedContactPoints):
        return msgpack.ExtType(value.extension_code,value.payload)
    if hasattr(value,'tolist'):
        return value.tolist()
    from pathlib import Path
    if isinstance(value,Path):
        return str(value)
    raise TypeError(type(value).__name__)


def decode_extension(code, payload):
    if code == CONTACT_EXT_CODE:
        return PackedContactPoints(payload)
    if code == PackedNativeFloat32ContactPoints.extension_code:
        return PackedNativeFloat32ContactPoints(payload)
    return msgpack.ExtType(code,payload)
