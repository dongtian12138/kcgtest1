"""Lossless contact-point blocks with the existing per-point read interface.

MessagePack extension 42 stores ten little-endian float64 values per point.
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

    def __init__(self, payload, *, native_finite_verified=False):
        if sys.byteorder != 'little' or not isinstance(payload, bytes) or len(payload) % POINT.size:
            raise ValueError('invalid little-endian float64 contact block')
        self.payload = payload
        self.native_finite_verified = bool(native_finite_verified)

    def __len__(self):
        return len(self.payload)//POINT.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        p = POINT.unpack_from(self.payload, index*POINT.size)
        return {'position_m':list(p[:3]),'normal':list(p[3:6]),
                'impulse_n_s':list(p[6:9]),'separation_m':p[9]}

    def tolist(self):
        return list(self)

    def as_array(self):
        return np.frombuffer(self.payload,dtype='<f8').reshape(-1,10)


def encode_extension(value):
    if isinstance(value, PackedContactPoints):
        return msgpack.ExtType(CONTACT_EXT_CODE,value.payload)
    if hasattr(value,'tolist'):
        return value.tolist()
    from pathlib import Path
    if isinstance(value,Path):
        return str(value)
    raise TypeError(type(value).__name__)


def decode_extension(code, payload):
    if code == CONTACT_EXT_CODE:
        return PackedContactPoints(payload)
    return msgpack.ExtType(code,payload)
