"""Lossless sensor history with a bounded cache of decoded Python objects.

Full history remains available for tare, regrasp initialization and recovery.
The persistent representation consists of bytes, which cyclic GC does not scan.
Records are snapshots; consumers must not mutate returned historical samples.
"""
from collections import OrderedDict
from collections.abc import Sequence
import msgpack
from pathlib import Path


def _normalize(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value,Path):
        return str(value)
    raise TypeError(type(value).__name__)


class EncodedSensorHistory(Sequence):
    def __init__(self, cache_rows=512, *, compression_backend='gzip'):
        if cache_rows < 1:
            raise ValueError('positive sensor history cache required')
        self._encoded = []
        self._cache = OrderedDict()
        self.cache_rows = cache_rows
        self.compression_backend = compression_backend
        self.normalized_numeric_snapshots = True

    def __len__(self):
        return len(self._encoded)

    def _remember(self, index, row):
        self._cache[index] = row
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_rows:
            self._cache.popitem(last=False)
        return row

    def append(self, row):
        payload = msgpack.packb(row, use_bin_type=True, default=_normalize)
        index = len(self._encoded)
        self._encoded.append(payload)
        self._remember(index, msgpack.unpackb(payload, raw=False))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        return self._remember(index, msgpack.unpackb(self._encoded[index], raw=False))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
