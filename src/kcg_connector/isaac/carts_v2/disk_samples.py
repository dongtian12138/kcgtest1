"""Append-only truth log with full indexed history and one cached payload.

This storage is for post-run evidence, not sensor feedback. Iteration and slices
read old rows from disk without retaining their large contact arrays in RAM.
"""

from array import array
from collections.abc import Sequence
import json
import operator
from pathlib import Path


class _SampleSlice(Sequence):
    def __init__(self, source, indices):
        self.source, self.indices = source, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _SampleSlice(self.source, self.indices[index])
        return self.source[self.indices[index]]


class DiskSamples(Sequence):
    def __init__(self, path):
        self.path = Path(path)
        self._writer = self.path.open('xb')
        self._offsets = array('Q')
        self._latest = None

    def append(self, sample):
        payload = (json.dumps(sample, separators=(',', ':')) + '\n').encode('utf-8')
        offset = self._writer.tell()
        self._writer.write(payload)
        self._writer.flush()
        self._offsets.append(offset)
        self._latest = sample

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _SampleSlice(self, range(len(self))[index])
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index == len(self) - 1:
            return self._latest
        with self.path.open('rb') as reader:
            reader.seek(self._offsets[index])
            return json.loads(reader.readline())

    def __iter__(self):
        count = len(self)
        with self.path.open('rb') as reader:
            for _ in range(count):
                yield json.loads(reader.readline())

    def close(self):
        self._writer.close()
