"""Compatibility required by compact contact and sensor-history recording."""
import json
import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import msgpack

from carts_v2.contact_codec import PackedContactPoints, decode_extension, encode_extension
from carts_v2.contact_codec import PackedNativeFloat32ContactPoints
from carts_v2.sample_store import GzipSampleStore
from carts_v2.sensor_history import EncodedSensorHistory
from trace_metadata import iter_truth_fields, read_truth_sample, write_gzip_array


class PackedRuntimeRecordsTests(unittest.TestCase):
    def test_native_float32_roundtrip_preserves_promoted_values_and_float64_review_math(self):
        import numpy as np
        bits = [0,0x80000000,1,0x807fffff,0x3f800000,0xbf800000,
                0x7f7fffff,0x00800000,0x3eaaaaab,0x12345678]
        raw = struct.pack('<10I',*bits)
        native = PackedNativeFloat32ContactPoints(raw)
        legacy = PackedContactPoints(struct.pack('<10d',*struct.unpack('<10f',raw)))
        restored = msgpack.unpackb(msgpack.packb(native,default=encode_extension),
                                   raw=False,ext_hook=decode_extension)
        self.assertEqual(restored.payload,raw)
        self.assertEqual(msgpack.packb(restored.tolist()),msgpack.packb(legacy.tolist()))
        self.assertEqual(restored.as_array().dtype,np.dtype('float64'))
        self.assertEqual(restored.as_array().tobytes(),legacy.as_array().tobytes())
        self.assertEqual(len(raw)*2,len(legacy.payload))

    def test_json_evaluation_witness_accepts_both_packed_contact_formats(self):
        from carts_v2.fast_json import _default
        for cls,fmt in ((PackedContactPoints,'<10d'),(PackedNativeFloat32ContactPoints,'<10f')):
            points=cls(struct.pack(fmt,*([.25,-0.]+[0.]*8)))
            witness={'worst_frame':{'contacts':{'poll_headers':[{'contacts':points}]}}}
            restored=json.loads(json.dumps(witness,default=_default))
            self.assertEqual(restored['worst_frame']['contacts']['poll_headers'][0]['contacts'],points.tolist())

    def test_indexed_and_projected_readers_preserve_every_contact_field(self):
        values = [1.25, -0.0, 2., 0., 0., 1., 1e-7, -2e-7, 0., -1e-6]
        points = PackedContactPoints(struct.pack('<10d', *values))
        expected = msgpack.packb(points.tolist(), use_bin_type=True)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            store = GzipSampleStore(directory/'truth_samples.msgpack.gz',
                                    block_size=2, cache_blocks=1, codec='msgpack')
            for i in range(7):
                store.append({'step':i,'contacts':{'poll_headers':[{'contacts':points}]},'positions':[i,0.,0.]})
            store.close()
            (directory/'motion_timing.json').write_text('{}')
            for i in (6,0,3,2):
                row = read_truth_sample(directory,i)
                actual = row['contacts']['poll_headers'][0]['contacts']
                self.assertEqual(msgpack.packb(actual.tolist(),use_bin_type=True),expected)
            projected = list(iter_truth_fields(directory,('positions',),2,5))
            self.assertEqual(projected,[{'step':i,'positions':[i,0.,0.]} for i in range(2,6)])

    def test_malformed_extension_fails_and_nonfinite_failure_values_survive(self):
        with self.assertRaises(ValueError):
            decode_extension(42,b'x')
        payload = struct.pack('<10d', *([float('inf'),float('-inf'),float('nan')]+[0.]*7))
        packed = PackedContactPoints(payload)
        restored = msgpack.unpackb(msgpack.packb(packed,default=encode_extension),
                                   ext_hook=decode_extension,raw=False)
        self.assertEqual(restored.payload,payload)
        self.assertFalse(restored.native_finite_verified)

    def test_sensor_history_keeps_tare_tail_recovery_and_snapshot_semantics(self):
        rows = [{'step':i,'phase':'tare' if i<2 else 'turn',
                 'wrench':[i,-0.0,None],'arm_control':{'target':[i+.125]}} for i in range(9)]
        history = EncodedSensorHistory(cache_rows=2)
        for row in rows:
            history.append(row)
        self.assertEqual(list(history),rows)
        for sl in (slice(None,None,-1),slice(0,2),slice(-3,None),slice(7,1,-2),slice(3,3)):
            self.assertEqual(history[sl],rows[sl])
        self.assertEqual(list(reversed(history)),list(reversed(rows)))
        self.assertLessEqual(len(history._cache),2)
        rows[0]['wrench'][0] = 999
        self.assertEqual(history[0]['wrench'][0],0)

    def test_fast_compression_remains_standard_gzip_and_keeps_custom_prepare(self):
        rows=[{'step':i,'value':[-0.0,i]} for i in range(7)]
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            history=EncodedSensorHistory(cache_rows=2,compression_backend='isal')
            store=GzipSampleStore(directory/'truth_samples.msgpack.gz',block_size=2,
                                  codec='msgpack',compression_backend='isal')
            for row in rows:
                history.append(row)
                store.append(row)
            store.close()
            self.assertEqual(read_truth_sample(directory,3),rows[3])
            write_gzip_array(directory/'sensors.json.gz',history,prepare=lambda row:{'index':row['step']})
            with gzip.open(directory/'sensors.json.gz','rt') as stream:
                self.assertEqual(json.load(stream),[{'index':i} for i in range(7)])


if __name__ == '__main__':
    unittest.main()
