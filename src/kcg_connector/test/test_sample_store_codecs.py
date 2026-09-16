"""Raw observations must survive both codecs, cache eviction and indexed reads."""
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import msgpack
import numpy as np

from carts_v2.sample_store import GzipSampleStore
from trace_metadata import iter_truth_samples, read_truth_sample, truth_archive_path


class SampleStoreCodecTests(unittest.TestCase):
    def test_every_row_survives_boundaries_and_cache_eviction(self):
        expected=[{'step':i,'phase':'保持','values':[i*.001,-0.0],
                   'contacts':{'actor_id':2**64-1,'points':[[.1,.2,float(i)]]}}
                  for i in range(197)]
        for codec in ('jsonl','msgpack'):
            with self.subTest(codec=codec),tempfile.TemporaryDirectory() as name:
                directory=Path(name);path=directory/f'truth_samples.{codec}.gz'
                store=GzipSampleStore(path,block_size=64,cache_blocks=1,codec=codec)
                for row in expected:store.append(row)
                self.assertEqual(list(store[60:132]),expected[60:132])
                store.close()
                self.assertEqual(list(iter_truth_samples(directory)),expected)
                for i in (0,63,64,127,128,196,2):
                    self.assertEqual(store[i],expected[i])
                    self.assertEqual(read_truth_sample(directory,i),expected[i])
                self.assertEqual(math.copysign(1.,read_truth_sample(directory,0)['values'][1]),-1.)

    def test_numpy_and_nonfinite_failure_evidence_are_retained(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name);path=directory/'truth_samples.msgpack.gz'
            store=GzipSampleStore(path,block_size=1,cache_blocks=1,codec='msgpack')
            store.append({'step':0,'array':np.array([1.,2.]),'source':Path('/observation'),
                          'failure':[float('nan'),float('inf'),-float('inf')]})
            store.close();row=read_truth_sample(directory,0)
            self.assertEqual(row['array'],[1.,2.]);self.assertEqual(row['source'],'/observation')
            self.assertTrue(math.isnan(row['failure'][0]))
            self.assertEqual(row['failure'][1:],[float('inf'),-float('inf')])

    def test_incomplete_binary_message_cannot_silently_drop_a_step(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name);path=directory/'truth_samples.msgpack.gz'
            payload=msgpack.packb({'step':0})+msgpack.packb({'step':1})[:-1]
            path.write_bytes(gzip.compress(payload))
            Path(str(path)+'.index.json').write_text(json.dumps({'sample_count':2}))
            with self.assertRaisesRegex(ValueError,'sealed count'):
                list(iter_truth_samples(directory))

    def test_different_episode_archives_are_not_silently_mixed(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            (directory/'truth_samples.msgpack.gz').touch()
            (directory/'truth_samples.jsonl.gz').touch()
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                truth_archive_path(directory)

    def test_indexed_step_identity_is_checked(self):
        for codec in ('jsonl','msgpack'):
            with self.subTest(codec=codec),tempfile.TemporaryDirectory() as name:
                directory=Path(name)
                store=GzipSampleStore(directory/f'truth_samples.{codec}.gz',codec=codec)
                store.append({'step':12});store.close()
                with self.assertRaisesRegex(ValueError,'requested physical step'):
                    read_truth_sample(directory,0)


if __name__=='__main__':unittest.main()
