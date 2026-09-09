from pathlib import Path
import sys
import tracemalloc

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaac'))
from carts_v2.disk_samples import DiskSamples


def test_full_history_and_phase_slices_survive_close(tmp_path):
    samples=DiskSamples(tmp_path/'truth.jsonl')
    expected=[{'step':i,'phase':'grasp' if i<3 else 'turn',
               'contacts':[{'force':i/10,'label':'胶垫'}]} for i in range(7)]
    for row in expected:samples.append(row)
    assert len(samples)==len(expected)
    for index in [0,2,-1,-7]:assert samples[index]==expected[index]
    assert list(samples[1:6:2])==expected[1:6:2]
    assert list(samples[::-1][1:4])==expected[::-1][1:4]
    samples.close()
    assert list(samples)==expected
    assert [r for r in samples if r['phase']=='turn']==expected[3:]
    with pytest.raises(IndexError):_ = samples[7]
    with pytest.raises(IndexError):_ = samples[-8]
    with pytest.raises(FileExistsError):DiskSamples(tmp_path/'truth.jsonl')


def test_contact_payload_memory_does_not_grow_with_history(tmp_path):
    samples=DiskSamples(tmp_path/'truth.jsonl')
    tracemalloc.start()
    try:
        for i in range(40):
            samples.append({'step':i,'contacts':str(i)+'x'*300_000})
        retained,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop();samples.close()
    assert samples.path.stat().st_size>12_000_000
    assert retained<1_000_000
    assert peak<3_000_000
    assert samples[0]['step']==0
    assert samples[-1]['step']==39
