"""Read selected fields of an already sealed archive without building contact trees.

The archive remains unchanged and retains every native contact point. This is
an offline reader for reviews that only require poses and phase identifiers.
"""
import gzip
import json
from pathlib import Path

import msgpack


def iter_sealed_fields(directory, fields, first_step=0, last_step=None):
    directory = Path(directory)
    archive = directory / 'truth_samples.msgpack.gz'
    index = json.loads(Path(str(archive) + '.index.json').read_text())
    if not index['blocks'] or index['blocks'][-1]['end'] != archive.stat().st_size:
        raise ValueError('Review requires a matching sealed archive')
    if index['format'] != 'CONCATENATED_GZIP_MSGPACK':
        raise ValueError('This projection reader requires the recorded MessagePack format')
    if not any((directory / name).is_file() for name in ('motion_timing.json', 'source_stage_probe_result.json')):
        raise ValueError('Review requires a recorded end of physical motion')
    last = index['sample_count'] - 1 if last_step is None else int(last_step)
    if not 0 <= first_step <= last < index['sample_count']:
        raise ValueError('Requested field interval is outside the sealed archive')
    wanted = set(fields) | {'step'}
    expected = int(first_step)
    with archive.open('rb') as stream:
        blocks = index['blocks'][first_step // index['block_size']:last // index['block_size'] + 1]
        for block in blocks:
            stream.seek(block['offset'])
            reader = msgpack.Unpacker(raw=False, max_buffer_size=0)
            reader.feed(gzip.decompress(stream.read(block['end'] - block['offset'])))
            for _ in range(block['count']):
                row = {}
                for _ in range(reader.read_map_header()):
                    key = reader.unpack()
                    if key in wanted:
                        row[key] = reader.unpack()
                    else:
                        reader.skip()
                if row['step'] < first_step or row['step'] > last:
                    continue
                if row['step'] != expected or set(row) != wanted:
                    raise ValueError('Projected archive row is incomplete or out of sequence')
                expected += 1
                yield row
    if expected != last + 1:
        raise ValueError('Projected archive ended before its declared sample count')
