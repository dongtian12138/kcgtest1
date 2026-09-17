"""Read run metadata without requiring a duplicate of the raw JSONL samples."""

import json
import gzip
import io
from pathlib import Path


def read_metadata_field(directory, field):
    directory = Path(directory)
    metadata = directory / "trace_metadata.json"
    if metadata.exists():
        with metadata.open() as stream:
            return json.load(stream)[field]
    # Completed historical runs keep their original files unchanged.
    import ijson
    with (directory / "trace.json").open("rb") as stream:
        return next(ijson.items(stream, field, use_float=True))


def iter_control_samples(record):
    """Read old embedded controller samples or the new single append-only file."""
    if "samples" in record:
        yield from record["samples"]
    else:
        count = 0
        with Path(record["control_samples_file"]).open() as stream:
            for line in stream:
                count += 1
                yield json.loads(line)
        if count != record["sample_count"]:
            raise ValueError("controller sample file does not match its sealed count")


def truth_archive_path(directory):
    directory=Path(directory)
    binary=directory/'truth_samples.msgpack.gz'
    compressed=directory/'truth_samples.jsonl.gz'
    plain=directory/'truth_samples.jsonl'
    if binary.exists() and (compressed.exists() or plain.exists()):
        raise ValueError('ambiguous raw observation archives in one episode')
    return next((p for p in (binary,compressed,plain) if p.exists()),None)


def iter_truth_samples(directory):
    """Stream preserved physical samples, including concatenated gzip blocks."""
    from carts_v2.fast_json import loads
    directory=Path(directory)
    path=truth_archive_path(directory)
    if path is not None and path.name.endswith('.msgpack.gz'):
        from carts_v2.sample_store import msgpack_rows
        count=0
        with gzip.open(path,'rb') as stream:
            for row in msgpack_rows(stream):
                count+=1
                yield row
        index_path=Path(str(path)+'.index.json')
        if index_path.exists() and count!=json.loads(index_path.read_text())['sample_count']:
            raise ValueError('raw observation stream does not match its sealed count')
        return
    for name,opener in (("truth_samples.jsonl.gz",gzip.open),("truth_samples.jsonl",open)):
        path=directory/name
        if path.exists():
            with opener(path,"rt") as stream:
                for line in stream:yield loads(line)
            return
    import ijson
    with (directory/"trace.json").open("rb") as stream:
        yield from ijson.items(stream,"samples.item",use_float=True)


def read_truth_sample(directory,step):
    """Read one sealed indexed block instead of scanning a multi-GB episode."""
    from carts_v2.fast_json import loads
    directory=Path(directory);path=truth_archive_path(directory)
    if path is None:return next((row for row in iter_truth_samples(directory) if row['step']==step),None)
    index_path=Path(str(path)+'.index.json')
    if index_path.exists():
        index=json.loads(index_path.read_text())
        if not 0<=step<index['sample_count']:raise IndexError(step)
        block=index['blocks'][step//index['block_size']]
        with path.open('rb') as f:
            f.seek(block['offset']);data=gzip.decompress(f.read(block['end']-block['offset']))
        if index['format']=='CONCATENATED_GZIP_MSGPACK':
            from carts_v2.sample_store import msgpack_rows
            unpacker=msgpack_rows(io.BytesIO(data))
            for _ in range(step-block['first']):unpacker.skip()
            row=unpacker.unpack()
        elif index['format']=='CONCATENATED_GZIP_JSONL':
            row=loads(data.splitlines()[step-block['first']])
        else:raise ValueError('unknown indexed raw observation codec')
        if row['step']!=step:raise ValueError('indexed sample does not match requested physical step')
        return row
    return next((row for row in iter_truth_samples(directory) if row['step']==step),None)


def write_gzip_array(path,rows,*,mode="x",prepare=None):
    """Write one ordinary JSON array without duplicating all rows in memory."""
    from carts_v2.fast_json import dumps
    from carts_v2.recording_compression import gzip_backend
    backend=gzip_backend(getattr(rows,'compression_backend','gzip'))
    known_numeric_preparer=(prepare is None or (
        getattr(prepare,'__name__',None)=='_json_ready' and
        getattr(prepare,'__module__','').split('.')[-1] in
        ('run_grasp_lift','te_foundationpose_handoff_runtime')))
    if getattr(rows,'normalized_numeric_snapshots',False) and known_numeric_preparer:
        prepare=None
    with backend.open(path,mode+"t",encoding="utf-8",compresslevel=1) as stream:
        stream.write('[')
        for index,row in enumerate(rows):
            if index:stream.write(',')
            stream.write(dumps(prepare(row) if prepare else row))
        stream.write(']\n')


def without_cyclic_gc(function,*args,**kwargs):
    """Avoid repeated heap scans during post-run JSON-tree processing only."""
    import gc
    enabled=gc.isenabled();gc.disable()
    try:
        return function(*args,**kwargs)
    finally:
        if enabled:gc.enable()


def iter_truth_fields(directory, fields, first_step=0, last_step=None):
    """Project an ended indexed MessagePack archive; retain the original archive.

    Reviews that need only poses avoid constructing unrelated contact arrays.
    """
    import msgpack
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
            from carts_v2.contact_codec import decode_extension
            reader = msgpack.Unpacker(raw=False, max_buffer_size=0, ext_hook=decode_extension)
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
