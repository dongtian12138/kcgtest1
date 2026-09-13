"""Read run metadata without requiring a duplicate of the raw JSONL samples."""

import json
import gzip
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


def iter_truth_samples(directory):
    """Stream preserved physical samples, including concatenated gzip blocks."""
    directory=Path(directory)
    for name,opener in (("truth_samples.jsonl.gz",gzip.open),("truth_samples.jsonl",open)):
        path=directory/name
        if path.exists():
            with opener(path,"rt") as stream:
                for line in stream:yield json.loads(line)
            return
    import ijson
    with (directory/"trace.json").open("rb") as stream:
        yield from ijson.items(stream,"samples.item",use_float=True)


def read_truth_sample(directory,step):
    """Read one sealed indexed block instead of scanning a multi-GB episode."""
    from carts_v2.fast_json import loads
    directory=Path(directory);path=directory/'truth_samples.jsonl.gz'
    index_path=Path(str(path)+'.index.json')
    if index_path.exists():
        index=json.loads(index_path.read_text())
        if not 0<=step<index['sample_count']:raise IndexError(step)
        block=index['blocks'][step//index['block_size']]
        with path.open('rb') as f:
            f.seek(block['offset']);data=gzip.decompress(f.read(block['end']-block['offset']))
        row=loads(data.splitlines()[step-block['first']])
        if row['step']!=step:raise ValueError('indexed sample does not match requested physical step')
        return row
    return next((row for row in iter_truth_samples(directory) if row['step']==step),None)


def write_gzip_array(path,rows,*,mode="x",prepare=None):
    """Write one ordinary JSON array without duplicating all rows in memory."""
    from carts_v2.fast_json import dumps
    with gzip.open(path,mode+"t",encoding="utf-8",compresslevel=1) as stream:
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
