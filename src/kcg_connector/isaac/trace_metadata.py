"""Read run metadata without requiring a duplicate of the raw JSONL samples."""

import json
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
