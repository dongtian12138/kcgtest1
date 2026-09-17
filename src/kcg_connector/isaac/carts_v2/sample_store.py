"""A bounded-memory sequence backed by concatenated gzip JSONL or MessagePack.

The robot recorder still appends dictionaries and readers still use sequence
indices. Closed blocks are read on demand; the archive is also readable with
gzip.open plus the declared codec. Raw samples are retained without decimation.
"""
from collections import OrderedDict
from collections.abc import Sequence
import gzip
import io
import json
from pathlib import Path
try:
    from .fast_json import dumps as encode_row, loads as decode_row, _default as normalize_value
except ImportError:
    from fast_json import dumps as encode_row, loads as decode_row, _default as normalize_value


def msgpack_rows(stream):
    """Read the same nested numeric observations without executable objects."""
    import msgpack
    try:
        from .contact_codec import decode_extension
    except ImportError:
        from contact_codec import decode_extension
    return msgpack.Unpacker(stream, raw=False, ext_hook=decode_extension)


class _SampleSlice(Sequence):
    def __init__(self,store,indices):
        self.store,self.indices=store,indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        if isinstance(index,slice):return _SampleSlice(self.store,self.indices[index])
        return self.store[self.indices[index]]

    def __iter__(self):
        for index in self.indices:yield self.store[index]


class GzipSampleStore(Sequence):
    def __init__(self,path,*,block_size=512,cache_blocks=2,codec='jsonl',compression_backend='gzip'):
        if block_size<1 or cache_blocks<1:raise ValueError('positive block/cache sizes required')
        if codec not in ('jsonl','msgpack'):raise ValueError('unsupported raw observation codec')
        self.codec=codec
        try:
            from .recording_compression import gzip_backend
        except ImportError:
            from recording_compression import gzip_backend
        self._gzip=gzip_backend(compression_backend)
        self.compression_backend=compression_backend
        self.format='CONCATENATED_GZIP_'+codec.upper()
        if codec=='msgpack':
            import msgpack
            try:
                from .contact_codec import encode_extension
            except ImportError:
                from contact_codec import encode_extension
            self._packer=msgpack.Packer(use_bin_type=True,default=encode_extension)
        self.name=str(path);self.block_size=block_size;self.cache_blocks=cache_blocks
        self._file=open(path,'xb');self._live=[];self._blocks=[];self._cache=OrderedDict()
        self._count=0;self.closed=False

    def __len__(self):
        return self._count

    def append(self,row):
        if self.closed:raise ValueError('sample archive is closed')
        if len(self._live)==self.block_size:self._seal_block()
        self._live.append(row);self._count+=1

    def _seal_block(self):
        if not self._live:return
        offset=self._file.tell()
        # Finish each member, so every closed block has its own checksum and
        # can be read without retaining or decompressing previous samples.
        with self._gzip.GzipFile(fileobj=self._file,mode='wb',compresslevel=1,mtime=0) as member:
            for row in self._live:
                member.write(self._packer.pack(row) if self.codec=='msgpack'
                             else (encode_row(row)+'\n').encode())
        self._file.flush()
        self._blocks.append({'first':self._count-len(self._live),'count':len(self._live),
                             'offset':offset,'end':self._file.tell()})
        self._remember(len(self._blocks)-1,self._live)
        self._live=[]

    def _remember(self,index,rows):
        self._cache[index]=rows;self._cache.move_to_end(index)
        while len(self._cache)>self.cache_blocks:self._cache.popitem(last=False)
        return rows

    def _read_block(self,index):
        if index in self._cache:
            self._cache.move_to_end(index);return self._cache[index]
        block=self._blocks[index]
        with open(self.name,'rb') as stream:
            stream.seek(block['offset']);data=stream.read(block['end']-block['offset'])
        raw=self._gzip.decompress(data)
        rows=(list(msgpack_rows(io.BytesIO(raw))) if self.codec=='msgpack'
              else [decode_row(line) for line in raw.splitlines()])
        if len(rows)!=block['count']:raise ValueError('sample archive block count differs')
        return self._remember(index,rows)

    def __getitem__(self,index):
        if isinstance(index,slice):return _SampleSlice(self,range(self._count)[index])
        if index<0:index+=self._count
        if not 0<=index<self._count:raise IndexError(index)
        live_first=self._count-len(self._live)
        if index>=live_first:return self._live[index-live_first]
        block_index=index//self.block_size
        return self._read_block(block_index)[index-self._blocks[block_index]['first']]

    def __iter__(self):
        # Snapshot the logical length. Recording and post-run evaluation use
        # this in one thread, and readers never extend the archive.
        for index in range(self._count):yield self[index]

    def close(self):
        if self.closed:return
        self._seal_block();self._file.close();self.closed=True
        Path(self.name+'.index.json').write_text(json.dumps({'format':self.format,
            'compression_backend':self.compression_backend,'contact_extension_schema':'f64le_xyz_normal_impulse_separation_v1_code42',
            'sample_count':self._count,'block_size':self.block_size,'blocks':self._blocks},indent=2)+'\n')

    def __del__(self):
        if hasattr(self,'closed') and not self.closed:
            self.close()
