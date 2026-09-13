"""A bounded-memory sequence backed by ordinary concatenated gzip JSONL.

The robot recorder still appends dictionaries and readers still use sequence
indices. Closed blocks are read on demand; the archive is also readable with
gzip.open without this module. Raw samples are retained without decimation.
"""
from collections import OrderedDict
from collections.abc import Sequence
import gzip
import json
from pathlib import Path
try:
    from .fast_json import dumps as encode_row, loads as decode_row
except ImportError:
    from fast_json import dumps as encode_row, loads as decode_row


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
    def __init__(self,path,*,block_size=512,cache_blocks=2):
        if block_size<1 or cache_blocks<1:raise ValueError('positive block/cache sizes required')
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
        with gzip.GzipFile(fileobj=self._file,mode='wb',compresslevel=1,mtime=0) as member:
            for row in self._live:
                member.write((encode_row(row)+'\n').encode())
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
        rows=[decode_row(line) for line in gzip.decompress(data).splitlines()]
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
        Path(self.name+'.index.json').write_text(json.dumps({'format':'CONCATENATED_GZIP_JSONL',
            'sample_count':self._count,'block_size':self.block_size,'blocks':self._blocks},indent=2)+'\n')

    def __del__(self):
        if hasattr(self,'closed') and not self.closed:
            self.close()
