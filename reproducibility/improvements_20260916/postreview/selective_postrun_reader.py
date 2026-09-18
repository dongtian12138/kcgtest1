"""Decode only fields needed by specified postrun reviews; raw archives unchanged."""
from pathlib import Path
import gzip,json,msgpack
from carts_v2.contact_codec import decode_extension

def review_rows(directory, *, hand_contacts=False):
    p=Path(directory);archive=p/'truth_samples.msgpack.gz';index=json.loads(Path(str(archive)+'.index.json').read_text())
    if index['format']!='CONCATENATED_GZIP_MSGPACK' or index['blocks'][-1]['end']!=archive.stat().st_size:
        raise ValueError('A sealed unchanged messagepack archive is required')
    wanted={'step','phase','simulation_time_s','object_part_positions_m','object_part_orientations_wxyz'}
    if hand_contacts:wanted.add('native_robot_link_pose_audit')
    expected=0
    with gzip.open(archive,'rb') as stream:
        u=msgpack.Unpacker(stream,raw=False,ext_hook=decode_extension)
        while True:
            try:count=u.read_map_header()
            except msgpack.OutOfData:break
            row={}
            for _ in range(count):
                key=u.unpack()
                if key in wanted:row[key]=u.unpack()
                elif hand_contacts and key=='contacts':
                    kept=[];table_impulse=None
                    for _ in range(u.read_map_header()):
                        field=u.unpack()
                        if field=='object_table_positive_normal_impulse_n_s':table_impulse=u.unpack();continue
                        if field!='poll_headers':u.skip();continue
                        for _ in range(u.read_array_header()):
                            paths=None;points=None
                            for _ in range(u.read_map_header()):
                                field=u.unpack()
                                if field=='paths':paths=u.unpack()
                                elif field=='contacts':
                                    if paths is None:raise ValueError('Contact paths must precede contact points')
                                    if any('/handbase_link' in x for x in paths[:2]):points=u.unpack()
                                    else:u.skip()
                                else:u.skip()
                            if points is not None:kept.append({'paths':paths,'contacts':points})
                    row['contacts']={'poll_headers':kept,'object_table_positive_normal_impulse_n_s':table_impulse}
                else:u.skip()
            if row['step']!=expected:raise ValueError('Discontinuous raw sequence')
            expected+=1
            yield row
    if expected!=index['sample_count']:raise ValueError('Raw count differs from sealed index')
