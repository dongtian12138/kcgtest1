"""Same complete shape-pair report, without per-point NumPy allocations.

Zero impulses contribute no wrench: their count, separation and material IDs
are still recorded, while their unused positions need not be converted. This
changes readback cost only, not native contacts, sampling or physics settings.
"""
import math


def read_shape_contact_pairs_fast(contact_interface, dt, actor_path, origin_world_m, *, decode_path):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive physics time step required')
    ox, oy, oz = map(float, origin_world_m)
    headers, data, anchors = contact_interface.get_full_contact_report()
    result = []

    def wrench(records, start, count, sign, with_separation):
        fx = fy = fz = tx = ty = tz = load = 0.
        depth = 0. if count else None
        materials = set()
        if count and (start < 0 or start+count > len(records)):
            raise RuntimeError('Native contact header exceeds its supplied buffer')
        for i in range(start, start+count):
            record = records[i]
            x, y, z = record.impulse
            if with_separation:
                depth = max(depth, -float(record.separation))
                materials.add((record.material0, record.material1))
                load += math.sqrt(x*x+y*y+z*z)
            if x or y or z:
                px, py, pz = record.position
                px -= ox; py -= oy; pz -= oz
                fx += x; fy += y; fz += z
                tx += py*z-pz*y
                ty += pz*x-px*z
                tz += px*y-py*x
        scale = sign/dt
        return [value*scale for value in (fx, fy, fz, tx, ty, tz)], load/dt, depth, materials

    for h in headers:
        actors = (decode_path(h.actor0), decode_path(h.actor1))
        if actor_path not in actors:
            continue
        own = actors.index(actor_path)
        sign = 1. if own == 0 else -1.
        shapes = (decode_path(h.collider0), decode_path(h.collider1))
        nc, fc = int(h.num_contact_data), int(h.num_friction_anchors_data)
        # Unused native starts may contain an unsigned sentinel.
        ns = int(h.contact_data_offset) if nc else 0
        fs = int(h.friction_anchors_offset) if fc else 0
        nw, load, depth, materials = wrench(data, ns, nc, sign, True)
        fw, _, _, _ = wrench(anchors, fs, fc, sign, False)
        decoded_materials = {(decode_path(a), decode_path(b)) for a, b in materials}
        result.append({
            'own_collider': shapes[own], 'other_collider': shapes[1-own],
            'other_actor': actors[1-own], 'normal_count': nc, 'friction_count': fc,
            'normal_wrench_n_nm': nw, 'friction_wrench_n_nm': fw,
            'normal_load_n': load, 'maximum_penetration_m': depth,
            'reported_material_pairs_actor_order': [list(pair) for pair in sorted(decoded_materials)],
        })
    return result
