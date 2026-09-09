"""Read-only contact helpers for installed Isaac Sim 6.0.1 / PhysX 110.1.13.

No Isaac imports at module load, no USD/physics mutation, no controller input.
Build RigidPrim only on existing rigid actors, never on child collision shapes:
its constructor automatically applies RigidBodyAPI to the paths it wraps.

SDK references:
  isaacsim.core.experimental.prims/impl/rigid_prim.py:1490,1550,1606,1664
  omni.physx/bindings/_physx.pyi:442,573,809,1328

raw_contact_data's seventh item is OTHER ACTOR IDs, not shape IDs. Its headers
are per sensor only. get_contact_force_data and get_friction_data are filtered
and their headers are sensor x filter. Passing dt=dt returns forces already.

For own-shape identity use get_full_contact_report: it contains collider0/1 and
separate normal-contact and friction-anchor arrays. Read once per physics step
and compare summed report forces to the tensor forces before treating missing
GPU report entries as zero force. Do not add the two backends together.
"""
from collections import defaultdict
import numpy as np


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _wrench(forces, points, origin):
    forces = np.asarray(forces, float).reshape(-1, 3)
    points = np.asarray(points, float).reshape(-1, 3)
    return np.r_[forces.sum(axis=0), np.cross(points - origin, forces).sum(axis=0)]


def _slice(start, count, capacity):
    start, count = int(start), int(count)
    if count == 0:
        return slice(0, 0)  # unused starts may contain an unsigned sentinel
    if start < 0 or count < 0 or start + count > capacity:
        raise RuntimeError("Native contact header exceeds its supplied buffer")
    return slice(start, start + count)


def read_filtered_contacts(view, dt, origins_world_m, *, active_only=False):
    """Return per-sensor/per-filter normals, friction, torque and separation.

    `view` is an existing experimental RigidPrim with contact_filter_paths.
    `origins_world_m` has one chosen wrench origin per sensor, shape (N,3).
    Each row retains sensor/filter INDEX; label using the view's explicit paths.
    Use a single Body sensor if constructing a new view. SDK special-cases
    len(filters)==len(sensors) as one filter per sensor, NOT all-to-all.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive physics time step required")
    f, p, n, separation, count, start = (
        _host(x) for x in view.get_contact_force_data(dt=dt)
    )
    ff, fp, fcount, fstart = (_host(x) for x in view.get_friction_data(dt=dt))
    f, separation = f.reshape(-1), separation.reshape(-1)
    origins = np.asarray(origins_world_m, float).reshape(-1, 3)
    if count.ndim != 2 or count.shape != fcount.shape or len(origins) != count.shape[0]:
        raise RuntimeError("Unexpected sensor x filter layout; inspect configured view paths")
    rows = []
    for sensor in range(count.shape[0]):
        for filter_index in range(count.shape[1]):
            if active_only and not count[sensor, filter_index] and not fcount[sensor, filter_index]:
                continue
            sl = _slice(start[sensor, filter_index], count[sensor, filter_index], len(f))
            fs = _slice(fstart[sensor, filter_index], fcount[sensor, filter_index], len(ff))
            normal_force = f[sl, None] * n[sl]
            rows.append({
                "sensor_index": sensor, "filter_index": filter_index,
                "normal_count": int(count[sensor, filter_index]),
                "friction_count": int(fcount[sensor, filter_index]),
                "normal_wrench_n_nm": _wrench(normal_force, p[sl], origins[sensor]).tolist(),
                "friction_wrench_n_nm": _wrench(ff[fs], fp[fs], origins[sensor]).tolist(),
                "normal_load_n": float(np.linalg.norm(normal_force, axis=1).sum()),
                "maximum_penetration_m": float(np.maximum(-separation[sl], 0).max()) if len(f[sl]) else None,
            })
    return rows


def read_shape_contact_pairs(contact_interface, dt, actor_path, origin_world_m, *, decode_path=None):
    """Read this physics step, retaining the exact own/other collision paths.

    Call after each physics update (the existing physics-step callback is also
    suitable). `origin_world_m` is the SAME world origin used by tensor checks.
    Includes normal CONTACT impulse and separate FRICTION ANCHOR impulse from
    the full report. Sign is actor0 positive / actor1 negative, as used by the
    existing diagnose_compliant_contact_stiffness.py; verify aggregate polarity
    against tensor data in the actual run.

    Empty output means no populated report for this actor; it is NOT proof of
    zero physical force. In GPU configurations report data may be incomplete.
    Never infer own collision shape from other_actor_ids in tensor raw data.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive physics time step required")
    if decode_path is None:
        from pxr import PhysicsSchemaTools
        decode_path = lambda value: str(PhysicsSchemaTools.intToSdfPath(value))
    origin = np.asarray(origin_world_m, float).reshape(3)
    headers, data, anchors = contact_interface.get_full_contact_report()
    rows = []
    for h in headers:
        actors = (decode_path(h.actor0), decode_path(h.actor1))
        if actor_path not in actors:
            continue
        own_index = actors.index(actor_path)
        sign = 1.0 if own_index == 0 else -1.0
        shapes = (decode_path(h.collider0), decode_path(h.collider1))
        ns = _slice(h.contact_data_offset, h.num_contact_data, len(data))
        fs = _slice(h.friction_anchors_offset, h.num_friction_anchors_data, len(anchors))
        normal_points, normal_forces, separations = [], [], []
        materials = set()
        for index in range(ns.start, ns.stop):
            record = data[index]
            normal_points.append(np.asarray(record.position, float))
            normal_forces.append(sign * np.asarray(record.impulse, float) / dt)
            separations.append(float(record.separation))
            materials.add((decode_path(record.material0), decode_path(record.material1)))
        friction_points, friction_forces = [], []
        for index in range(fs.start, fs.stop):
            anchor = anchors[index]
            friction_points.append(np.asarray(anchor.position, float))
            friction_forces.append(sign * np.asarray(anchor.impulse, float) / dt)
        rows.append({
            "own_collider": shapes[own_index], "other_collider": shapes[1-own_index],
            "other_actor": actors[1-own_index],
            "normal_count": len(normal_points), "friction_count": len(friction_points),
            "normal_wrench_n_nm": _wrench(normal_forces, normal_points, origin).tolist(),
            "friction_wrench_n_nm": _wrench(friction_forces, friction_points, origin).tolist(),
            "normal_load_n": float(np.linalg.norm(np.asarray(normal_forces).reshape(-1, 3), axis=1).sum()),
            "maximum_penetration_m": float(np.maximum(-np.asarray(separations), 0).max()) if separations else None,
            "reported_material_pairs_actor_order": [list(pair) for pair in sorted(materials)],
        })
    return rows


def group_shape_pairs(rows, own_collider_groups, other_collider_groups=None):
    """Aggregate exact paths, e.g. core/band/seal plus other-side Clip000..127.

    Other-side mapping takes precedence, allowing all pin/clip pairs to map to
    'contacts128' while core/Socket, band/Socket and seal/Socket remain distinct.
    Unknown pairs retain their paths in the 'unclassified_pairs' return item.
    """
    other_collider_groups = other_collider_groups or {}
    groups = defaultdict(lambda: {"normal_wrench_n_nm": np.zeros(6),
                                  "friction_wrench_n_nm": np.zeros(6),
                                  "normal_count": 0, "friction_count": 0,
                                  "normal_load_n": 0., "maximum_penetration_m": None})
    unknown = []
    for row in rows:
        label = other_collider_groups.get(row["other_collider"], own_collider_groups.get(row["own_collider"]))
        if label is None:
            unknown.append([row["own_collider"], row["other_collider"]])
            continue
        value = groups[label]
        for key in ("normal_wrench_n_nm", "friction_wrench_n_nm"):
            value[key] += row[key]
        for key in ("normal_count", "friction_count", "normal_load_n"):
            value[key] += row[key]
        depth = row["maximum_penetration_m"]
        if depth is not None:
            value["maximum_penetration_m"] = max(value["maximum_penetration_m"] or 0., depth)
    result = {label: {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in entry.items()} for label, entry in groups.items()}
    return {"groups": result, "unclassified_pairs": unknown,
            "populated_shape_report": any(row["normal_count"] or row["friction_count"] for row in rows)}


def create_link_velocity_reader(physics_sim_view, articulation_root_path, actor_paths):
    """Create an independent native articulation read view AFTER world.reset.

    Use the already active simulation view (e.g. the one used by RigidPrim),
    not a fresh simulator, not another backend and not any set/reset operation.
    Native API docs explicitly support articulation links in RigidBodyView;
    this is a cross-check of this run's readbacks, not a claim that the existing
    API is categorically invalid.
    """
    view = physics_sim_view.create_articulation_view(articulation_root_path)
    if view.count != 1:
        raise RuntimeError(f"Expected one existing articulation, got {view.count}")
    paths = list(view.link_paths[0])
    indices = [paths.index(path) for path in actor_paths]
    return view, indices


def read_link_velocity_crosscheck(view, indices):
    """Return independent link pose, COM world velocities and passive q/dq.

    get_link_velocities: (articulations, links, 6), world v_COM and omega rad/s.
    get_link_transforms: (articulations, links, 7), world p and quaternion xyzw.
    Copy buffers immediately; tensors may be reused by subsequent native reads.
    """
    velocities = _host(view.get_link_velocities()).copy()[0, indices]
    transforms = _host(view.get_link_transforms()).copy()[0, indices]
    q = _host(view.get_dof_positions()).copy()[0]
    dq = _host(view.get_dof_velocities()).copy()[0]
    return {
        "link_paths": [view.link_paths[0][i] for i in indices],
        "link_positions_world_m": transforms[:, :3].tolist(),
        "link_quaternions_xyzw": transforms[:, 3:].tolist(),
        "link_com_linear_velocity_world_m_s": velocities[:, :3].tolist(),
        "link_angular_velocity_world_rad_s": velocities[:, 3:].tolist(),
        "native_dof_positions_rad": q.tolist(),
        "native_dof_velocities_rad_s": dq.tolist(),
        "scope": "INDEPENDENT_NATIVE_ARTICULATION_READ_VIEW_NO_MODEL_OR_CONTROL_CHANGE",
    }
