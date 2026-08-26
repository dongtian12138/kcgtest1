#!/usr/bin/env python3
"""Scan frozen remaining candidates for raw and collider task-first order."""
from __future__ import annotations
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import diagnose_q09_a13_contact_asset_gap as d
from kcg_connector.grasp.carts_v2.b0_surface_semantics import bind_b0_external_load_bearing_surfaces
from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model
from kcg_connector.grasp.carts_v2.models import load_v2_inputs

ROOT, RUN = d.ROOT, d.RUN
SOURCE = RUN / "funnel_normal_first_b0_exact_run02/result.json"
TELEMETRY = RUN / "contact_telemetry_positive_control/result.json"
OUTPUT = RUN / "remaining_candidate_task_first/result.json"
REJECTED = "contactopt_g_q09_a13_z3_p0"

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    source, telemetry = json.loads(SOURCE.read_text()), json.loads(TELEMETRY.read_text())
    if telemetry["contact_infrastructure_after"] != "CONTACT_TELEMETRY_UNVERIFIED":
        raise ValueError("this fail-closed scan is bound to unverified telemetry")
    rows = [x for x in source["candidates"] if x.get("local_isaac_input_ready") is True]
    if len(rows) != 5 or {x["candidate_id"] for x in rows if x["candidate_id"] == REJECTED} != {REJECTED}:
        raise ValueError("frozen five-input identity changed")
    rows = [x for x in rows if x["candidate_id"] != REJECTED]
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(ROOT, config_path=d.CONFIG, object_id=d.OBJECT_ID))
    surface, obj = inputs.task_grip_surfaces["finger_1_pad"], inputs.object_contract.model.mesh
    raw = d._raw_first_finger(surface); object_raw = build_fcl_bvh_model(obj.vertices_m, obj.faces)
    object_faces = [("object", i) for i in range(len(obj.faces))]
    task_model = build_fcl_bvh_model(surface.points_local_m, surface.faces)
    task_faces = [("f1Link3", int(i)) for i in surface.source_face_indices]
    manifest = json.loads(d.OBJECT_MANIFEST.read_text()); isaac = ROOT.parent / "isaacsim/.conda-env"; env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{isaac/'lib'}:{env.get('LD_LIBRARY_PATH','')}"
    worker = subprocess.run([str(isaac/"bin/python"), str(Path(d.__file__).resolve()), "--usd-worker",
        str(d.HAND_USD), str(ROOT/manifest["asset"])], text=True, capture_output=True, env=env, check=True)
    usd = json.loads(worker.stdout.rsplit("__PHYSX_RUNTIME_OFFSETS_JSON__=", 1)[1].splitlines()[0])
    object_collision, object_lineage = d._combine(usd["object"])
    pieces = [(x["link"], x["path"], *d._combine([x])) for x in usd["hand"]]
    def scan(candidate):
        pre = candidate["input_seed"]["pregrasp_joint_positions_rad"]
        base = np.asarray(candidate["target_world_from_handbase_row_major"], float).reshape(4, 4); cache = {}
        def evaluate(q):
            key = round(float(q), 12)
            if key in cache: return cache[key]
            values = {"f1j1":pre[0], "f1j2":q, "f1j3":q, "f2j1":pre[2], "f2j2":pre[2],
                      "f3j1":pre[0], "f3j2":pre[3], "f3j3":pre[3]}
            fk = d._fk(inputs.hand_model, {"world_from_handbase_row_major":base.ravel(), "joint_positions_rad":values})
            task = d._query(task_model, fk["f1Link3"], object_raw, inputs.frozen_world_from_object, task_faces, object_faces)
            non = [d._query(item["model"], fk[link], object_raw, inputs.frozen_world_from_object,
                [(link, int(i)) for i in item["faces"]], object_faces) for link, item in raw.items()]
            task_tri = d._world_triangles(surface.triangles_local_m, fk["f1Link3"])
            non_tri = np.concatenate([d._world_triangles(item["mesh"].face_vertices_m[item["faces"]], fk[link]) for link,item in raw.items()])
            collider = [False, False]
            for link, _path, model, lineage in pieces:
                hit = d._query(model, fk[link], object_collision, inputs.frozen_world_from_object, lineage, object_lineage)
                if hit["collision"]:
                    point = np.asarray(hit["left_witness_world"]); is_task = link == "f1Link3" and d._point_distance(point, task_tri) < d._point_distance(point, non_tri)
                    collider[0 if is_task else 1] = True
            best_non = min(non, key=lambda x:(not x["collision"], x["distance_m"]))
            cache[key] = (task["collision"], best_non["collision"], *collider, best_non["distance_m"]); return cache[key]
        q0, q1 = float(pre[1]), float(candidate["proxy_interval"]["finger_intervals"][0]["proxy_q_safe_max_rad"])
        grid = np.linspace(q0, q1, 25); samples = [evaluate(q) for q in grid]
        def root(field):
            found = next((i for i,x in enumerate(samples) if x[field]), None)
            if found is None: return None
            lo, hi = (q0, q0) if found == 0 else (float(grid[found-1]), float(grid[found]))
            for _ in range(16):
                mid=(lo+hi)/2
                if evaluate(mid)[field]: hi=mid
                else: lo=mid
            return hi
        qr, qn, qc, qcn = (root(i) for i in range(4)); lead = None if qr is None or qn is None else qn-qr
        risk = None if qr is None else evaluate(qr)[4] <= 0.00075
        return {"candidate_id":candidate["candidate_id"], "nominal_12n_feasible":candidate.get("nominal_12n_task_pass") is True,
            "q_task_contact_raw":qr, "q_non_task_intersection_raw":qn, "task_first_lead_raw":lead,
            "q_task_contact_physx":None, "q_non_task_contact_physx":None, "task_first_lead_physx":None,
            "q_task_collider_intersection_surrogate":qc, "q_non_task_collider_intersection_surrogate":qcn,
            "collider_task_first_lead_surrogate":None if qc is None or qcn is None else qcn-qc,
            "risk_within_0p75mm_at_raw_task_contact":risk}
    results = [scan(row) for row in rows]
    report = {"schema_version":"remaining_candidate_task_first_v1", "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"OFFLINE_PASS", "hardware_authorized":False, "generator_rerun":False, "candidate_count":4,
        "permanent_rejection":{REJECTED:"REJECTED_NON_TASK_GEOMETRY_PRECEDES_TASK_CONTACT"},
        "physx_contact_fields_status":"UNVERIFIED_NULL_BECAUSE_CONTACT_TELEMETRY_UNVERIFIED",
        "decision":"PARKED_NO_DYNAMIC_SELECTION", "candidates":results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(report, indent=2)+"\n"); print(OUTPUT)

if __name__ == "__main__": main()
