#!/usr/bin/env python3
"""Evaluate only the initial Body lift/hold, before any Nut handoff."""

import argparse
import gzip
import json
from pathlib import Path

import ijson
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def iter_truth_samples(directory):
    path = directory / "truth_samples.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)
    else:
        with (directory / "trace.json").open("rb") as stream:
            yield from ijson.items(stream, "samples.item", use_float=True)


def evaluate(directory):
    repository = Path(__file__).resolve().parents[3]
    output = directory / "initial_body_pickup_posthoc_v1.json"
    if output.exists():
        raise FileExistsError(output)
    hulls = []
    for name in ("plug_body", "coupling_nut"):
        path = repository / f"artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/{name}_visual_mesh.npz"
        points = np.load(path)["vertices_m"]
        hulls.append(points[ConvexHull(points).vertices])
    links = ("f1Link3", "f2Link2", "f3Link3")
    rows = []
    for sample in iter_truth_samples(directory):
        if sample["phase"] not in ("lift", "hold"):
            continue
        positions = np.asarray(sample["object_part_positions_m"])
        rotations = Rotation.from_quat(np.roll(np.asarray(
            sample["object_part_orientations_wxyz"]), -1, axis=1)).as_matrix()
        # Recover the fixed table datum from the existing recorded definition;
        # compute the actual clearance using source support vertices and tilt.
        table_z = min(positions[0, 2], positions[1, 2]+.0015493999235332012) - sample["object_bottom_clearance_m"]
        clearance = min(float((v @ r.T+p)[:, 2].min())
                        for v, r, p in zip(hulls, rotations, positions))-table_z
        body, nut = np.zeros(3, bool), np.zeros(3, bool)
        for header in sample["contacts"]["tensor_headers"]:
            if not any(c["normal_impulse_n_s"] > 1e-9 for c in header["contacts"]):
                continue
            for i, link in enumerate(links):
                if not any(p.endswith("/"+link) for p in header["paths"]):
                    continue
                body[i] |= any(p.endswith("/Body") for p in header["paths"])
                nut[i] |= any(p.endswith("/CouplingNut") for p in header["paths"])
        rows.append({"step": sample["step"], "phase": sample["phase"],
                     "simulation_time_s": sample["simulation_time_s"],
                     "source_surface_table_clearance_m": clearance,
                     "three_fingers_body_only": bool(body.all() and not nut.any()),
                     "table_contact": sample["contacts"]["object_table_positive_normal_impulse_n_s"] > 1e-9,
                     "robot_table_contact_records": sample["contacts"]["robot_table"],
                     "robot_fixture_contact_records": sample["contacts"]["robot_fixture"],
                     "unauthorized_robot_object_records": sample["contacts"]["robot_object_unauthorized"]})
    hold = [row for row in rows if row["phase"] == "hold"]
    if not rows or not hold:
        raise ValueError("the episode has no complete initial lift/hold to evaluate")
    dt = float(np.median(np.diff([row["simulation_time_s"] for row in rows])))
    if dt <= 0 or not np.allclose(np.diff([row["simulation_time_s"] for row in rows]),
                                  dt, rtol=0, atol=1e-9):
        raise ValueError("initial lift/hold samples must have uniform physics times")
    forces = []
    with gzip.open(directory / "wrist_ft_samples.json.gz", "rb") as stream:
        for row in ijson.items(stream, "item", use_float=True):
            if row["phase"] in ("lift", "hold"):
                forces.append(float(row["resultant_force_n"]))
    result = {"scope": "POSTRUN_INITIAL_BODY_PICKUP_ONLY_NOT_ASSEMBLY", "run": str(directory),
              "sample_count": len(rows), "hold_sample_count": len(hold),
              "physics_dt_s": dt, "hold_duration_s": len(hold)*dt,
              "three_fingers_body_only_throughout_lift_hold": all(r["three_fingers_body_only"] for r in rows),
              "three_fingers_body_only_throughout_hold": all(r["three_fingers_body_only"] for r in hold),
              "lift_hold_samples_without_three_finger_body_only_contact": [r["step"] for r in rows if not r["three_fingers_body_only"]],
              "maximum_source_surface_clearance_m": max(r["source_surface_table_clearance_m"] for r in rows),
              "minimum_source_surface_clearance_during_hold_m": min(r["source_surface_table_clearance_m"] for r in hold),
              "table_positive_contact_samples_during_hold": sum(r["table_contact"] for r in hold),
              "robot_table_contact_records": sum(r["robot_table_contact_records"] for r in rows),
              "robot_fixture_contact_records": sum(r["robot_fixture_contact_records"] for r in rows),
              "unauthorized_robot_object_records": sum(r["unauthorized_robot_object_records"] for r in rows),
              "maximum_wrist_resultant_during_lift_hold_n": max(forces),
              "clearance_method": "Minimum transformed source convex-hull support vertices; linear extrema equal those of the complete source vertices. Original outer surfaces are unchanged by the internal thread reconstruction.",
              "online_control_used": False}
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(evaluate(parser.parse_args().directory.resolve()), indent=2))
