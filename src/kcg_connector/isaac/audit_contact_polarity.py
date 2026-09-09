#!/usr/bin/env python3
"""Check signed native contact impulses and forbidden contacts after a run."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(directory):
    counters = {name: 0 for name in ("robot_table", "robot_fixture", "robot_socket",
                                     "robot_object_nonterminal", "body_socket", "nut_socket", "object_table")}
    newly_detected = {name: [] for name in ("robot_table", "robot_fixture", "robot_socket")}
    signed, negative, opposite_pairs, example = 0, 0, 0, None
    steps = 0
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            s = json.loads(line)
            steps += 1
            present, known = set(), {}
            for h in s["contacts"]["tensor_headers"]:
                paths = h["paths"]
                pair = tuple(sorted(paths))
                impulses = np.asarray([c["impulse_n_s"] for c in h["contacts"]]).reshape(-1, 3)
                scalars = np.asarray([c["normal_impulse_n_s"] for c in h["contacts"]])
                signed += int(np.sum(np.abs(scalars) > 1e-9))
                negative += int(np.sum(scalars < -1e-9))
                total = impulses.sum(0)*(1 if paths[0] == pair[0] else -1)
                if pair in known:
                    opposite_pairs += 1
                    if not np.allclose(total, known[pair][0], rtol=1e-6, atol=1e-9):
                        raise ValueError("opposite actor contact vectors disagree")
                    if example is None and len(scalars) and scalars.sum()*known[pair][1] < 0:
                        example = {"step": s["step"], "pair": pair,
                                   "two_sensor_side_scalar_sums_n_s": [float(scalars.sum()), known[pair][1]],
                                   "canonical_vector_sum_n_s": total.tolist()}
                    continue
                known[pair] = (total, float(scalars.sum()))
                if not np.any(np.linalg.norm(impulses, axis=1) > 1e-9):
                    continue
                robot = next((p for p in paths if p.startswith("/World/HandArm/")), None)
                body = any(p.endswith("/TE_J35FreeSplitPlug/Body") for p in paths)
                nut = any(p.endswith("/TE_J35FreeSplitPlug/CouplingNut") for p in paths)
                socket = any("FixedReceptaclePose" in p for p in paths)
                table = any(p.endswith("/Table") for p in paths)
                fixture = any(p.endswith("/FixedFixture") for p in paths)
                if robot and table:
                    present.add("robot_table")
                if robot and fixture:
                    present.add("robot_fixture")
                if robot and socket:
                    present.add("robot_socket")
                if robot and (body or nut) and not robot.endswith(("/f1Link3", "/f2Link2", "/f3Link3")):
                    present.add("robot_object_nonterminal")
                if body and socket:
                    present.add("body_socket")
                if nut and socket:
                    present.add("nut_socket")
                if (body or nut) and table:
                    present.add("object_table")
            for name in present:
                counters[name] += 1
                if name in newly_detected and name in s["contacts"] and s["contacts"][name] == 0:
                    newly_detected[name].append(s["step"])
    result = {"scope": "COMPLETED_EPISODE_SIGNED_NORMAL_IMPULSE_AUDIT",
              "run": str(directory), "sample_count": steps,
              "nonzero_signed_scalar_records": signed, "negative_scalar_records": negative,
              "mirrored_pairs_checked": opposite_pairs, "mirrored_signed_scalar_example": example,
              "physical_contact_presence_from": "NONZERO_IMPULSE_VECTOR_NORM; MIRRORED_ACTOR_PAIR_COUNTED_ONCE",
              "contact_sample_counts": counters,
              "newly_detected_by_absolute_impulse_against_existing_named_counters": newly_detected,
              "online_control_used": False,
              "not_a_source_pad_surface_or_full_assembly_verdict": True}
    output = directory / "contact_polarity_posthoc_v1.json"
    output.open("x").write(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(audit(parser.parse_args().directory.resolve()), indent=2))
