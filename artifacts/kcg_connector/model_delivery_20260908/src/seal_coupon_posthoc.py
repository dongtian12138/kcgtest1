#!/usr/bin/env python3
"""Read one isolated seal coupon and recommend one point-stiffness mapping.

No scene is edited. Reference material/geometry are fixed before the coupon.
The result is a numerical mapping recommendation, not an assembly success gate
or a claim that the TE elastomer was identified.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(coupon, reference):
    coupon, reference = Path(coupon), Path(reference)
    result = json.loads((coupon / "result.json").read_text())
    if result["scope"] != "SINGLE_SEAL_COUPON_NOT_ASSEMBLY":
        raise ValueError("mixed whole-connector loads are not accepted for this mapping")
    rows = [json.loads(line) for line in (coupon / "samples.jsonl").read_text().splitlines()]
    curve = json.loads(reference.read_text())
    ref_depth = np.array([r["depth_m"] for r in curve["rows"]])
    ref_normal = np.array([r["normal_force_sum_n"] for r in curve["rows"]])
    ref_axial = np.array([r["forward_total_axial_n"] for r in curve["rows"]])
    bins = []
    compression = [r for r in rows if r["phase"] == "compression"]
    commands = np.array([r["command_depth_m"] for r in compression])
    if np.any(np.diff(commands) < -1e-12):
        raise ValueError("expected the one monotonically commanded compression stroke")
    # Native velocity does not reliably agree with pose displacement in the
    # present GPU recorder. Do not select force samples using its sign. All
    # samples in the declared monotone-command loading stroke are eligible,
    # including short solver oscillations and samples with absent contacts.
    selected = [r for r in compression if .01435 <= r["actual_depth_m"] <= .014605]
    # Fixed 25 µm depth bands avoid overweighting the slow ends of a quintic
    # drive and compare real depth, not a commanded displacement.
    for lo in np.arange(.01435, .014605, .000025):
        subset = [r for r in selected if lo <= r["actual_depth_m"] < lo+.000025]
        if len(subset) < 3:
            continue
        depth = float(np.mean([r["actual_depth_m"] for r in subset]))
        normal = float(np.mean([r["normal_force_magnitude_sum_n"] for r in subset]))
        axial = float(np.mean([r["total_axial_resistance_n"] for r in subset]))
        target = float(np.interp(depth, ref_depth, ref_normal))
        bins.append({"depth_m": depth, "sample_count": len(subset), "normal_sum_n": normal,
                     "axial_n": axial, "reference_normal_sum_n": target,
                     "reference_axial_n": float(np.interp(depth, ref_depth, ref_axial)),
                     "normal_point_count_mean": float(np.mean([r["normal_point_count"] for r in subset])),
                     "penetration_sum_mean_m": float(np.mean([r["penetration_sum_m"] for r in subset]))})
    if len(bins) < 4 or bins[-1]["depth_m"]-bins[0]["depth_m"] < .00008:
        raise ValueError("insufficient actual compression coverage to identify a scale")
    x = np.array([r["normal_sum_n"] for r in bins])
    y = np.array([r["reference_normal_sum_n"] for r in bins])
    if x.max() < 1e-3:
        raise ValueError("native compliant response is absent; do not inflate stiffness from zero response")
    scale = float(x@y/(x@x))
    old_k = float(result["configuration"]["seal_stiffness"])
    residual = float(np.linalg.norm(scale*x-y)/np.linalg.norm(y))
    # D=0 lets the held force/separation sum expose the original point law.
    held = [r for r in rows if r["phase"] == "compressed_hold"]
    held = held[len(held)//2:]
    ratio = [r["normal_force_magnitude_sum_n"]/r["penetration_sum_m"] for r in held if r["penetration_sum_m"] > 1e-9]
    output = {"scope": "ONE_SEAL_COUPON_POINT_STIFFNESS_MAPPING_NOT_TE_MATERIAL_IDENTIFICATION",
              "coupon": str(coupon.resolve()), "fixed_reference": str(reference.resolve()),
              "native_point_stiffness_before_n_m": old_k,
              "sample_selection": "ALL_MONOTONE_COMMAND_COMPRESSION_SAMPLES_BINNED_BY_ACTUAL_POSE_DEPTH; NO_NATIVE_VELOCITY_FILTER",
              "eligible_compression_samples": len(selected),
              "normal_load_scale_least_squares": scale,
              "recommended_native_point_stiffness_n_m": old_k*scale,
              "normal_curve_relative_rms_residual_after_scale": residual,
              "native_held_force_over_penetration_sum_n_m_median": float(np.median(ratio)) if ratio else None,
              "shape_agreement_note": "A single scale is insufficient if residual is large; inspect actual contact motion before modifying constitutive assumptions.",
              "does_not_edit_or_recommend_changing_material_modulus_or_friction": True,
              "does_not_use_robot_success_as_fit_target": True,
              "fitted_native_parameter_requires_one_dynamic_recheck": True,
              "source_model_approximation_limits_retained": curve["limitations"],
              "bins": bins}
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coupon", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1]/"references/seal_reference_curve.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = analyze(args.coupon, args.reference)
    args.output.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps({k: v for k, v in data.items() if k != "bins"}, indent=2))
