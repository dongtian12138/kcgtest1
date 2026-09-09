"""Offline scale estimate for the declared homogenized pin-root seal proxy.

This is not a TE material curve or an independent finite-element solution.
It fixes the desired order of elastic resistance BEFORE assembly is attempted,
using source boss geometry and a published fluorosilicone stress value.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def reference_curve(repository, depths_m, *, modulus100_mpa=1.9, friction=.45):
    repo = Path(repository)
    audit = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260907/seal_boss_geometric_compression_audit.json"
    source = json.loads(audit.read_text())
    rows = source["sampled_source_boss_profile"]
    z_data = np.array([r["body_z_m"] for r in rows])
    r_data = np.array([r["radius_m"] for r in rows])
    # Neo-Hooke nominal tensile stress P = G (lambda - lambda^-2).
    # At 100% extension lambda=2. The chosen 52 Shore A Dow blend reports
    # 1.9 MPa; neither the Shore value nor this blend is asserted for TE.
    shear_modulus_pa = modulus100_mpa * 1e6 / (2. - 2.**-2)
    boss_height_m = -.0132715 - (-.0145923)
    z = np.linspace(z_data.min(), z_data.max(), 4097)
    radius = np.interp(z, z_data, r_data)
    dr_dz = np.gradient(radius, z)
    area_density = 2*np.pi*radius*np.sqrt(1+dr_dz**2)
    bore_r, cone_end_z = .00046355, -.00120015
    result = []
    for depth in depths_m:
        socket_z = -float(depth)-z
        available = bore_r + np.maximum(socket_z-cone_end_z, 0.)
        inside = radius > available
        # Nearest point on each ray of the unchanged conical-entry/bore
        # meridian. Clamping the ray parameters avoids an artificial pressure
        # jump at the cone/bore junction.
        dr, dz = radius-bore_r, socket_z-cone_end_z
        cone_parameter = np.maximum((dr+dz)/2., 0.)
        cone_delta_r, cone_delta_z = dr-cone_parameter, dz-cone_parameter
        cone_distance = np.hypot(cone_delta_r, cone_delta_z)
        bore_delta_r, bore_delta_z = dr, np.maximum(dz, 0.)
        bore_distance = np.hypot(bore_delta_r, bore_delta_z)
        use_cone = cone_distance < bore_distance
        normal_depth = np.where(inside, np.minimum(cone_distance, bore_distance), 0.)
        strain_proxy = normal_depth / boss_height_m
        if strain_proxy.max() >= .5:
            raise ValueError("declared foundation-length approximation exceeded its analysis range")
        lam = 1. - strain_proxy
        pressure = shear_modulus_pa * (lam**-2-lam)
        normal_z = np.where(use_cone, cone_delta_z, bore_delta_z)
        nz = np.divide(np.abs(normal_z), normal_depth, out=np.zeros_like(normal_depth), where=normal_depth>0)
        nz = np.minimum(nz, 1.)
        tangent_z = np.sqrt(1.-nz**2)
        normal_axial = 128 * float(np.trapz(pressure*area_density*nz, z))
        friction_axial = 128 * float(np.trapz(pressure*area_density*tangent_z*friction, z))
        normal_sum = 128 * float(np.trapz(pressure*area_density, z))
        result.append({"depth_m": float(depth), "normal_force_sum_n": normal_sum,
                       "elastic_axial_n": normal_axial, "sliding_friction_axial_n": friction_axial,
                       "forward_total_axial_n": normal_axial+friction_axial,
                       "maximum_normal_overlap_m": float(normal_depth.max()),
                       "foundation_overlap_over_length_max": float(strain_proxy.max())})
    return {"scope": "OFFLINE_HOMOGENIZED_SEAL_SCALE_ESTIMATE_NOT_TE_CALIBRATION_OR_FEM",
            "reference_material": "Dow LS-2940 U / LS-2970 U 70/30 blend, 52 Shore A",
            "reference_nominal_tensile_stress_at_100_percent_pa": modulus100_mpa*1e6,
            "neo_hooke_shear_modulus_pa": shear_modulus_pa,
            "small_strain_young_modulus_pa": 3*shear_modulus_pa,
            "homogenization_length_m": boss_height_m,
            "length_definition": "SOURCE_BOSS_HEIGHT; REPRESENTATIVE_NORMAL_DEFORMATION_LENGTH_ALLOWING_EXTRUSION",
            "not_actual_TE_thickness_or_compressive_strain": True,
            "friction_reference": friction,
            "method": "Axisymmetric source boss lateral-envelope quadrature; local normal gap to unchanged socket cone/bore; Neo-Hooke compressive foundation pressure; Coulomb sliding direction",
            "limitations": ["not measured TE hardness/material/force curve", "source boss material assignment inferred",
                            "foundation permits effective lateral/axial extrusion without explicitly conserving volume",
                            "omits nonlocal seal coupling, end-face details and viscoelastic rate dependence",
                            "native contact patch count mapping must be measured", "not an electrical or environmental seal certification"],
            "geometry_audit": str(audit), "rows": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reference_curve(args.repository, [.0142, .014306884, .01435, .0144, .01445, .0145, .01455, .0146, .014605])
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
