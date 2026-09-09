"""Native mass and pose accounting for the additional physical band components.

Only the postrun recorder may consume these values. They are not observations
for the assembly controller.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def rotations_wxyz(quaternions):
    return Rotation.from_quat(np.asarray(quaternions)[:, [1, 2, 3, 0]]).as_matrix()


def aggregate_properties(masses, centers, inertias):
    masses, centers, inertias = map(np.asarray, (masses, centers, inertias))
    center = np.average(centers, axis=0, weights=masses)
    delta = centers-center
    shifted = masses[:, None, None] * (
        np.sum(delta*delta, axis=1)[:, None, None]*np.eye(3)
        - np.einsum("ni,nj->nij", delta, delta))
    return float(masses.sum()), center, np.sum(inertias+shifted, axis=0)


class RadialBandAudit:
    def __init__(self, contact_prim, sensor_paths, report):
        from isaacsim.core.experimental.utils.backend import use_backend

        self.contact_prim = contact_prim
        self.paths = [report["core_path"]] + [r["body_path"] for r in report["physical_components"]]
        self.indices = [list(sensor_paths).index(path) for path in self.paths]
        if len(set(self.indices)) != len(self.paths):
            raise ValueError("each physical band component needs a unique native sensor")
        with use_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
            self.masses = contact_prim.get_masses(indices=self.indices).numpy().reshape(-1).astype(float)
            coms, axes = contact_prim.get_coms(indices=self.indices)
            self.local_coms = coms.numpy().astype(float)
            mass_rotations = rotations_wxyz(axes.numpy())
            inertia = contact_prim.get_inertias(indices=self.indices).numpy().reshape(-1, 3, 3).astype(float)
        self.body_frame_inertias = mass_rotations @ inertia @ np.transpose(mass_rotations, (0, 2, 1))
        if not np.isfinite(self.masses).all() or np.any(self.masses <= 0):
            raise ValueError("all native band masses must be finite and positive")
        positions, rotations, centers = self.read_components()
        reference_rotation = rotations[0]
        centers_body = (centers-positions[0]) @ reference_rotation
        relative_rotations = reference_rotation.T @ rotations
        inertias_body = relative_rotations @ self.body_frame_inertias @ np.transpose(relative_rotations, (0, 2, 1))
        mass, center, inertia = aggregate_properties(self.masses, centers_body, inertias_body)
        expected_mass = float(report["rest_aggregate_body_mass_kg"])
        expected_center = np.asarray(report["rest_aggregate_body_com_m"])
        expected_inertia = np.asarray(report["rest_aggregate_body_inertia_kg_m2"])
        errors = {"mass_kg": abs(mass-expected_mass),
                  "center_m": float(np.linalg.norm(center-expected_center)),
                  "maximum_inertia_entry_kg_m2": float(np.max(abs(inertia-expected_inertia)))}
        self.initial_mass_audit = {
            "source": "NATIVE_TENSOR_MASS_COM_PRINCIPAL_INERTIA_AND_RIGID_POSES",
            "physical_component_paths": self.paths, "physical_masses_kg": self.masses.tolist(),
            "physical_local_coms_m": self.local_coms.tolist(),
            "physical_inertias_in_link_axes_kg_m2": self.body_frame_inertias.tolist(),
            "aggregate_mass_kg": mass, "aggregate_com_body_m": center.tolist(),
            "aggregate_inertia_body_kg_m2": inertia.tolist(), "errors_against_original_body": errors,
            "used_for_online_control": False,
        }
        if errors["mass_kg"] > 1e-9 or errors["center_m"] > 1e-6 or errors["maximum_inertia_entry_kg_m2"] > 1e-11:
            raise ValueError(f"native partition does not preserve original aggregate Body properties: {errors}")
        self.additional_mass_kg = float(self.masses[1:].sum())

    def read_components(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
            positions, orientations = self.contact_prim.get_world_poses(indices=self.indices)
        positions = positions.numpy().astype(float)
        rotations = rotations_wxyz(orientations.numpy())
        centers = positions + np.einsum("nij,nj->ni", rotations, self.local_coms)
        if not np.isfinite(centers).all():
            raise RuntimeError("nonfinite physical band component pose")
        return positions, rotations, centers

    def capture(self):
        positions, rotations, centers = self.read_components()
        additional_center = np.average(centers[1:], axis=0, weights=self.masses[1:])
        return {
            "scope": "NATIVE_EXTRA_BODY_COMPONENTS_POSTRUN_ONLY",
            "additional_mass_kg": self.additional_mass_kg,
            "additional_center_world_m": additional_center.tolist(),
            "aggregate_body_center_world_m": np.average(centers, axis=0, weights=self.masses).tolist(),
            "maximum_relative_sector_axis_error": float(np.max(abs(
                rotations-rotations[0]))),
            "used_for_online_control": False,
        }
