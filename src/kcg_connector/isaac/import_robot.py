#!/usr/bin/env python3

"""Convert the exported authoritative KUKA/KCG URDF into a USD asset."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--usd-directory", required=True)
    arguments = parser.parse_args()

    urdf_path = Path(arguments.urdf).expanduser().resolve()
    output_directory = Path(arguments.usd_directory).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )

    try:
        from isaacsim.asset.importer.urdf import (
            URDFImporter,
            URDFImporterConfig,
        )

        configuration = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_directory),
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,
            allow_self_collision=False,
            fix_base=True,
        )
        output_path = URDFImporter(configuration).import_urdf()
        for _ in range(5):
            simulation_app.update()
        if not output_path:
            raise RuntimeError("URDF importer returned an empty output path")
        print(f"ISAAC ROBOT USD EXPORTED: {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
