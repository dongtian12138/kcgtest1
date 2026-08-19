#!/usr/bin/env python3

"""Convert the exported authoritative KUKA/KCG URDF into a USD asset."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--usd-directory", required=True)
    parser.add_argument(
        "--allow-self-collision",
        action="store_true",
        help="Author self-collision for the successor physical-r7 asset.",
    )
    arguments = parser.parse_args()

    urdf_path = Path(arguments.urdf).expanduser().resolve()
    output_directory = Path(arguments.usd_directory).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if not arguments.allow_self_collision:
        raise ValueError(
            "the physical-r7 successor import requires --allow-self-collision"
        )
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite successor USD directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=False)

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
            allow_self_collision=arguments.allow_self_collision,
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
