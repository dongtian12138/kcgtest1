#!/usr/bin/env python3

"""Validate the authored connector USD topology before running physics."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    arguments = parser.parse_args()

    asset_path = Path(arguments.asset).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(asset_path))
        if stage is None:
            raise RuntimeError(f"could not open connector USD: {asset_path}")

        body_path = "/World/Plug/BodyAssembly"
        nut_path = "/World/Plug/CouplingNut"
        joint_path = "/World/Plug/CouplingNutJoint"
        plug = stage.GetPrimAtPath("/World/Plug")
        body = stage.GetPrimAtPath(body_path)
        nut = stage.GetPrimAtPath(nut_path)
        joint_prim = stage.GetPrimAtPath(joint_path)
        if plug.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(
                "connector must remain ordinary rigid bodies for thread coupling"
            )
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("plug body is not an independent rigid body")
        if not nut or not nut.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("coupling nut is not an independent rigid body")
        if not joint_prim or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError("coupling nut revolute joint is missing")

        joint = UsdPhysics.RevoluteJoint(joint_prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        if body0 != [body_path] or body1 != [nut_path]:
            raise RuntimeError(
                f"unexpected coupling joint bodies: body0={body0}, body1={body1}"
            )
        if joint.GetAxisAttr().Get() != "Z":
            raise RuntimeError("coupling nut joint must rotate around the Z axis")

        collisions = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        nut_collisions = [
            path for path in collisions if path.startswith(f"{nut_path}/Segment_")
        ]
        if len(nut_collisions) < 8:
            raise RuntimeError("coupling nut collision ring is incomplete")

        metrics = {
            "asset": str(asset_path),
            "body0": body0,
            "body1": body1,
            "collision_count": len(collisions),
            "coupling_axis": joint.GetAxisAttr().Get(),
            "nut_collision_segments": len(nut_collisions),
        }
        print(json.dumps(metrics, sort_keys=True))
        print("ISAAC CONNECTOR ASSET TOPOLOGY PASSED")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
