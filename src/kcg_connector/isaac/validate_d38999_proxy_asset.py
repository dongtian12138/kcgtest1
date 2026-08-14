#!/usr/bin/env python3

"""Fail-closed topology check for the public-dimensional D38999 proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback


ROOT = "/World/D38999Shell25JProxy"
FIXED = ROOT + "/FixedReceptacle"
LOOSE = ROOT + "/LoosePlug"
BODY = LOOSE + "/BodyAssembly"
NUT = LOOSE + "/CouplingNut"
JOINT = LOOSE + "/CouplingNutJoint"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    arguments = parser.parse_args()
    asset = Path(arguments.asset).expanduser().resolve()
    if not asset.is_file() or asset.stat().st_size <= 0:
        parser.error(f"asset is missing or empty: {asset}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    passed = False
    metrics = {"asset": str(asset), "passed": False}
    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(asset))
        if stage is None:
            raise RuntimeError("USD stage could not be opened")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise RuntimeError("asset stage must use metres")
        root = stage.GetPrimAtPath(ROOT)
        fixed = stage.GetPrimAtPath(FIXED)
        body = stage.GetPrimAtPath(BODY)
        nut = stage.GetPrimAtPath(NUT)
        joint_prim = stage.GetPrimAtPath(JOINT)
        if not all(prim.IsValid() for prim in (root, fixed, body, nut)):
            raise RuntimeError("required proxy prim is missing")
        metadata = root.GetCustomDataByKey("kcg")
        expected_metadata = {
            "certificationClaim": "none",
            "fidelity": "public_dimensional_visual_physics_proxy",
            "fixedPartNumber": "D38999/20KJ61PN",
            "loosePartNumber": "D38999/26KJ61SN",
            "proxyId": "d38999_shell25j_61_pair_proxy_v1",
            "threadCollisionMode": "none",
        }
        if metadata != expected_metadata:
            raise RuntimeError(f"unexpected proxy metadata: {metadata}")
        if fixed.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("fixed receptacle must remain static")
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("loose plug body is not a rigid body")
        if not nut.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("coupling nut is not a rigid body")
        if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError("coupling-nut revolute joint is missing")
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        if body0 != [BODY] or body1 != [NUT]:
            raise RuntimeError("coupling-nut joint bodies are incorrect")
        if joint.GetAxisAttr().Get() != "Z":
            raise RuntimeError("coupling nut must rotate about local Z")
        if bool(joint.GetCollisionEnabledAttr().Get()):
            raise RuntimeError("coupling joint collision must be disabled")

        collisions = tuple(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        nut_segments = tuple(
            path for path in collisions if path.startswith(NUT + "/Segment_")
        )
        fixed_entry_segments = tuple(
            path
            for path in collisions
            if path.startswith(FIXED + "/EntryShell/Segment_")
        )
        socket_visuals = tuple(
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(BODY + "/Sockets/Socket_")
        )
        pin_visuals = tuple(
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(FIXED + "/Pins/Pin_")
        )
        if len(nut_segments) != 24 or len(fixed_entry_segments) != 20:
            raise RuntimeError("proxy collision rings are incomplete")
        if len(socket_visuals) != 61 or len(pin_visuals) != 61:
            raise RuntimeError("visual-only 61-contact layout is incomplete")
        if any(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in socket_visuals + pin_visuals
        ):
            raise RuntimeError("contact-pin visuals must not collide")

        passed = True
        metrics.update(
            {
                "body0": body0,
                "body1": body1,
                "collision_count": len(collisions),
                "fixed_entry_collision_segments": len(
                    fixed_entry_segments
                ),
                "metadata": metadata,
                "nut_collision_segments": len(nut_segments),
                "passed": True,
                "pin_visual_count": len(pin_visuals),
                "sha256": _sha256(asset),
                "socket_visual_count": len(socket_visuals),
            }
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print("ISAAC D38999 PROXY ASSET PASSED", flush=True)
    except BaseException as exception:
        metrics["error"] = f"{type(exception).__name__}: {exception}"
        traceback.print_exc()
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print("ISAAC D38999 PROXY ASSET FAILED", flush=True)
    finally:
        app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
