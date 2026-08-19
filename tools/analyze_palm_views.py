#!/usr/bin/env python3
"""Quick numeric analysis of phase6 PALM/WRIST captures (RGB+depth)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1")
DIRS = [
    "phase6_t_hp_h0_capture_v2",
    "phase6_t_hp_h0_palm_capture_v1",
    "phase6_t_hp_h0_palm_c1_v1",
    "phase6_t_hp_h0_palm_c2_v1",
]


def main() -> int:
    for name in DIRS:
        view = "PALM_H0" if "palm" in name else "WRIST_H0"
        root = BASE / name / "seed000" / "formal_views" / view
        rgb = np.asarray(Image.open(root / "rgb.png").convert("RGB"))
        depth = np.load(root / "depth_m.npy")
        cam = json.loads((root / "camera.json").read_text())
        finite = np.isfinite(depth) & (depth > 0.0)
        print(f"=== {name} ===")
        print(f"  rgb {rgb.shape} mean={rgb.reshape(-1,3).mean(0).round(1)}")
        print(f"  depth finite={finite.mean():.3f} min={depth[finite].min():.3f} max={depth[finite].max():.3f}")
        for pct in (1, 10, 50, 90, 99):
            print(f"    depth p{pct}={np.percentile(depth[finite], pct):.4f}")
        eye = np.asarray(cam["eye_m"]); tgt = np.asarray(cam["target_m"])
        d = tgt - eye
        print(f"  eye={eye.round(3)} target={tgt.round(3)} dist={np.linalg.norm(d):.3f} dir={(d/np.linalg.norm(d)).round(3)}")
        flat = rgb.reshape(-1, 3)
        bins = (flat // 32) * 32
        uniq, counts = np.unique(bins, axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        for idx in order:
            print(f"    color~{uniq[idx]} count={counts[idx]} ({100*counts[idx]/len(flat):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
