#!/usr/bin/env python3

'''Pure offline E0 diagnostics for one tabletop physical grasp episode.

Reads nominal_physics_report.json and controller_steps.jsonl, recomputes the
frozen post-hoc decomposition candidates, and writes a NEW JSON file.  This
tool never mutates the original report/steps and never changes the original
PASS result.  No Isaac dependency: it is plain Python plus the pure
kcg_connector.grasp.posthoc_wrench_analysis module.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from kcg_connector.grasp.posthoc_wrench_analysis import analyze_episode


def _config_document(repository: Path, config_path: Path) -> dict:
    raw = Path(config_path)
    if not raw.is_absolute():
        raw = repository / raw
    path = raw.resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"config {path} is not a mapping")
    return document


def _relative_document(
    repository: Path, base_path: Path, relative: str
) -> dict:
    # Repository-relative configs resolve against the repository root, mirroring
    # the strict loaders.  References that are already anchored resolve as-is.
    base = Path(relative)
    if base.is_absolute():
        return _config_document(repository, base)
    return _config_document(repository, Path(base_path).parent / base)


def load_plug_nut_mass_kg(
    repository: Path, physical_grasp_config: Path
) -> float:
    physical_path = Path(physical_grasp_config)
    if not physical_path.is_absolute():
        physical_path = repository / physical_path
    physical = _config_document(repository, physical_path)
    # base.pick_config is repository-relative (mirrors the strict loader).
    pick_config_path = (repository / physical["base"]["pick_config"]).resolve()
    pick_document = _config_document(repository, pick_config_path)
    # scene.proxy_config resolves against the pick config directory.
    proxy_path = pick_config_path.parent / pick_document["scene"]["proxy_config"]
    proxy = _config_document(repository, proxy_path)
    assumptions = proxy["physics_assumptions"]
    body_mass = float(assumptions["plug_body_mass_kg"])
    nut_mass = float(assumptions["coupling_nut_mass_kg"])
    total = body_mass + nut_mass
    if total <= 0.0:
        raise SystemExit("configured plug+nut mass is not positive")
    return total


def load_gravity_m_s2(
    repository: Path, physical_grasp_config: Path
) -> float:
    physical_path = Path(physical_grasp_config)
    if not physical_path.is_absolute():
        physical_path = repository / physical_path
    physical = _config_document(repository, physical_path)
    pick_config_path = (repository / physical["base"]["pick_config"]).resolve()
    pick_document = _config_document(repository, pick_config_path)
    tabletop_path = (
        pick_config_path.parent
        / pick_document["scene"]["tabletop_config"]
    )
    tabletop = _config_document(repository, tabletop_path)
    gravity = abs(float(tabletop["physics"]["gravity_m_s2"]))
    if gravity <= 0.0:
        raise SystemExit("configured gravity magnitude is not positive")
    return gravity


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="E0 offline posthoc wrench diagnostics"
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        help=(
            "directory containing nominal_physics_report.json and "
            "controller_steps.jsonl"
        ),
    )
    parser.add_argument(
        "--physical-grasp-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output JSON path; defaults to "
            "<episode-dir>/posthoc_e0_analysis.json"
        ),
    )
    arguments = parser.parse_args()
    episode_dir = Path(arguments.episode_dir).expanduser().resolve()
    report_path = episode_dir / "nominal_physics_report.json"
    steps_path = episode_dir / "controller_steps.jsonl"
    if not report_path.is_file() or not steps_path.is_file():
        raise SystemExit(
            "episode dir must contain nominal_physics_report.json and "
            "controller_steps.jsonl"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config_path = Path(arguments.physical_grasp_config).expanduser().resolve()
    plug_nut_mass_kg = load_plug_nut_mass_kg(repository, config_path)
    gravity_m_s2 = load_gravity_m_s2(repository, config_path)
    result = analyze_episode(
        report,
        steps,
        plug_nut_mass_kg=plug_nut_mass_kg,
        gravity_m_s2=gravity_m_s2,
    )
    output_path = (
        Path(arguments.output).expanduser().resolve()
        if arguments.output
        else episode_dir / "posthoc_e0_analysis.json"
    )
    if output_path.resolve() in (
        report_path.resolve(),
        steps_path.resolve(),
    ):
        raise SystemExit(
            "refusing to overwrite the original report or controller steps"
        )
    output_path.write_text(
        json.dumps(
            result, allow_nan=False, ensure_ascii=False, indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"E0 posthoc diagnostics written to {output_path}")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "reference_absolute_moment_norm_nm",
                    "current_absolute_moment_norm_nm",
                    "vector_delta_moment_norm_nm",
                    "moment_magnitude_increase_candidate_nm",
                    "delta_fz_n",
                    "delta_fz_to_plug_nut_weight_ratio",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
