from pathlib import Path

import pytest
import yaml

from kcg_connector.geometry import helical_travel
from kcg_connector.task_logic import ConnectorTaskConfig, load_connector_task_config


def test_connector_yaml_geometry_and_helical_relation():
    config_path = Path(__file__).parents[1] / "config" / "connector_task.yaml"
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    geometry = document["geometry"]
    success = document["success"]
    assert geometry["plug_nose_radius"] < geometry["receptacle_entry_radius"]
    assert geometry["receptacle_entry_radius"] < geometry["receptacle_body_radius"]
    expected = helical_travel(
        2.0 * 3.141592653589793,
        success["helical_lead_per_revolution"],
    )
    assert expected == pytest.approx(success["helical_lead_per_revolution"])


def test_yaml_success_thresholds_match_the_versioned_defaults():
    config_path = Path(__file__).parents[1] / "config" / "connector_task.yaml"
    assert load_connector_task_config(config_path) == ConnectorTaskConfig()
