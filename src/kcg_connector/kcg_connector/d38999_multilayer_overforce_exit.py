"""Latched, truth-free overforce exit decision for D38999 insertion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_overforce_exit_v1"
EXPECTED_FRAME_ID = "connector_task_frame"
FORMAL_FORCE_COMPONENT_LIMIT_N = 8.0
FORMAL_MOMENT_COMPONENT_LIMIT_NM = 0.30
FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml": (
        "0f16e9b2fc5d615a4e8035dfa21c4ec9a18a341b4320ecbc3e66b874be489703"
    ),
    "src/kcg_connector/kcg_connector/wrist_moment_safety_guard.py": (
        "779f7601a69f31c87ba44ad88584f540c2178c13b8c2bc09f5bde69385df0db8"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/"
    "TASK_RESULT.json": (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D7-COMPLIANT-INSERTION/"
    "TASK_RESULT.json": (
        "da156677861d69dfa2db8a292b4da6b9040cfe7e81a98232ab835970d6e38327"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrench6(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 6 or not all(math.isfinite(item) for item in result):
        return None
    return result


def load_overforce_exit_contract(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D8 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D8 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})
    master = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
        .read_text(encoding="utf-8")
    )
    compliant = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml")
        .read_text(encoding="utf-8")
    )
    b5 = json.loads(
        (root / "artifacts/agent_control/tasks/"
         "EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/TASK_RESULT.json")
        .read_text(encoding="utf-8")
    )
    d7 = json.loads(
        (root / "artifacts/agent_control/tasks/"
         "EIGHT-HOUR-D7-COMPLIANT-INSERTION/TASK_RESULT.json")
        .read_text(encoding="utf-8")
    )
    limits = master["acceptance_limits"]
    safety = compliant["safety"]
    recovery = compliant["recovery"]
    if (
        limits["force_component_limit_n_per_driven_body"]
        != FORMAL_FORCE_COMPONENT_LIMIT_N
        or limits["torque_component_limit_nm"]
        != FORMAL_MOMENT_COMPONENT_LIMIT_NM
        or b5["moment_limit_nm"] != FORMAL_MOMENT_COMPONENT_LIMIT_NM
        or d7["outcome"] != "OFFLINE_PASS"
        or d7["dynamic_compliant_insertion_pass_claimed"] is not False
        or safety["hard_axial_force_n"] != 5.0
        or safety["hard_lateral_force_n"] != 2.0
        or safety["hard_bending_moment_nm"] != 0.18
        or safety["hard_torsional_moment_nm"] != 0.05
        or safety["maximum_sample_age_s"] != 0.020
        or recovery["backoff_distance_m"] != 0.00040
        or recovery["backoff_speed_m_s"] != 0.00030
    ):
        raise ValueError("authoritative D8 overforce exit contract changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_frame_id": EXPECTED_FRAME_ID,
        "formal_force_component_limit_n": FORMAL_FORCE_COMPONENT_LIMIT_N,
        "formal_moment_component_limit_nm": FORMAL_MOMENT_COMPONENT_LIMIT_NM,
        "experimental_abort_envelope": {
            "axial_force_n": safety["hard_axial_force_n"],
            "lateral_force_n": safety["hard_lateral_force_n"],
            "bending_moment_nm": safety["hard_bending_moment_nm"],
            "torsional_moment_nm": safety["hard_torsional_moment_nm"],
        },
        "maximum_sample_age_s": safety["maximum_sample_age_s"],
        "backoff_distance_m": recovery["backoff_distance_m"],
        "backoff_speed_m_s": recovery["backoff_speed_m_s"],
        "comparison": "strictly_greater_triggers_exact_equality_passes",
        "sources": sources,
    }


class OverforceExitLatch:
    """Latch the first invalid or unsafe wrist-wrench observation."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        required = {
            "expected_frame_id",
            "formal_force_component_limit_n",
            "formal_moment_component_limit_nm",
            "experimental_abort_envelope",
            "maximum_sample_age_s",
            "backoff_distance_m",
            "backoff_speed_m_s",
        }
        if not isinstance(contract, Mapping) or not required.issubset(contract):
            raise ValueError("D8 contract is incomplete")
        if contract["expected_frame_id"] != EXPECTED_FRAME_ID:
            raise ValueError("D8 task frame changed")
        if contract["formal_force_component_limit_n"] != 8.0:
            raise ValueError("D8 formal force limit changed")
        if contract["formal_moment_component_limit_nm"] != 0.30:
            raise ValueError("D8 formal moment limit changed")
        self.contract = dict(contract)
        self.latched = False
        self.failure_reason: str | None = None
        self.action = "CONTINUE_DIAGNOSTIC_ONLY"
        self.last_timestamp_s: float | None = None
        self.observation_count = 0
        self.reset_count = 0
        self.last_metrics: dict[str, float] | None = None

    @classmethod
    def from_frozen_contracts(
        cls, repository_root: str | Path
    ) -> "OverforceExitLatch":
        return cls(load_overforce_exit_contract(repository_root))

    def _decision(self, *, sample_consumed: bool) -> dict[str, Any]:
        retract = self.latched and self.action == "SAFE_STOP_RETRACT_REOBSERVE"
        return {
            "schema_version": SCHEMA_VERSION,
            "safe_to_continue": not self.latched,
            "fault_latched": self.latched,
            "failure_reason": self.failure_reason,
            "action": self.action,
            "sample_consumed": sample_consumed,
            "observation_count": self.observation_count,
            "last_timestamp_s": self.last_timestamp_s,
            "last_metrics": dict(self.last_metrics) if self.last_metrics else None,
            "insertion_twist_candidate_task": [0.0] * 6,
            "retract_requested": retract,
            "requested_retract_distance_m": (
                self.contract["backoff_distance_m"] if retract else 0.0
            ),
            "requested_retract_speed_m_s": (
                self.contract["backoff_speed_m_s"] if retract else 0.0
            ),
            "motion_command_emitted": False,
            "control_authorized": False,
            "dynamic_overforce_exit_pass_claimed": False,
        }

    def _latch(
        self, reason: str, action: str, *, sample_consumed: bool
    ) -> dict[str, Any]:
        self.latched = True
        if self.failure_reason is None:
            self.failure_reason = reason
            self.action = action
        return self._decision(sample_consumed=sample_consumed)

    def observe(
        self,
        compensated_wrench_task: Sequence[float],
        *,
        timestamp_s: float,
        sample_age_s: float,
        frame_id: str,
        ft_valid: bool,
        ft_tared: bool,
        payload_compensated: bool,
    ) -> dict[str, Any]:
        if self.latched:
            return self._decision(sample_consumed=False)
        if frame_id != EXPECTED_FRAME_ID:
            return self._latch(
                "UNEXPECTED_WRENCH_FRAME", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        if (
            isinstance(timestamp_s, bool)
            or not isinstance(timestamp_s, (int, float))
            or not math.isfinite(float(timestamp_s))
        ):
            return self._latch(
                "INVALID_TIMESTAMP", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        timestamp = float(timestamp_s)
        if self.last_timestamp_s is not None and timestamp <= self.last_timestamp_s:
            return self._latch(
                "NONMONOTONIC_TIMESTAMP", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        if (
            isinstance(sample_age_s, bool)
            or not isinstance(sample_age_s, (int, float))
            or not math.isfinite(float(sample_age_s))
            or float(sample_age_s) < 0.0
            or float(sample_age_s) > self.contract["maximum_sample_age_s"]
        ):
            return self._latch(
                "STALE_OR_INVALID_WRENCH", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        if not all(type(value) is bool and value for value in (
            ft_valid, ft_tared, payload_compensated
        )):
            return self._latch(
                "FT_TARE_OR_PAYLOAD_INVALID", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        wrench = _wrench6(compensated_wrench_task)
        if wrench is None:
            return self._latch(
                "NONFINITE_OR_INVALID_WRENCH", "SAFE_STOP_REOBSERVE",
                sample_consumed=False,
            )
        self.last_timestamp_s = timestamp
        self.observation_count += 1
        axial = abs(wrench[2])
        lateral = math.hypot(wrench[0], wrench[1])
        bending = math.hypot(wrench[3], wrench[4])
        torsional = abs(wrench[5])
        self.last_metrics = {
            "axial_force_n": axial,
            "lateral_force_n": lateral,
            "bending_moment_nm": bending,
            "torsional_moment_nm": torsional,
            "maximum_force_component_n": max(abs(value) for value in wrench[:3]),
            "maximum_moment_component_nm": max(abs(value) for value in wrench[3:]),
        }
        if self.last_metrics["maximum_force_component_n"] > 8.0:
            return self._latch(
                "FORMAL_FORCE_COMPONENT_LIMIT",
                "SAFE_STOP_RETRACT_REOBSERVE",
                sample_consumed=True,
            )
        if self.last_metrics["maximum_moment_component_nm"] > 0.30:
            return self._latch(
                "FORMAL_MOMENT_COMPONENT_LIMIT",
                "SAFE_STOP_RETRACT_REOBSERVE",
                sample_consumed=True,
            )
        envelope = self.contract["experimental_abort_envelope"]
        for value, key, reason in (
            (axial, "axial_force_n", "EXPERIMENTAL_AXIAL_FORCE_ABORT"),
            (lateral, "lateral_force_n", "EXPERIMENTAL_LATERAL_FORCE_ABORT"),
            (bending, "bending_moment_nm", "EXPERIMENTAL_BENDING_MOMENT_ABORT"),
            (torsional, "torsional_moment_nm", "EXPERIMENTAL_TORSIONAL_MOMENT_ABORT"),
        ):
            if value > envelope[key]:
                return self._latch(
                    reason, "SAFE_STOP_RETRACT_REOBSERVE", sample_consumed=True
                )
        return self._decision(sample_consumed=True)

    def reset_latch(self, *, explicit_authorization: bool, reason: str) -> None:
        if explicit_authorization is not True:
            raise ValueError("explicit reset authorization is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset reason is required")
        self.latched = False
        self.failure_reason = None
        self.action = "CONTINUE_DIAGNOSTIC_ONLY"
        self.last_timestamp_s = None
        self.last_metrics = None
        self.reset_count += 1

    def report(self) -> dict[str, Any]:
        return {
            **self._decision(sample_consumed=False),
            "status": "OFFLINE_OVERFORCE_EXIT_LATCH",
            "reset_count": self.reset_count,
            "contract": self.contract,
            "truth_firewall": {
                "object_pose_input": False,
                "contact_identity_input": False,
                "contact_normal_input": False,
                "event_truth_input": False,
                "post_run_pose_write": False,
            },
            "hardware_authorized": False,
        }


__all__ = [
    "EXPECTED_FRAME_ID",
    "FORMAL_FORCE_COMPONENT_LIMIT_N",
    "FORMAL_MOMENT_COMPONENT_LIMIT_NM",
    "FROZEN_SOURCES",
    "OverforceExitLatch",
    "SCHEMA_VERSION",
    "load_overforce_exit_contract",
]
