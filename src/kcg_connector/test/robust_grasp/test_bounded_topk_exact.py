from __future__ import annotations

import hashlib
import json

import pytest

from kcg_connector.grasp.robust.bounded_topk_exact import (
    BoundedTopKExactError,
    ExactV9CompletedRecord,
    MAXIMUM_EXACT_CANDIDATE_COUNT,
    METHOD_ID,
    _completed_codec,
    _timeout_summary,
    load_selected_proxy_candidates,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    AttemptStatus,
    CanonicalV9Parameters,
)


def _profile() -> dict[str, object]:
    rows = []
    keys = []
    for index in range(MAXIMUM_EXACT_CANDIDATE_COUNT):
        parameters = (index / 8.0, 0.5, 0.5, 0.5, 0.5)
        import numpy as np

        key = np.asarray(parameters, dtype=">f8").tobytes().hex()
        keys.append(key)
        rows.append(
            {
                "proxy_rank": index + 1,
                "first_attempt_index": (155, 174, 14, 198)[index],
                "canonical_key_hex": key,
                "parameters_unit": list(parameters),
                "selected_for_exact_v9": True,
            }
        )
    return {
        "schema_version": "carts_multifidelity_proxy_rank_profile_v8",
        "status": "COMPLETED_TOP4_PROXY_BOUND",
        "object_id": "current_d38999_26kj61sn_public_spec",
        "exact_v9_evaluator_called": False,
        "generation_checkpoint_written": False,
        "isaac_launched": False,
        "exact_top_k_ceiling": 4,
        "exact_selected_unique_count": 4,
        "proxy_rank_result": {
            "proxy_certifies_or_rejects": False,
            "formal_selected_candidate": None,
            "formal_selected_contact_range_policy": None,
            "full_hand_collision_state": "NOT_CERTIFIABLE",
            "dynamic_launch_allowed": False,
            "hardware_authorized": False,
            "ranked_survivors": rows,
            "exact_selected_keys": keys,
        },
    }


def _write_profile(tmp_path, document):
    value = (json.dumps(document, sort_keys=True) + "\n").encode()
    path = tmp_path / "profile.json"
    path.write_bytes(value)
    return path, hashlib.sha256(value).hexdigest()


def test_load_selected_proxy_candidates_preserves_frozen_top4(tmp_path) -> None:
    path, digest = _write_profile(tmp_path, _profile())

    _document, rows = load_selected_proxy_candidates(
        path, expected_sha256=digest
    )

    assert [row.proxy_rank for row in rows] == [1, 2, 3, 4]
    assert [row.first_attempt_index for row in rows] == [155, 174, 14, 198]


def test_load_selected_proxy_candidates_rejects_hash_or_claim_drift(
    tmp_path,
) -> None:
    document = _profile()
    path, digest = _write_profile(tmp_path, document)
    with pytest.raises(BoundedTopKExactError, match="SHA-256"):
        load_selected_proxy_candidates(path, expected_sha256="0" * 64)

    document["isaac_launched"] = True
    path, digest = _write_profile(tmp_path, document)
    with pytest.raises(BoundedTopKExactError, match="boundary"):
        load_selected_proxy_candidates(path, expected_sha256=digest)


def test_timeout_summary_is_unresolved_and_never_geometric_rejection(
    tmp_path,
) -> None:
    path, digest = _write_profile(tmp_path, _profile())
    _document, rows = load_selected_proxy_candidates(
        path, expected_sha256=digest
    )

    summary = _timeout_summary(rows[0], wall_seconds=60.01)

    assert summary["execution_status"] == (
        "COMPUTATION_UNRESOLVED_WALL_TIMEOUT"
    )
    assert summary["exact_attempt_status"] is None
    assert summary["v9_failure_reason"] is None
    assert summary["accepted_static"] is False
    assert summary["timeout_is_geometric_rejection"] is False


def test_completed_exact_record_round_trips_without_pickle() -> None:
    parameters = (0.25, 0.5, 0.5, 0.5, 0.5)
    key = bytes.fromhex(_profile()["proxy_rank_result"]["exact_selected_keys"][2])
    record = ExactV9CompletedRecord(
        method_id=METHOD_ID,
        profile_sha256="1" * 64,
        run_id="run",
        generator_contract_sha256="2" * 64,
        v9_model_contract_sha256="3" * 64,
        proxy_rank=3,
        first_attempt_index=14,
        canonical_parameters=CanonicalV9Parameters(
            values=parameters,
            exact_key=key,
        ),
        wall_seconds=1.25,
        status=AttemptStatus.V9_REJECTED,
        candidate=None,
        sequential_closure_policy=None,
        audit=None,
        invocation_binding=None,
        v9_failure_reason="TEST_REJECTION",
    )

    codec = _completed_codec()
    encoded = codec.canonical_bytes(record)
    decoded = codec.decode_canonical_bytes(encoded)

    assert decoded == record
