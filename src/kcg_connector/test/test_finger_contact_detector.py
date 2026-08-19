import pytest

from kcg_connector.grasp.finger_contact_detector import (
    FingerContactDetector,
    FingerContactDetectorConfig,
    FingerContactState,
)


def _config() -> FingerContactDetectorConfig:
    return FingerContactDetectorConfig(
        sample_period_s=0.01,
        lowpass_alpha=0.9,
        derivative_alpha=0.9,
        contact_sigma_multiplier=6.0,
        minimum_contact_delta_nm=0.10,
        release_ratio=0.45,
        minimum_release_delta_nm=0.02,
        minimum_rise_rate_nm_s=0.01,
        maximum_stall_velocity_rad_s=0.02,
        minimum_tracking_error_rad=0.005,
        confirm_steps=3,
        release_confirm_steps=3,
        maximum_sample_gap_s=0.02,
        position_velocity_window_steps=6,
    )


def _update(detector, torque, step, *, position=0.0, velocity=0.0, command=0.02):
    return detector.update(
        torque,
        joint_position_rad=position,
        joint_velocity_rad_s=velocity,
        commanded_position_rad=command,
        timestamp_s=step * 0.01,
    )


def test_baseline_is_frozen_and_noise_does_not_trigger_contact():
    detector = FingerContactDetector(_config(), name="f1")
    detector.calibrate([0.001, -0.001] * 8)
    baseline = detector.baseline_mean_nm
    for step in range(1, 21):
        observation = _update(detector, 0.005, step, command=0.0)
        assert observation.state == FingerContactState.APPROACH
    assert detector.baseline_mean_nm == baseline


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_signed_root_torque_proxy_uses_magnitude_and_multistep_confirmation(sign):
    detector = FingerContactDetector(_config(), name="f1")
    detector.calibrate([0.0] * 16)
    states = [_update(detector, sign * 0.5, step).state for step in range(1, 4)]
    assert states[0] == FingerContactState.CONTACT_CANDIDATE
    assert states[1] == FingerContactState.CONTACT_CANDIDATE
    assert states[2] == FingerContactState.CONTACT_CONFIRMED


def test_soft_hold_release_hysteresis_is_debounced():
    detector = FingerContactDetector(_config(), name="f2")
    detector.calibrate([0.0] * 16)
    for step in range(1, 4):
        _update(detector, 0.5, step)
    detector.mark_soft_hold()
    assert detector.state == FingerContactState.SOFT_HOLD
    states = [_update(detector, 0.0, step).state for step in range(4, 9)]
    assert states[-1] == FingerContactState.SLIP_SUSPECTED
    assert FingerContactState.SOFT_HOLD in states


def test_stale_and_nonfinite_samples_fail_closed():
    detector = FingerContactDetector(_config(), name="f3")
    detector.calibrate([0.0] * 16)
    _update(detector, 0.0, 1)
    with pytest.raises(ValueError, match="stale"):
        _update(detector, 0.0, 4)
    assert detector.state == FingerContactState.FAILED

    detector.calibrate([0.0] * 16)
    with pytest.raises(ValueError, match="non-finite"):
        _update(detector, float("nan"), 1)
    assert detector.state == FingerContactState.FAILED


def _prod_config(window=6) -> FingerContactDetectorConfig:
    return FingerContactDetectorConfig(
        sample_period_s=1.0 / 240.0,
        lowpass_alpha=0.18,
        derivative_alpha=0.15,
        contact_sigma_multiplier=6.0,
        minimum_contact_delta_nm=0.020,
        release_ratio=0.45,
        minimum_release_delta_nm=0.008,
        minimum_rise_rate_nm_s=0.040,
        maximum_stall_velocity_rad_s=0.020,
        minimum_tracking_error_rad=0.004,
        confirm_steps=12,
        release_confirm_steps=18,
        maximum_sample_gap_s=0.0125,
        position_velocity_window_steps=window,
    )


def test_free_motion_with_false_low_reported_qd_never_contacts():
    # Production-like f2 GUI signature: q advances at target rate while the
    # reported joint velocity falsely reads ~0.013 rad/s; torque trends
    # smoothly across the threshold but the filtered rate stays below the
    # rising gate.  The position-derived velocity must veto the stall.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    q = 0.25
    for step in range(1, 601):
        q += 0.000735
        observation = detector.update(
            0.00005 * step,
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
        assert observation.state == FingerContactState.APPROACH
        assert observation.candidate_steps == 0


def test_stall_only_candidate_and_confirm_branch_isolation():
    # True stall-only proof.  Phase 1: q advances at the production target
    # rate with a falsely-low reported qd while a slow torque ramp crosses
    # the threshold; the filtered rate stays below the rising gate, so
    # load>=threshold alone must never candidate.  Phase 2: q freezes and
    # the reported qd flips to a false-high 0.05; once the old motion
    # samples leave the 7-sample window the FIRST candidate frame must be
    # a pure stall (no rising contribution), and exactly 12 continuous
    # candidate frames must follow before CONFIRMED.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    step = 0
    q = 0.25
    while True:
        step += 1
        q += 0.000735
        observation = detector.update(
            0.03 * min(1.0, step / 240.0),
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
        assert observation.state == FingerContactState.APPROACH
        assert observation.candidate_steps == 0
        if step >= 200 and observation.absolute_load_nm >= 0.02:
            break
    frozen_q = q
    freeze_step = step
    first_candidate_step = None
    confirm_step = None
    for extra in range(1, 40):
        step += 1
        observation = detector.update(
            0.03 * min(1.0, step / 240.0),
            joint_position_rad=frozen_q,
            joint_velocity_rad_s=0.05,
            commanded_position_rad=frozen_q
            + 0.005
            + 0.000735 * (step - freeze_step),
            timestamp_s=step / 240.0,
        )
        if first_candidate_step is None and observation.candidate_steps == 1:
            first_candidate_step = step
            assert observation.stalled is True
            assert (
                abs(observation.filtered_rate_nm_s)
                < 0.040
            )
            assert observation.position_derived_velocity_rad_s == 0.0
            assert observation.reported_joint_velocity_rad_s == 0.05
            assert observation.velocity_disagreement_rad_s == pytest.approx(
                0.05
            )
            assert observation.stall_velocity_source == "position_history"
        elif (
            first_candidate_step is not None
            and observation.state == FingerContactState.CONTACT_CANDIDATE
        ):
            assert observation.candidate_steps == (
                step - first_candidate_step + 1
            )
        if observation.state == FingerContactState.CONTACT_CONFIRMED:
            confirm_step = step
            break
    assert first_candidate_step is not None
    # The first candidate lands 6 frames after the freeze (the moved
    # sample leaves the window), then exactly 12 continuous frames.
    assert first_candidate_step == freeze_step + 6
    assert confirm_step == first_candidate_step + 11


def test_velocity_disagreement_is_evidence_only():
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    # Warmup frames: q frozen, reported qd 0.05.
    for step in range(1, 4):
        observation = detector.update(
            0.03,
            joint_position_rad=0.25,
            joint_velocity_rad_s=0.05,
            commanded_position_rad=0.25 + 0.005 + 0.000735 * step,
            timestamp_s=step / 240.0,
        )
        assert observation.position_derived_velocity_rad_s is None
        assert observation.velocity_disagreement_rad_s is None
        assert observation.position_velocity_window_span_s is None
        assert observation.position_velocity_sample_count == step
        assert observation.stall_velocity_source == "insufficient_history"
        assert observation.stalled is False
    # After the window fills with a frozen q: derived = 0, disagreement =
    # |0.05 - 0|, source position_history, stalled true (tracking grows).
    for step in range(4, 15):
        observation = detector.update(
            0.03,
            joint_position_rad=0.25,
            joint_velocity_rad_s=0.05,
            commanded_position_rad=0.25 + 0.005 + 0.000735 * step,
            timestamp_s=step / 240.0,
        )
        if step < 7:
            assert (
                observation.stall_velocity_source == "insufficient_history"
            )
            assert observation.position_derived_velocity_rad_s is None
        else:
            assert observation.stall_velocity_source == "position_history"
            assert observation.position_derived_velocity_rad_s == 0.0
            assert observation.velocity_disagreement_rad_s == pytest.approx(
                0.05
            )
            assert observation.position_velocity_sample_count == 7
            assert observation.position_velocity_window_span_s == (
                pytest.approx(6.0 / 240.0)
            )
    assert observation.stalled is True


def test_rising_contact_during_warmup_still_confirms():
    # Contact torque jumps on frame 2 while the history is still warming
    # up: the rising branch must carry the candidate and confirm.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    q = 0.25
    states = []
    for step in range(1, 16):
        q += 0.000735
        torque = 0.0 if step == 1 else 0.15
        observation = detector.update(
            torque,
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
        states.append(observation.state)
    assert FingerContactState.CONTACT_CANDIDATE in states[:3]
    assert FingerContactState.CONTACT_CONFIRMED in states


def test_candidate_interrupt_resets_continuity():
    # Stall-driven candidate (slow torque ramp keeps the filtered rate
    # below the rising gate).  Five consecutive candidate frames, then one
    # frame of genuine motion kills the stall while load stays high: the
    # count must reset to zero while the state keeps CANDIDATE, and a full
    # 12-frame continuous run is required before CONFIRMED.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    step = 0
    while True:
        step += 1
        torque = 0.03 * min(1.0, step / 240.0)
        observation = detector.update(
            torque,
            joint_position_rad=0.25,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=0.25 + 0.005 + 0.000735 * step,
            timestamp_s=step / 240.0,
        )
        assert observation.state in (
            FingerContactState.APPROACH,
            FingerContactState.CONTACT_CANDIDATE,
        )
        if observation.candidate_steps == 5:
            break
        assert step < 1000, "candidate never started"
    step += 1
    interrupt = detector.update(
        0.03 * min(1.0, step / 240.0),
        joint_position_rad=0.25 + 0.000735,
        joint_velocity_rad_s=0.013,
        commanded_position_rad=0.25 + 0.005 + 0.000735 * step,
        timestamp_s=step / 240.0,
    )
    assert interrupt.state == FingerContactState.CONTACT_CANDIDATE
    assert interrupt.candidate_steps == 0
    # Freeze again: the moved sample leaves the 7-sample window after 6
    # frames, then exactly 12 consecutive candidate frames confirm.
    confirm_at = None
    for extra in range(1, 30):
        step += 1
        observation = detector.update(
            0.03 * min(1.0, step / 240.0),
            joint_position_rad=0.25 + 0.000735,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=0.25 + 0.005 + 0.000735 * step,
            timestamp_s=step / 240.0,
        )
        if observation.state == FingerContactState.CONTACT_CONFIRMED:
            confirm_at = extra
            break
        assert observation.candidate_steps == max(0, extra - 5)
    assert confirm_at == 17


def test_calibrate_clears_position_history():
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    q = 0.25
    for step in range(1, 12):
        q += 0.000735
        detector.update(
            0.005,
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
    detector.calibrate([0.0] * 24)
    observation = detector.update(
        0.005,
        joint_position_rad=0.25,
        joint_velocity_rad_s=0.013,
        commanded_position_rad=0.255,
        timestamp_s=13 / 240.0,
    )
    assert observation.position_velocity_sample_count == 1
    assert observation.position_derived_velocity_rad_s is None
    assert observation.stall_velocity_source == "insufficient_history"


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, 25, 1000])
def test_position_velocity_window_steps_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _prod_config(window=bad)


@pytest.mark.parametrize("ok", [1, 24])
def test_position_velocity_window_steps_accepts_boundaries(ok):
    assert _prod_config(window=ok).position_velocity_window_steps == ok


def test_three_detector_histories_are_independent():
    first = FingerContactDetector(_prod_config(), name="f1")
    second = FingerContactDetector(_prod_config(), name="f2")
    third = FingerContactDetector(_prod_config(), name="f3")
    for detector in (first, second, third):
        detector.calibrate([0.0] * 24)
    q = 0.25
    for step in range(1, 12):
        q += 0.000735
        first.update(
            0.005,
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
    first_observation = first.update(
        0.005,
        joint_position_rad=q,
        joint_velocity_rad_s=0.013,
        commanded_position_rad=q + 0.005,
        timestamp_s=12 / 240.0,
    )
    second_observation = second.update(
        0.005,
        joint_position_rad=0.5,
        joint_velocity_rad_s=0.013,
        commanded_position_rad=0.505,
        timestamp_s=1 / 240.0,
    )
    third_observation = third.update(
        0.005,
        joint_position_rad=0.6,
        joint_velocity_rad_s=0.013,
        commanded_position_rad=0.605,
        timestamp_s=1 / 240.0,
    )
    assert first_observation.position_velocity_sample_count == 7
    assert second_observation.position_velocity_sample_count == 1
    assert third_observation.position_velocity_sample_count == 1
    assert first_observation.stall_velocity_source == "position_history"
    assert second_observation.stall_velocity_source == "insufficient_history"
    assert third_observation.stall_velocity_source == "insufficient_history"



def test_derived_velocity_subtraction_overflow_fails_closed():
    # Finite inputs, but q_new - q_old overflows to infinity inside the
    # position window: the detector must fail closed with a precise error.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    with pytest.raises(ValueError, match="non-finite position-derived"):
        for step in range(1, 8):
            position = 1e308 if step <= 6 else -1e308
            detector.update(
                0.0,
                joint_position_rad=position,
                joint_velocity_rad_s=0.0,
                commanded_position_rad=position,
                timestamp_s=step / 240.0,
            )
    assert detector.state == FingerContactState.FAILED


def test_disagreement_overflow_fails_closed():
    # Finite reported qd and finite derived velocity, but their absolute
    # difference overflows: fail closed instead of emitting inf evidence.
    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    with pytest.raises(ValueError, match="non-finite velocity disagreement"):
        for step in range(1, 8):
            position = 1.25e306 if step <= 6 else -1.25e306
            detector.update(
                0.0,
                joint_position_rad=position,
                joint_velocity_rad_s=1e308,
                commanded_position_rad=position,
                timestamp_s=step / 240.0,
            )
    assert detector.state == FingerContactState.FAILED


def test_observation_as_dict_is_json_safe_warmup_and_normal():
    import json

    detector = FingerContactDetector(_prod_config(), name="f2")
    detector.calibrate([0.0] * 24)
    q = 0.25
    for step in range(1, 10):
        q += 0.000735
        observation = detector.update(
            0.005,
            joint_position_rad=q,
            joint_velocity_rad_s=0.013,
            commanded_position_rad=q + 0.005,
            timestamp_s=step / 240.0,
        )
        json.dumps(observation.as_dict(), allow_nan=False)
    assert observation.stall_velocity_source == "position_history"

