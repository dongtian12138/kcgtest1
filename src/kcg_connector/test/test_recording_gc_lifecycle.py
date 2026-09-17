"""Recording cleanup must preserve completed frames, aborts and caller GC state."""
import gc

import pytest
from carts_v2.controller import JointSignalStepper


class FrameProbe(JointSignalStepper):
    def __init__(self, error=None, stop=None):
        self.step_index = 0
        self.wall_times = {}
        self.abort_reason = None
        self.latest = object()
        self.error = error
        self.stop = stop
        self.events = []

    def _advance_step(self, *args, **kwargs):
        self.step_index += 1
        self.events.append(('record_and_update_latest', self.step_index))
        self.abort_reason = self.stop
        self.events.append(('apply_original_stops', self.step_index))
        if self.error:
            raise self.error
        return self.latest


def test_periodic_and_nonperiodic_tail_collection_follow_record_and_stops(monkeypatch):
    before = gc.isenabled()
    gc.enable()
    probe = FrameProbe()
    probe.enable_deferred_recording_gc()
    calls = []
    def collect(generation):
        assert probe.events[-1][0] == 'apply_original_stops'
        calls.append((probe.step_index, generation, gc.isenabled()))
        return 0
    monkeypatch.setattr(gc, 'collect', collect)
    try:
        for _ in range(129):
            assert probe.advance('hold', None, None) is probe.latest
            assert gc.isenabled()
        assert calls[127] == (128, 2, False)
        assert calls[128] == (129, 0, False)
        probe.finish_deferred_recording_gc()
        assert calls[-1] == (129, 2, True)
        assert probe.recording_gc_audit['episode_end_collection_completed']
        count = len(calls)
        probe.finish_deferred_recording_gc()
        assert len(calls) == count
        assert probe.wall_times['recording_gc_s'] >= 0.
    finally:
        if not before:
            gc.disable()


def test_preexisting_disabled_gc_is_not_overridden(monkeypatch):
    before = gc.isenabled()
    probe = FrameProbe()
    probe.enable_deferred_recording_gc()
    monkeypatch.setattr(gc, 'collect', lambda _: pytest.fail('Caller disabled GC'))
    gc.disable()
    try:
        assert probe.advance('hold', None, None) is probe.latest
        probe.finish_deferred_recording_gc()
        assert not gc.isenabled()
        assert probe.recording_gc_audit['externally_disabled_calls'] == 1
        assert probe.recording_gc_audit['episode_end_collection_skipped_external_gc_disabled']
    finally:
        if before:
            gc.enable()


def test_nut_only_policy_keeps_earlier_stages_on_original_gc(monkeypatch):
    before = gc.isenabled()
    gc.enable()
    probe = FrameProbe()
    probe.enable_deferred_recording_gc(only_nut_phases=True)
    calls = []
    monkeypatch.setattr(gc, 'collect', lambda generation: calls.append(generation) or 0)
    try:
        probe.advance('body_transport', None, None)
        assert calls == [] and gc.isenabled()
        probe.advance('key_probe_nut_grip_hold', None, None)
        assert calls == [0] and gc.isenabled()
        probe.finish_deferred_recording_gc()
        assert calls == [0, 2]
    finally:
        if not before:
            gc.disable()


def test_cleanup_failure_preserves_primary_physics_exception(monkeypatch):
    before = gc.isenabled()
    gc.enable()
    probe = FrameProbe(error=ValueError('original physics failure'))
    probe.enable_deferred_recording_gc()
    def failed_collection(_):
        raise RuntimeError('cleanup failure')
    monkeypatch.setattr(gc, 'collect', failed_collection)
    try:
        with pytest.raises(ValueError, match='original physics failure'):
            probe.advance('hold', None, None)
        assert gc.isenabled()
        assert probe.abort_reason is None
        assert probe.recording_gc_audit['cleanup_errors'][0]['error'] == 'cleanup failure'
    finally:
        if not before:
            gc.disable()


@pytest.mark.parametrize('original_stop', [None, 'ORIGINAL_WRIST_STOP'])
def test_cleanup_error_stops_or_preserves_existing_stop(monkeypatch, original_stop):
    before = gc.isenabled()
    gc.enable()
    probe = FrameProbe(stop=original_stop)
    probe.enable_deferred_recording_gc()
    def failed_collection(_):
        raise RuntimeError('cleanup failure')
    monkeypatch.setattr(gc, 'collect', failed_collection)
    try:
        assert probe.advance('hold', None, None) is probe.latest
        assert probe.abort_reason == (original_stop or 'RECORDING_GC_CLEANUP_FAILED')
        assert gc.isenabled()
    finally:
        if not before:
            gc.disable()
