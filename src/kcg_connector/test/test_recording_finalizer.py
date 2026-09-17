"""A write failure must not hide the motion error or skip independent closeouts."""
import ast
import json
from pathlib import Path
import sys

import pytest


def finalizer_with_faults(fail_at):
    # Importing the command wrapper starts Isaac; isolate its real cleanup body.
    source = Path(__file__).parents[1] / 'isaac/run_body_assembly_with_video.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'finalize_recording')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    events = []
    errors = {name: OSError(name + ' failed') for name in fail_at}

    def record(name):
        events.append(name)
        if name in errors:
            raise errors[name]

    class Archive:
        closed = False

        def close(self):
            record('archive')
            self.closed = True

    class Stepper:
        recording_gc_audit = {'cleanup_errors': []}

        def finish_deferred_recording_gc(self, primary_error=None):
            record('gc')

    class Video:
        def close(self):
            record('video')
            return {'frames': 1}

    class Output:
        def __init__(self, name=''):
            self.name = name

        def __truediv__(self, name):
            return Output(name)

        def write_text(self, text):
            json.loads(text)
            record(self.name)

    state = {'runtime': {'truth_stream': Archive(), 'body_assembly_video': Video()},
             'stepper': Stepper(), 'output': Output()}
    namespace = {'recording_state': state, 'sys': sys, 'json': json}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['finalize_recording'], state, events, errors


@pytest.mark.parametrize('fail_at', [[], ['recording_gc_audit.json'],
                                    ['archive', 'gc', 'recording_gc_audit.json', 'video']])
def test_primary_motion_exception_survives_cleanup_failures(fail_at):
    finalize, state, events, _ = finalizer_with_faults(fail_at)
    original = ValueError('original physics failure')
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        finally:
            finalize()
    assert caught.value is original
    assert 'video' in events
    assert len(state['runtime'].get('recording_finalize_errors', [])) == len(fail_at)


@pytest.mark.parametrize('fail_at', [['recording_gc_audit.json'], ['archive', 'video']])
def test_first_cleanup_error_propagates_after_remaining_closeouts(fail_at):
    finalize, _, events, errors = finalizer_with_faults(fail_at)
    with pytest.raises(OSError) as caught:
        finalize()
    assert caught.value is errors[fail_at[0]]
    assert 'gc' in events and 'video' in events


def test_normal_closeout_and_second_call_keep_video_closed_once():
    finalize, state, events, _ = finalizer_with_faults([])
    finalize()
    assert events == ['archive', 'gc', 'recording_gc_audit.json',
                      'video', 'video_render_isolation.json']
    finalize()
    assert events.count('archive') == 1 and events.count('video') == 1
    assert 'recording_finalize_errors' not in state['runtime']
