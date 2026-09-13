#!/usr/bin/env python3
"""Real-time limit including simulator startup, work, and graceful evidence close."""
import os
import signal
import subprocess
import sys
import time
import json


def main(argv):
    if not argv:raise SystemExit('simulator command required')
    limit=float(os.environ.get('KCG_EXPERIMENT_WALL_LIMIT_S','300'))
    grace=float(os.environ.get('KCG_EXPERIMENT_CLOSEOUT_RESERVE_S','30'))
    if not 0<grace<limit<=300:raise SystemExit('this experiment requires 0 < closeout reserve < total limit <= 300 seconds')
    started=time.monotonic();env=os.environ.copy()
    env.update(KCG_BOUNDED_EXPERIMENT='1',KCG_EXPERIMENT_ACTION_DEADLINE=str(started+limit-grace))
    child=subprocess.Popen(argv,env=env,start_new_session=True)
    terminate_grace=min(2.,grace/2.)
    try:
        return child.wait(timeout=limit-terminate_grace)
    except subprocess.TimeoutExpired:
        print('EXPERIMENT_WALL_BUDGET_REACHED: terminating this process group; no automatic extension',flush=True)
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=terminate_grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        return 124
    except KeyboardInterrupt:
        os.killpg(child.pid,signal.SIGINT)
        try:child.wait(timeout=grace)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        return 130
    finally:
        print('BOUNDED_EXPERIMENT_TIMING '+json.dumps({'wall_seconds':time.monotonic()-started,
            'wall_limit_seconds':limit,'child_returncode':child.returncode}),flush=True)


if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
