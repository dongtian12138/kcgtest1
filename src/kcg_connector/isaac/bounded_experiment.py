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
    local_turn=(any(str(a).endswith('/diagnose_saved_hand_wrench.py') for a in argv)
                and '--interface-twist-deg' in argv and '--interface-release-at-end' in argv)
    # A complete finite local turn may explicitly budget its measured full
    # connector solve cost. Default and other experiment ceilings stay300s.
    full_current_hand_turn=(local_turn and '--interface-regrasp-stroke-deg' in argv
                           and '--shared-hand-mechanism' in argv)
    visual_assembly=(any(str(a).endswith('/run_body_assembly_with_video.py') for a in argv)
                     and all(a in argv for a in ('--visual-body-start','--body-key-entry',
                                                 '--body-nut-regrasp','--hand-mechanism-config')))
    source_grip_probe=(any(str(a).endswith('/diagnose_saved_hand_wrench.py') for a in argv)
                       and '--source-stage-probe' in argv and '--shared-hand-mechanism' in argv)
    fixed_camera_prefix=(any(str(a).endswith('/run_body_assembly_with_video.py') for a in argv)
                         and all(a in argv for a in ('--visual-body-start','--postgrasp-key-observation',
                                                      '--hand-mechanism-config'))
                         and '--body-assembly-transport' not in argv)
    fixed_camera_transport=(any(str(a).endswith('/run_body_assembly_with_video.py') for a in argv)
                         and all(a in argv for a in ('--visual-body-start','--postgrasp-key-observation',
                            '--hand-mechanism-config','--body-assembly-transport'))
                         and '--body-key-entry' not in argv)
    initial_grasp_only=(any(str(a).endswith('/run_body_assembly_with_video.py') for a in argv)
                       and all(a in argv for a in ('--visual-body-start','--hand-mechanism-config'))
                       and '--body-assembly-transport' not in argv
                       and '--postgrasp-key-observation' not in argv)
    visual_entry_prefix=(any(str(a).endswith('/run_body_assembly_with_video.py') for a in argv)
                         and all(a in argv for a in ('--visual-body-start','--body-key-entry',
                                                     '--hand-mechanism-config','--body-assembly-transport'))
                         and '--body-nut-regrasp' not in argv)
    # The current-hand baseline costs about 43 physics-wall seconds per
    # simulated second. An explicitly requested four-stroke run also includes
    # three measured releases, wrist returns and new grip preparations.
    # The earlier whole visual episode used about 20,093 wall seconds for
    # 455 simulated seconds. Faster Nut motion reduces that part, while the
    # original pickup, visual insertion and evidence archive remain costly.
    # Only an explicitly requested complete visual run may set this budget.
    source_probe_ceiling=900.
    if source_grip_probe:
        probe_file=argv[argv.index('--source-stage-probe')+1]
        with open(probe_file) as stream:probe_settings=json.load(stream)
        additional=int(probe_settings.get('additional_strokes',0))
        # Two numerically sampled90degree profiles total179.999999947deg.
        # This tolerance classifies only the wall budget, not a motion limit.
        late=180.-1e-6<=float(probe_settings.get('initial_loaded_command_deg',0.))<=360.+1e-6
        if late:source_probe_ceiling=5400. if probe_settings.get('release_after_rotation') else 1500.
        elif (80.-1e-6<=float(probe_settings.get('initial_loaded_command_deg',0.))<180.-1e-6
                and not additional and probe_settings.get('release_after_rotation')):
            # A declared captured-interface source probe includes a fresh grasp,
            # one bounded turn and a real3s released observation. The preserved
            # deep-grasp25 run cost2122.49s in this same recording mode. Permit
            # an explicit2400s ceiling for this earlier one-stroke release case;
            # no physics, force, angle or recording limit changes or live extension.
            source_probe_ceiling=2400.
        if additional:
            if not 1<=additional<=3:raise SystemExit('finite source continuation requires one to three additional strokes')
            # The measured deep pin/bore contact workload is larger than the
            # initial capture workload. Reuse the existing 3000s ceiling for
            # an explicitly declared late two-stroke diagnostic as well.
            # Dense late mating plus actual regrasp and a final released hold
            # costs far more than a single turn. Use an explicit finite budget
            # based on the measured late contact-recording workload.
            source_probe_ceiling=5400. if late and additional==1 else 1800. if additional==1 else 3000.
    # The verified late release adds a fifth grip/turn to the original visual
    # episode: fast04 cost7476s before its failed third turn, the valid late
    # transfer14 cost3733s, final turn/released hold22 cost1592s, and the new
    # third-grasp/turn/release24 cost2152s. Allow an explicit18000s ceiling
    # for this measured dense-contact workload and finite evidence closeout;
    # this changes no motion, force, geometry, or physics limit and does not
    # extend a run that has already started.
    # The 2026-09-17 candidate permits one more finite regrasp within the
    # same360degree budget and compares articulation contact solve order.
    # The measured near-seat9s source segment alone cost2102.68wall seconds.
    # Allow an explicitly selected21600s complete-episode ceiling; defaults,
    # local probes, all motion/force limits and the no-live-extension rule stay.
    # The user-authorized fixed-camera prefix includes the original19s lift
    # and2s physical hold at960Hz. Permit an explicit15min budget only for
    # this no-transport visibility run; no runtime budget is extended live.
    ceiling=21600. if visual_assembly else 7200. if visual_entry_prefix else 3600. if fixed_camera_transport else 900. if fixed_camera_prefix or initial_grasp_only else 2400. if full_current_hand_turn else source_probe_ceiling if source_grip_probe else 600. if local_turn else 300.
    if not 0<grace<limit<=ceiling:raise SystemExit(f'this experiment requires 0 < closeout reserve < total limit <= {ceiling:g} seconds')
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
