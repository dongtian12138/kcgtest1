# Offline collision audit v1

This standalone package checks the exact command samples used by the current
synthetic connector tabletop pick.  It does not replace the active SRDF,
modify the robot asset, start Isaac Sim, or use a GPU.

The audited sequence is the 240 Hz schedule from initial settle through Home
hand opening, the three approach segments, pregrasp hold, open-hand descent,
tare, physical hand closure, preload, lift, and final hold.  It contains
10,656 command samples.

The scene contains conservative collision solids for the table, fixture,
fixed endpoint and loose endpoint.  During closure, only loose-endpoint
contact with the eight finger links is ignored.  At lift, the endpoint is
attached rigidly to `grasp_tcp` with those same touch links.  This attachment
is an offline collision-checking assumption, not evidence that physics can
form or retain the grasp.

Two discrete policies are checked at every sample:

1. The independent 1,000,000-trial candidate SRDF as generated.
2. A stricter in-memory ACM in which all 76 `reason=Never` pairs are restored
   to collision checking.  The 16 `Adjacent` pairs remain disabled.

Run from the workspace root:

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run kcg_moveit_collision_audit connector_pick_collision_audit \
  --project-root "$PWD" \
  --config src/kcg_moveit_collision_audit/config/connector_pick_collision_audit_v1.yaml \
  --output artifacts/kcg_connector/planning_scene_collision_audit_v1/report.yaml
```

The default command exits with code 2 unless every required gate passes.
`--report-only` may be added for diagnostics; it changes only the process exit
code and never changes `passed` or `status` in the report.

The 2026-08-12 run checked all 10,656 samples.  Both discrete policies found
zero self-collision samples and zero robot-environment collision samples.  The
minimum forbidden self distance was 1.5373855 mm between `handbase_link` and
`f3Link2` at Home.  The minimum environment distance was only 3.33 nm between
the attached endpoint and table on the first lift sample.  This near-zero
value is the expected consequence of beginning the lift at the tabletop
contact surface; it is not a positive environment-clearance margin.

The final result deliberately remains
`FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED`.  The installed MoveIt 2 Humble
FCL backend does not implement continuous collision checking.  A finite set
of collision-free samples, even at 240 Hz and with a bounded 0.0019532 rad
maximum command step, cannot prove the swept path collision-free.  The report
also does not verify world-world contacts, grasp/contact physics, or fidelity
to a flight connector CAD model.

## D38999 Shell 25/J profile

The independent D38999 profile consumes the versioned D38999 pick, tabletop
scene and public-dimensional proxy directly.  Its stable pick input SHA-256 is
`bf288bfde8550bbfa7ca0583beed5e23fb51efe49dd3c93faa28e06448ee5519`.
It checks the primary blend-0.80 candidate that descends open to TCP z=0.247
m, closes toward `[1.0, 0.72, 0.56, 0.72]`, seats the closed hand to z=0.236
m, then preloads and lifts.

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run kcg_moveit_collision_audit connector_pick_collision_audit \
  --project-root "$PWD" \
  --config src/kcg_moveit_collision_audit/config/d38999_pick_collision_audit_v1.yaml \
  --output artifacts/kcg_connector/planning_scene_collision_audit_d38999_v1/report_z247_z236_b080.yaml
```

The final 2026-08-12 CPU run checked all 11,376 samples.  Both the candidate ACM
and the strict policy that restores all 76 `Never` pairs found zero
self-collision, environment-collision and robot-link/table-collision samples.
The minimum robot-link/table distance was 5.8749658 mm at the final closed-hand
seating sample, between `f2Link2` and the table, so the explicit 5 mm table
margin gate passed.  The report SHA-256 is
`5d154b08a3345f5c8aa4c19deafb437306d6748296c6f9b58c9aa28e295ba116`.

At the preload target, isolated FCL queries put all three fingers beyond first
contact with the 48 mm conservative plug proxy: -4.5945 mm, -9.9252 mm and
-4.5957 mm signed target distance.  This establishes geometric reachability,
not a realizable penetrating state or force closure; PhysX must stop the
effort-controlled fingers at finite contact.  The overall status remains
`FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED` solely because Humble FCL cannot
provide the required continuous swept-path proof.

The earlier z=0.242 to z=0.230, hand `[1.0, 0.90, 0.70, 0.90]` screening run is
retained as a rejection artifact in `report.yaml`.  It first self-collided at
closure step 621 and accumulated 2,380 self-colliding samples; it is not the
current candidate.

## D38999 closure-clearance analysis

The independent closure analyzer was used to screen the earlier z=0.242
closure state.  It restores all 76 `Never` pairs, scans 20,000
closure blends, and bisects the first 1 mm signed self-clearance boundary.  It
also removes the normal closure contact mask one finger at a time to query
that finger's signed distance to the conservative 48 mm plug proxy.

```bash
ros2 run kcg_moveit_collision_audit d38999_closure_clearance_analyzer \
  --project-root "$PWD" \
  --config src/kcg_moveit_collision_audit/config/d38999_pick_collision_audit_v1.yaml \
  --output artifacts/kcg_connector/d38999_closure_clearance_analysis_v1/report.yaml
```

The exact 1 mm prefix boundary is blend 0.8792183472, command
`[1.0, 0.7912965125, 0.6154528430, 0.7912965125]`.  Its limiting pair is
`f2Link2` / `f3Link3`.  In the actual 840-step closure schedule, step 616 is
the last sample before that boundary, step 617 falls to 0.9425 mm, and step
621 first collides.

For the first lower-preload physics candidate, blend 0.80 gives command
`[1.0, 0.72, 0.56, 0.72]` and 1.9976200 mm strict self distance at the
z=0.242 m analysis pose.  All three fingers can geometrically reach the plug;
their target signed distances are -3.7530 mm, -9.6669 mm and -3.7543 mm.
Blend 0.85 is retained only as a higher-preload fallback.  Its command is
`[1.0, 0.765, 0.595, 0.765]`, strict self distance is 1.9969280 mm, and target
plug distances are -11.1987 mm, -14.7826 mm and -11.3085 mm.

These negative plug distances are not realizable collision-free position
states.  They mean the position target lies beyond first rigid contact and is
therefore only a PD/effort preload candidate.  PhysX must demonstrate that the
2 N m per-finger limit stops the fingers at finite three-finger contact without
self-collision.  The analyzer does not prove force closure, contact dynamics,
or continuous collision safety.
