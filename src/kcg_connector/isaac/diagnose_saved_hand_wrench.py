#!/usr/bin/env python3
"""Static original-hand wrench check against an explicitly mounted Nut fixture.

Initialization uses a completed run's pose and finite drive targets. The Nut is
a declared laboratory fixture, not an assembly success or hidden support.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time


def initialize_fourbar_pose(positions, couplings, joint_limits_rad, geometry_uncertainty_m):
    """Resolve a new pre-physics mechanism pose without relaxing hinge limits.

    A declared exact source lower stop may need a tiny closure correction from
    CAD alignment. Permit only that case and only within the contract's linear
    geometry uncertainty; the corrected angle comes from closure inversion.
    """
    result={name:float(value) for name,value in positions.items()}
    records=[]
    if not math.isfinite(geometry_uncertainty_m) or geometry_uncertainty_m<=0:
        raise ValueError('Four-bar initialization requires a positive CAD alignment uncertainty')
    for follower,coupling in couplings.items():
        source=coupling.source_joint
        source_bounds=joint_limits_rad[source];follower_bounds=joint_limits_rad[follower]
        low,high=coupling.source_interval(source_bounds,follower_bounds)
        original=result[source];corrected=original
        if not low<=original<=high:
            source_lower=float(source_bounds[0])
            p,_=coupling.position_and_derivative(original)
            radius=math.hypot(*coupling.distal_anchor_xy_m)
            lower_stop_displacement=2.*radius*abs(math.sin((p-follower_bounds[0])/2.))
            tolerance=64.*math.ulp(max(1.,abs(original),abs(source_lower)))
            if (original<low and abs(original-source_lower)<=tolerance
                    and p<follower_bounds[0] and lower_stop_displacement<=geometry_uncertainty_m):
                corrected=low
            else:
                raise ValueError(f'Initial {source}={original} is outside its measured closure interval [{low}, {high}]')
        result[source]=corrected
        value,derivative=coupling.position_and_derivative(corrected)
        old_follower=result[follower];result[follower]=value
        records.append({'source_joint':source,'follower_joint':follower,
            'source_before_rad':original,'source_after_rad':corrected,
            'follower_before_rad':old_follower,'follower_after_rad':value,
            'source_feasible_interval_rad':[low,high],'dp_dq':derivative,
            'source_lower_roundoff_corrected':corrected!=original,
            'source_and_follower_physical_limits_unchanged':True})
    return result,records


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument('--robot-asset',type=Path,help='Explicit existing robot/hand variant for a declared comparison; authored before physics')
parser.add_argument('--hand-role',choices=('production-nails','nailfree-comparison'),default='production-nails',
                    help='Production requires original nails; nail-free results are comparison-only')
parser.add_argument('--palm-layout-mechanism',choices=('self-lock','legacy-servo','shared-worm-pd'),
                    help='User-confirmed worm self-lock is the production default; legacy servo is comparison-only')
parser.add_argument('--shared-hand-mechanism',type=Path,
                    help='Use the same four-motor runtime as the visual assembly path, including the actively driven palm')
parser.add_argument('--source-stage-probe',type=Path,
                    help='Bounded current-controller diagnostic from a sealed source-stage state')
parser.add_argument('--source-stage-root',choices=('socket_transport','sequence','.'),default='socket_transport',
                    help='Recorded stage layout: a full visual episode or a declared local continuation')
parser.add_argument('--truth-archive-codec',choices=('jsonl','msgpack'),default='jsonl',
                    help='lossless raw observation codec for source-stage diagnostics')
parser.add_argument('--contact-audit-mode',choices=('full','native-report'),default='full',
                    help='retain all native contact points, optionally omitting duplicate tensor/friction diagnostics')
parser.add_argument('--finger-mechanism',type=Path,
                    help='Explicit measured four-bar candidate contract; default retains historical linear mimic')
parser.add_argument('--finger-worm-self-lock',action='store_true',
                    help='Route motor position commands through the validated single-finger passive split transmission candidate')
parser.add_argument('--interface-grip-only',action='store_true',
                    help='Zero-twist free-connector grip/hold/release; disable insertion force following')
parser.add_argument("--assembly-config",type=Path,
                    help="Explicit local physical-model configuration; defaults to the sealed source configuration.")
parser.add_argument("--mounted-grasp-recipe",type=Path,
                    help="Explicit fixed-Nut laboratory grasp/torque recipe, not an assembly run.")
parser.add_argument("--velocity-iterations", type=int, choices=(0,1,4,16), default=1)
parser.add_argument("--contact-convergence-check", action="store_true")
parser.add_argument("--diagnostic-center-socket-before-start", action="store_true")
parser.add_argument("--main-read-sequence", action="store_true",
                    help="Read the main runner's native gravity/projected/other-articulation signals; targets stay fixed.")
parser.add_argument("--wrist-reference-loads",action="store_true",
                    help="Keep the source hand open and apply bounded reference loads to its wrist sensor subtree; no grasp or assembly.")
parser.add_argument("--position-iterations",type=int,default=32)
parser.add_argument('--experimental-connector-position-convergence',action='store_true',
                    help='Explicit CPU960Hz/4velocity comparison of64 versus128 position iterations; source model and force limits remain unchanged')
parser.add_argument("--closing-drive-cap-nm",type=float,default=1.)
parser.add_argument('--finite-drive-sensitivity',action='store_true',
                    help='Explicit bounded actuator-reference sensitivity up to4Nm; not a hardware rating')
parser.add_argument('--native-model-velocity-limits',action='store_true',
                    help='Enforce the kinematic model joint velocity references in native physics as well as controller commands')
parser.add_argument('--hand-stiffness-nm-rad',type=float,default=12.,help='Finite hand position-drive gain; torque caps remain independent')
parser.add_argument('--hand-damping-nm-s-rad',type=float,default=2.,help='Finite hand drive damping for explicit coupled-contact stability comparisons')
parser.add_argument('--arm-stiffness-nm-rad',type=float,choices=(2500.,10000.),default=2500.,
                    help='Declared arm position-bandwidth comparison; original100Nm effort boundary retained')
parser.add_argument('--mimic-natural-frequency',type=float,
                    help='Explicit finite PhysX mimic compliance preserving source1:1 coupling; uncalibrated transmission stiffness reference')
parser.add_argument("--frozen-connector-model",type=Path)
parser.add_argument("--connector-initial-pose",type=Path,
                    help="Declared bench Body/Nut relative pose, rigidly relocated to the saved Nut before physics; includes relocating its static socket")
parser.add_argument("--interface-twist-deg",type=float,
                    help="Minimal original-hand interface test: bounded wrist-joint7 stroke over one second, then hold")
parser.add_argument('--interface-motion-duration-s',type=float,help='Finite declared local comparison stroke duration')
parser.add_argument('--interface-wall-limit-s',type=float,default=240.,help='Finite local test wall time, at most360seconds')
parser.add_argument('--interface-release-at-end',action='store_true',help='Physically open the fingers and observe passive connector retention after the finite turn')
parser.add_argument('--interface-regrasp-stroke-deg',type=float,
                    help='Finite wrist stroke followed by real finger opening and open-hand wrist return; total angle remains interface-twist-deg')
parser.add_argument('--contact-record-stride',type=int,choices=(4,8,30,120),default=120,
                    help='Physics steps between posthoc contact-point records; never a control input')
parser.add_argument('--raw-contact-only',action='store_true',
                    help='Use native unfiltered contact records; omit unused per-filter contact matrices and invalid friction readback')
parser.add_argument('--sdf-resolution-override',type=int,choices=(512,1024),
                    help='Explicit collision-grid resolution comparison; original CAD triangles and frozen asset remain unchanged')
parser.add_argument("--interface-start-open",action='store_true',
                    help="Start at the existing open-hand grasp posture and close with finite drives before the interface stroke")
parser.add_argument('--interface-extra-closure-deg',type=float,default=0.,
                    help='Bounded additional target closure on the three independent closing joints; drive caps unchanged')
parser.add_argument('--interface-grasp-relation',type=Path,
                    help='Declared CAD Nut-to-hand relation for a local pre-physics fixture placement; never robot alignment success')
parser.add_argument('--interface-force-following',action='store_true',
                    help='Restore original bounded axial/lateral wrist-force following in the declared local fixture test')
parser.add_argument('--interface-grip-recipe',type=Path,
                    help='Explicit original-CAD grasp posture and robot-sensor force recipe; historical15N contact reference is not calibrated hardware capacity')
parser.add_argument('--interface-control-decimation',type=int,choices=(1,2,4),default=1,
                    help='Explicit controller period in physics steps; physical contact solving and evaluation remain at960Hz')
parser.add_argument('--interface-legacy-pad-sensitivity','--interface-source-pad-material',dest='interface_source_pad_material',action='store_true',
                    help='Unvalidated legacy material sensitivity only; the asset coefficient is not a material calibration or formal acceptance basis')
parser.add_argument("--cpu-wrist-reference",type=Path)
parser.add_argument('--probe-in-turn-feedback',action='store_true',
                    help='Use current RGBD and encoder feedback between bounded nut-turn intervals; no object truth.')
parser.add_argument('--probe-segmented-observation-only',action='store_true',
                    help='Declared ablation: acquire the same interval images but retain the prepared rigid grip relation.')
parser.add_argument("--probe-open-grasp-recipe",type=Path,
                    help="Declared pre-physics open-hand joint state for a fresh grasp on the free connector")
parser.add_argument("--fabric-gpu-interop",action="store_true",
                    help="Explicitly request GPU-resident Fabric output at process startup for the bounded rendering consistency check.")
parser.add_argument('--cpu-fabric-output',action='store_true',
                    help='Keep CPU physics and its parameters; publish transforms through Fabric instead of per-tick USD writes.')
parser.add_argument("--gpu-host-readback",action="store_true",
                    help="Retain GPU dynamics but allow host result readback before initialization; a bounded physics-output/data-path diagnostic.")
parser.add_argument('--experimental-connector-gpu-comparison',action='store_true',
                    help='Explicit CPU/GPU comparison of the unchanged connector; does not transfer CPU model validation to GPU')
parser.add_argument('--experimental-connector-time-resolution',action='store_true',
                    help='Explicit CPU rate comparison: mounted interface480Hz or source-stage240/480Hz; default960Hz validation remains separate')
parser.add_argument("--independent-robot-rigid-frames",action="store_true",
                    help="Before reset, give nested robot rigid bodies independent transform stacks while preserving all source world poses and joint properties.")
parser.add_argument("--standard-render-steps",action="store_true",
                    help="Bounded output-path check: use the SDK normal render-step path with rendering_dt equal to physics_dt and verify exact step count.")
parser.add_argument("--arm-damping", type=float,
                    help="Default to the source sample's actual damping, including its earlier lift transition.")
parser.add_argument("--free-plug-in-socket", action="store_true",
                    help="Reuse the source assembly scene and its free Body-Nut joint, with no diagnostic mount.")
parser.add_argument("--paused-world-render", action="store_true",
                    help="At one second call the still-existing observation World.render once while paused.")
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0,1), default=1)
parser.add_argument("--hand-friction-effort-from-urdf", action="store_true")
parser.add_argument("--nut-fingertip-sdf", action="store_true")
parser.add_argument("--nut-fingertip-sdf-link",nargs='+',choices=('f1Link3','f2Link2','f3Link3'))
parser.add_argument("--physics-device", choices=("cuda:0","cpu"), default="cuda:0")
parser.add_argument("--solver-type", choices=("TGS","PGS"), default="TGS")
parser.add_argument("--robot-state-before-reset", action="store_true",
                    help="Author the complete source robot joint state and finite drives before the SDK's two warmup ticks.")
parser.add_argument("--audit-source-key-sdf", action="store_true",
                    help="After the static measurement, read native core SDF at original key-surface samples; never a controller input.")
parser.add_argument("--replay-source-drive-targets", action="store_true",
                    help="In the two-second diagnostic replay the following sealed robot commands, then hold the last one; no online object/contact inputs.")
parser.add_argument("--physics-hz", type=int, choices=(240,480,960), default=240,
                    help="Bounded time-resolution check for the two-second fixed-target diagnostic.")
parser.add_argument("--source-step", type=int)
parser.add_argument("--sensor-stage", default="nut_regrasp_after_index")
parser.add_argument("--source-rotation-stage", default="nut_rotation_after_index")
parser.add_argument("--source-grip-stage", default="nut_regrasp_after_index")
parser.add_argument("--probe-reobserve-axis", action="store_true")
parser.add_argument("--probe-additional-turn-deg", type=float)
parser.add_argument("--probe-speed-deg-s", type=float, default=2.)
parser.add_argument("--probe-arm-speed-limit-rad-s", type=float)
parser.add_argument("--probe-accel-deg-s2", type=float)
parser.add_argument("--probe-torsional-stop-nm", type=float)
parser.add_argument("--probe-lateral-stop-n", type=float)
parser.add_argument("--probe-grasp-axis-shift-m", type=float, default=0.)
parser.add_argument("--probe-grip-effort-scale", type=float, default=1.)
parser.add_argument("--closing-reaction-stop-nm", type=float, default=.9)
parser.add_argument('--closing-reaction-record-only',action='store_true',help='Record legacy projected-joint reaction reference; finite motor drive caps remain active')
parser.add_argument("--probe-visual-alignment-feedback", action="store_true")
parser.add_argument("--probe-visual-position-feedback", action="store_true")
parser.add_argument("--probe-hold-grip-through-preparation", action="store_true")
parser.add_argument("--probe-grip-hold-stiffness-scale", type=float)
parser.add_argument("--probe-guided-fit-continuation", action="store_true")
parser.add_argument("--probe-geometric-fit-feedback", action="store_true")
parser.add_argument("--probe-pin-fit-budget-m", type=float, default=.00004)
parser.add_argument("--probe-planned-contact-force-stop-n", type=float)
parser.add_argument("--probe-grip-effort-weights", type=float, nargs=3)
parser.add_argument("--probe-grip-lateral-balance", action="store_true")
parser.add_argument("--probe-repeat-after-reindex", action="store_true")
parser.add_argument("--probe-unload-after-lateral-hold-stop", action="store_true")
parser.add_argument("--probe-second-grasp-axis-shift-m", type=float)
parser.add_argument("--probe-second-regrasp-effort-time-constant-s", type=float)
parser.add_argument("--probe-start-open", action="store_true")
parser.add_argument("--probe-regrasp-only", action="store_true")
parser.add_argument("--probe-hold-finger-targets", action="store_true")
parser.add_argument("--probe-planar-stiffness-n-m", type=float)
parser.add_argument("--probe-hold-planar-position", action="store_true")
parser.add_argument("--probe-freeze-planar-after-preparation", action="store_true")
parser.add_argument("--probe-force-consistent-planar-range",action='store_true')
parser.add_argument("--probe-guided-socket-pivot", action="store_true")
args = parser.parse_args()
if args.shared_hand_mechanism:
    if not args.robot_state_before_reset or not args.finger_mechanism or (args.interface_twist_deg is None and args.source_stage_probe is None) or args.finger_worm_self_lock:
        parser.error('Shared hand requires pre-reset source state, fourbar contract and interface mode, without a second worm integrator')
    args.palm_layout_mechanism='shared-worm-pd'
if args.palm_layout_mechanism is None:
    args.palm_layout_mechanism='self-lock' if args.hand_role=='production-nails' else 'legacy-servo'
if args.hand_role=='production-nails' and args.palm_layout_mechanism not in ('self-lock','shared-worm-pd'):
    parser.error('The production palm mechanism must reflect the user-confirmed mechanical self-lock')
if args.palm_layout_mechanism=='self-lock' and not args.robot_state_before_reset:
    parser.error('The mechanical palm lock must be authored at the declared pre-physics joint state')
if args.finger_worm_self_lock and (args.finger_mechanism is None or args.interface_twist_deg is None
        or args.hand_friction_effort_from_urdf):
    parser.error('Worm integration requires the measured four-bar local interface and no duplicate legacy URDF friction model')
if args.interface_grip_only and (args.interface_twist_deg!=0. or not args.interface_force_following
        or args.interface_regrasp_stroke_deg is not None):
    parser.error('Grip-only requires zero requested twist and the existing force-feedback grip controller')
if args.source_stage_probe and (not args.shared_hand_mechanism or not args.free_plug_in_socket
        or args.interface_twist_deg is not None or args.probe_additional_turn_deg is not None):
    parser.error('Source-stage diagnosis requires the shared hand and free connector; other probe modes are mutually exclusive')
source_stage_recipe=json.loads(args.source_stage_probe.read_text()) if args.source_stage_probe else None
if source_stage_recipe is not None:
    from te_nut_motion import source_probe_rotation_degrees
    try:
        source_probe_rotation_degrees(source_stage_recipe)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if 'solve_articulation_contact_last' in source_stage_recipe:
        same_rate_profile=(args.physics_hz==960 and args.velocity_iterations==4
            and args.position_iterations in ((16,32,64,128) if args.experimental_connector_position_convergence else (64,)))
        declared_rate_profile=(args.experimental_connector_time_resolution and args.physics_hz in (240,480)
            and args.position_iterations==64 and args.velocity_iterations==4
            and not args.experimental_connector_position_convergence)
        declared_gpu_profile=(args.experimental_connector_gpu_comparison and args.physics_device=='cuda:0'
            and args.gpu_host_readback and args.physics_hz in (480,960)
            and (args.position_iterations,args.velocity_iterations)==(64,4)
            and (args.physics_hz==960 or args.experimental_connector_time_resolution))
        if (type(source_stage_recipe['solve_articulation_contact_last']) is not bool
                or args.frozen_connector_model is None or (args.physics_device != 'cpu' and not declared_gpu_profile)
                or args.solver_type != 'TGS' or not args.external_forces_every_iteration
                or not (same_rate_profile or declared_rate_profile or declared_gpu_profile)):
            parser.error('Contact order requires the declared CPU source-stage numerical profile; model acceptance must be rechecked')
finger_mechanism_document=None
if args.finger_mechanism is not None:
    args.finger_mechanism=args.finger_mechanism.resolve()
    finger_mechanism_document=json.loads(args.finger_mechanism.read_text())
    if (finger_mechanism_document.get('schema')!='kcg.hand.fourbar.v1'
            or not finger_mechanism_document.get('mechanism_id')):
        parser.error('A named measured four-bar mechanism contract is required')
    if not args.robot_state_before_reset or args.interface_grasp_relation is None:
        parser.error('Four-bar candidates require a matching CAD grasp and pre-reset robot initialization')
    if (args.probe_additional_turn_deg is not None or args.mounted_grasp_recipe is not None
            or args.wrist_reference_loads or args.probe_open_grasp_recipe is not None or args.replay_source_drive_targets):
        parser.error('Four-bar updates are integrated only in this local interface loop; delegated legacy probes and replay are not supported')
    if args.interface_force_following and args.interface_grip_recipe is None:
        parser.error('Four-bar force following requires a mechanism-matched grip recipe')
    candidate_geometry=json.loads(args.interface_grasp_relation.read_text())
    if candidate_geometry.get('finger_mechanism_id')!=finger_mechanism_document['mechanism_id']:
        parser.error('The CAD grasp was generated for a different finger mechanism; legacy grasp reuse is prohibited')
if not math.isfinite(args.hand_stiffness_nm_rad) or not 0<args.hand_stiffness_nm_rad<=500.:
    parser.error('Hand position-drive stiffness must be finite and at most500Nm/rad')
if not math.isfinite(args.hand_damping_nm_s_rad) or not 0<=args.hand_damping_nm_s_rad<=5.:
    parser.error('Hand drive damping must be finite and within the declared0to5Nm*s/rad comparison range')
if args.mimic_natural_frequency is not None and not (math.isfinite(args.mimic_natural_frequency) and 0<args.mimic_natural_frequency<=20000):
    parser.error('Mimic natural frequency must be finite and within the bounded comparison range')
diagnostic_started=time.monotonic()
frozen_requirements=None
if args.frozen_connector_model is not None:
    manifest_path=args.frozen_connector_model.resolve().with_name('validation_manifest.json')
    if manifest_path.is_file():
        frozen_requirements=json.loads(manifest_path.read_text()).get('runtime_requirements')
if args.connector_initial_pose is not None and (args.frozen_connector_model is None or not args.free_plug_in_socket):
    parser.error('The declared benchmark pose requires the full free connector and its frozen model')
if args.interface_twist_deg is not None and (args.connector_initial_pose is None or
        not ((0 < args.interface_twist_deg <= 360.) or (args.interface_grip_only and args.interface_twist_deg==0.))
        or args.probe_additional_turn_deg is not None or args.replay_source_drive_targets):
    parser.error('The local grasp comparison requires a declared assembled pose and a finite stroke no larger than360degrees')
wall_cap=(2400. if args.shared_hand_mechanism else 1080.) if args.interface_regrasp_stroke_deg is not None else 600.
if not 0<args.interface_wall_limit_s<=wall_cap:
    parser.error('The local test exceeds its finite wall-time range')
if args.interface_release_at_end and args.interface_grasp_relation is None:
    parser.error('A physical release requires the declared original-CAD open-hand posture')
if args.interface_regrasp_stroke_deg is not None and not (args.interface_force_following
        and args.interface_grip_recipe and args.interface_release_at_end
        and 0<args.interface_regrasp_stroke_deg<=120.):
    parser.error('Physical regrasp requires a finite stroke, root-load recipe, force following and final release')
if args.interface_motion_duration_s is not None and not 0 < args.interface_motion_duration_s <= 20.:
    parser.error('The local comparison motion duration must be finite and at most20seconds')
if args.interface_start_open and args.interface_twist_deg is None:
    parser.error('The open-hand initialization is scoped to the minimal interface test')
if args.interface_grasp_relation is not None and not args.interface_start_open and not args.source_stage_probe:
    parser.error('The declared CAD grasp relation requires the open-hand local interface test')
if args.interface_force_following and not args.interface_start_open:
    parser.error('Force following requires the declared open-hand interface test')
if args.interface_control_decimation!=1 and not args.interface_force_following:
    parser.error('Controller decimation is scoped to the explicit force-following interface test')
interface_grip_recipe=None
# User explicitly removed the contact-normal15N restriction. Force targets
# are experimental commands, with finite motor effort and travel limits;
# the directional tip-rope test is not a universal contact-force capacity.
if args.interface_grip_recipe is not None:
    interface_grip_recipe=json.loads(args.interface_grip_recipe.read_text())
    if finger_mechanism_document is not None:
        geometry_path=Path(interface_grip_recipe['source_geometry_plan'])
        if not geometry_path.is_absolute():geometry_path=Path(__file__).resolve().parents[3]/geometry_path
        recipe_geometry=json.loads(geometry_path.read_text())
        if recipe_geometry.get('finger_mechanism_id')!=finger_mechanism_document['mechanism_id']:
            parser.error('The force-feedback geometry belongs to a different finger mechanism')
    if float(interface_grip_recipe.get('hand_position_stiffness_nm_rad',12.))!=args.hand_stiffness_nm_rad:
        parser.error('Hand force-controller conversion must match the finite position-drive stiffness')
    if (not args.interface_force_following or not args.interface_grasp_relation
            or len(interface_grip_recipe.get('source_normal_targets_n',[]))!=3
            or not all(math.isfinite(x) and x>0 for x in interface_grip_recipe['source_normal_targets_n'])):
        parser.error('A grasp recipe requires force following, its CAD relation, and three finite positive force references')
    if interface_grip_recipe.get('sidewall_sequence') and (
            len(interface_grip_recipe.get('sidewall_targets_n',[]))!=3 or not all(
            math.isfinite(x) and x>0 for x in interface_grip_recipe.get('sidewall_targets_n',[]))):
        parser.error('Sidewall force references must be finite and positive')
    if args.interface_regrasp_stroke_deg is not None and (not interface_grip_recipe.get('root_moment_control')
            or interface_grip_recipe.get('sidewall_sequence')):
        parser.error('The regrasp comparison uses the direct root-load controller')
    if not 0<float(interface_grip_recipe.get('planned_wrist_force_limit_n',3.0400615))<=(110. if interface_grip_recipe.get('capacity_tested_wrench_envelope') else 20.):
        parser.error('The local planned-contact force must stay within the validated10N connector-load envelope')
if not 0. <= args.interface_extra_closure_deg <= 2. or (args.interface_extra_closure_deg and not args.interface_start_open):
    parser.error('Extra closure requires the open-hand interface mode and is bounded to2degrees')
if args.interface_source_pad_material and args.interface_twist_deg is None:
    parser.error('Source-pad material selection belongs to the explicit interface baseline')
if not 1 <= args.position_iterations <= 255:parser.error('position iterations out of range')
if not 0 < args.closing_drive_cap_nm <= (4.1 if args.finite_drive_sensitivity else 2.7):
    parser.error('closing motor cap outside the bounded simulation diagnostic range')
if args.finite_drive_sensitivity and args.interface_twist_deg is None and not args.source_stage_probe:
    parser.error('Finite actuator-reference sensitivity is scoped to the declared local assembly test')
if args.contact_convergence_check and (args.velocity_iterations!=4 or args.physics_device!='cpu'
        or (args.frozen_connector_model is None and not args.wrist_reference_loads)
        or args.solver_type!='TGS' or not args.external_forces_every_iteration):
    parser.error('the explicit convergence check changes only CPU TGS velocity iterations from 1 to 4')
if args.velocity_iterations==4 and not args.contact_convergence_check and not frozen_requirements:
    parser.error('four velocity iterations require the explicit unvalidated convergence diagnostic')
if args.probe_visual_position_feedback and not args.probe_visual_alignment_feedback:
    parser.error('actual centre feedback requires the current RGBD alignment-feedback mode')
if args.probe_geometric_fit_feedback and not args.probe_visual_position_feedback:
    parser.error('guided feature admission requires actual centre/axis visual feedback')
if args.probe_in_turn_feedback and (args.physics_device!='cpu' or not args.probe_start_open
        or not args.probe_geometric_fit_feedback or not args.probe_hold_grip_through_preparation
        or args.cpu_wrist_reference is None or args.probe_additional_turn_deg is None):
    parser.error('observed-turn pilot requires the CPU wrist reference, real open/regrasp, current pose fit and fixed finite grip')
if args.probe_segmented_observation_only and not args.probe_in_turn_feedback:
    parser.error('segmentation-only ablation requires the declared observed-turn mode')
if not .00004 <= args.probe_pin_fit_budget_m <= .000065:
    parser.error('nominal guided pin admission budget must remain between40 and65micrometres')
if args.probe_grip_hold_stiffness_scale is not None and (not args.probe_hold_grip_through_preparation
        or not 1. < args.probe_grip_hold_stiffness_scale <= 4.):
    parser.error('holding impedance scale requires fixed finite preload and is bounded to four')
if args.probe_guided_fit_continuation and (args.probe_start_open or args.probe_grasp_axis_shift_m
        or not args.probe_reobserve_axis or not args.probe_hold_grip_through_preparation
        or args.probe_grip_hold_stiffness_scale is not None):
    parser.error('guided-fit continuation requires a declared loaded source, current RGBD and unchanged-gain fixed preload')
if args.probe_planned_contact_force_stop_n is not None and (not args.probe_visual_position_feedback
        or not 3.0400615 <= args.probe_planned_contact_force_stop_n <= 5.):
    parser.error('the explicit centred-contact observation window is bounded from the original reference through 5 N')
if args.diagnostic_center_socket_before_start and (not args.free_plug_in_socket or args.probe_additional_turn_deg is not None):
    parser.error('diagnostic socket centering is only a declared pre-physics fixed-command comparison, not an assembly controller')
if args.wrist_reference_loads and (not args.mounted_grasp_recipe or args.physics_device!='cpu'):
    parser.error('wrist reference loading requires the explicit open-hand CPU mounted recipe')
if args.probe_open_grasp_recipe is not None and (not args.probe_start_open or not args.free_plug_in_socket):
    parser.error('open-grasp recipe requires a declared free-connector open-hand start')
if args.solver_type == "PGS" and args.external_forces_every_iteration:
    parser.error("PGS does not support per-iteration external forces; pass --external-forces-every-iteration 0")
if args.gpu_host_readback and args.physics_device=="cpu":
    parser.error("host-readback diagnostic must retain the GPU dynamics scene")
if args.standard_render_steps and not args.mounted_grasp_recipe:
    parser.error("standard render-step check is confined to the mounted Nut diagnostic")
if args.probe_regrasp_only:
    if args.probe_additional_turn_deg is not None:
        parser.error("regrasp-only mode has no requested rotation")
    args.probe_additional_turn_deg = 0.
args.run = args.run.resolve(); args.output = args.output.resolve()
if any(Path(value).name != value for value in (args.sensor_stage,args.source_rotation_stage,args.source_grip_stage)):
    parser.error("source stages must be local directory names")
if args.probe_additional_turn_deg is not None and (not args.free_plug_in_socket or args.source_step is None):
    parser.error("a turn probe requires the free plug scene and an explicit completed source step")
if args.probe_additional_turn_deg is not None and args.physics_hz != 240 and args.frozen_connector_model is None:
    parser.error("the time-resolution check currently applies only to the fixed-target diagnostic")
if args.frozen_connector_model is not None:
    required=frozen_requirements or dict(physics_hz=960,position_iterations=128,
                                       velocity_iterations=4 if args.contact_convergence_check else 1)
    experimental_gpu=(args.experimental_connector_gpu_comparison and args.physics_device=='cuda:0'
                      and args.gpu_host_readback and (args.interface_twist_deg is not None or args.source_stage_probe is not None))
    experimental_rate=(args.experimental_connector_time_resolution
                       and (args.physics_device=='cpu' or (experimental_gpu and args.physics_hz==480))
                       and ((args.physics_hz==480 and args.interface_control_decimation==2
                             and args.interface_twist_deg is not None)
                            or (args.source_stage_probe is not None and args.physics_hz in (240,480))))
    balanced_iterations=bool(experimental_rate and source_stage_recipe
        and source_stage_recipe.get('balanced_cpu_iteration_budget',False)
        and (args.physics_hz,args.position_iterations,args.velocity_iterations)==(240,255,16))
    experimental_position=bool(args.experimental_connector_position_convergence
        and args.source_stage_probe is not None and args.physics_device=='cpu'
        and args.solver_type=='TGS' and args.external_forces_every_iteration
        and args.physics_hz==required['physics_hz']==960
        and args.velocity_iterations==required['velocity_iterations']==4
        and required['position_iterations']==64 and args.position_iterations in (16,32,64,128)
        and not args.experimental_connector_time_resolution and not args.experimental_connector_gpu_comparison)
    if (not args.free_plug_in_socket or (args.physics_device!='cpu' and not experimental_gpu)
            or (args.cpu_wrist_reference is None and not args.source_stage_probe)
            or (args.physics_hz!=required['physics_hz'] and not experimental_rate)
            or (not balanced_iterations and not experimental_position
                and any(getattr(args,k)!=required[k] for k in ('position_iterations','velocity_iterations')))):
        parser.error('The delivered connector requires its declared CPU runtime configuration and a CPU wrist reference')
if args.experimental_connector_position_convergence and not (args.frozen_connector_model and experimental_position):
    parser.error('Position convergence requires an explicit frozen-connector CPU960Hz/16,32,64,128/4 source-stage comparison')
if args.experimental_connector_gpu_comparison and not (args.frozen_connector_model and args.physics_device=='cuda:0'
        and args.gpu_host_readback and (args.interface_twist_deg is not None or args.source_stage_probe is not None)):
    parser.error('Experimental GPU comparison requires the frozen connector, local interface probe, CUDA and host readback')
if args.experimental_connector_time_resolution and not (args.frozen_connector_model and experimental_rate):
    parser.error('Time-resolution comparison requires the frozen connector and the declared bounded CPU interface or source-stage probe')
if args.audit_source_key_sdf and (not args.free_plug_in_socket or args.physics_device == "cpu" or args.probe_additional_turn_deg is not None):
    parser.error("the source-key SDF audit is a post-measurement read on the static GPU/free-connector diagnostic")
if args.replay_source_drive_targets and (args.probe_additional_turn_deg is not None or args.source_step is None):
    parser.error("the bounded source-command replay requires a declared source step and no controller probe")
if args.mounted_grasp_recipe and (args.free_plug_in_socket or args.probe_additional_turn_deg is not None
                                  or not args.robot_state_before_reset):
    parser.error("the mounted grasp recipe uses the explicit Nut fixture and pre-reset robot initialization")
args.output.mkdir(parents=True,exist_ok=False)
(args.output/'launch.json').write_text(json.dumps({
    'argv':[sys.executable,*sys.argv],
    'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2)+'\n')
(args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
if finger_mechanism_document is not None:
    (args.output/'finger_mechanism_contract.json').write_text(json.dumps(finger_mechanism_document,indent=2)+'\n')
    (args.output/'finger_mechanism_candidate.json').write_text(json.dumps({
        'scope':'MECHANISM_CANDIDATE_NOT_PRODUCTION_ASSEMBLY_ACCEPTANCE',
        'mechanism_id':finger_mechanism_document['mechanism_id'],
        'contract_path':str(args.finger_mechanism),'representation':'tangent',
        'grasp_geometry_path':str(args.interface_grasp_relation.resolve()),
        'production_acceptance_eligible':False,
        'finger_self_lock_and_complete_assembly_still_require_verification':True,
        'warmup':'INITIAL_EXACT_CLOSURE_AND_FIXED_TANGENT_DURING_RESET_TWO_TICKS',
        'main_loop':'RELINEARIZE_FROM_ROBOT_ENCODERS_BEFORE_EVERY_PHYSICS_STEP'},indent=2)+'\n')
    for relative in ('isaac/te_hand_fourbar.py','isaac/te_worm_drive.py','isaac/carts_v2/controller.py',
                     'kcg_connector/grasp/robust/finger_fourbar.py'):
        source=Path(__file__).resolve().parents[1]/relative
        (args.output/('candidate_'+source.name)).write_bytes(source.read_bytes())
for module_name in ('te_local_interface_following.py','te_interface_stroke_schedule.py'):
    module_path=Path(__file__).with_name(module_name)
    (args.output/module_name).write_bytes(module_path.read_bytes())
sys.path.insert(0,str(Path(__file__).with_name("carts_v2")))
if source_stage_recipe is not None and source_stage_recipe.get('cuda_allocation_trace_library'):
    # Diagnostic-only: CUPTI supports one subscriber, so acquire it before Kit's
    # optional GPU profiler. This observer never changes allocation parameters.
    import ctypes
    allocation_observer=ctypes.CDLL(str(Path(source_stage_recipe['cuda_allocation_trace_library']).resolve()))
    allocation_observer.start_allocation_trace.argtypes=[ctypes.c_char_p]
    status=allocation_observer.start_allocation_trace(str(args.output/'cuda_allocations.jsonl').encode())
    if status!=0:raise RuntimeError(f'CUPTI allocation observer unavailable: {status}')
from isaacsim import SimulationApp
app = SimulationApp({"headless":args.probe_additional_turn_deg is None,"multi_gpu":False,
                     "active_gpu":0,"physics_gpu":0,"fast_shutdown":True,"shutdown_watchdog_timeout":10.,
                     "extra_args":((["--/rtx/hydra/supportMultiTickRate=false"]
                                    if args.probe_additional_turn_deg is not None else [])
                                   +(["--/physics/fabricUseGPUInterop=true"] if args.fabric_gpu_interop else [])
                                   +(["--/app/profilerBackend=cpu","--/profiler/enabled=true",
                                      "--/profiler/gpu=false","--/app/profilerMask=0"]
                                     if source_stage_recipe and source_stage_recipe.get('native_cpu_timeline',False) else []))})
failed=False
requested_exit_code=0
previous_physics_dispatch_settings=None
try:
    import carb
    import numpy as np
    import omni.usd
    from scipy.spatial.transform import Rotation
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.experimental.utils.backend import use_backend
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    import controller
    fourbar_couplings={};fourbar_authoring=None;fourbar_initial_targets=None
    shared_hand_runtime=None;shared_hand_setup=None
    if args.finger_mechanism is not None:
        from te_hand_fourbar import load_fourbar_contract,author_fourbar_rods,update_fourbar_tangents
        finger_mechanism_document,fourbar_couplings=load_fourbar_contract(args.finger_mechanism)

    repo=Path(__file__).resolve().parents[3]
    local_metadata=json.loads((args.run/"trace_metadata.json").read_text())
    base_run=(args.run if "motion_plan" in local_metadata
              else Path(local_metadata.get("base_run",local_metadata["source_run"])))
    metadata=(local_metadata if base_run==args.run else json.loads((base_run/"trace_metadata.json").read_text()))
    source_rotation_file=args.run/args.source_stage_root/args.source_rotation_stage/"nut_rotation_controller_result.json"
    if source_rotation_file.is_file():
        control_record=json.loads(source_rotation_file.read_text())
    elif (source_stage_recipe and source_stage_recipe.get('use_current_rotation_config')
            and args.source_step is not None):
        # A source may have completed its grip but never started its turn.
        # Its physical state is still explicit; do not invent a turn record.
        control_record=None
    else:
        raise FileNotFoundError(source_rotation_file)
    sample_step=int(control_record["first_step"])-1 if args.source_step is None else args.source_step
    with gzip.open(args.run/args.source_stage_root/args.sensor_stage/"joint_ft_samples.json.gz","rt") as f:
        source_sensor_records=json.load(f)
        sensor_sample=next(s for s in source_sensor_records if s["step"]==sample_step)
    source_nominal_arm_targets=list(sensor_sample['active_targets_rad'][:7])
    replay_records=([s for s in source_sensor_records if sample_step<s["step"]<=sample_step+2*args.physics_hz]
                    if args.replay_source_drive_targets else [])
    del source_sensor_records
    if args.replay_source_drive_targets and (not replay_records or [s["step"] for s in replay_records]!=list(range(sample_step+1,replay_records[-1]["step"]+1))):
        raise ValueError("the sealed source command interval is empty or discontinuous")
    trace=args.run/"truth_samples.jsonl"
    physical_sample=None
    if any((args.run/name).exists() for name in ('truth_samples.jsonl.gz','truth_samples.msgpack.gz')):
        from trace_metadata import read_truth_sample
        physical_sample=read_truth_sample(args.run,sample_step)
    elif trace.exists():
        with trace.open() as f:
            f.seek(max(0,trace.stat().st_size-64000000));f.readline()
            physical_sample=next((json.loads(line) for line in f if f'"step":{sample_step},' in line[:100]),None)
    if physical_sample is None:
        # An earlier completed open-hand boundary need not be in the tail.
        # Scan prefixes only, avoiding decoding unrelated large contact arrays.
        if trace.exists():
            with trace.open() as f:
                physical_sample=next((json.loads(line) for line in f if f'"step":{sample_step},' in line[:100]),None)
    if physical_sample is None:
        raise ValueError(f"the declared source step {sample_step} is absent from the sealed trace")
    lab_recipe=None;lab_fixture_pose=None
    interface_final_hand_targets=None
    if args.interface_start_open:
        import copy,yaml
        from kcg_connector.robot_model import expand_active_hand_positions
        interface_final_hand_targets=np.asarray(sensor_sample['active_targets_rad'][7:],float).copy()
        interface_source_hand_targets=interface_final_hand_targets.copy()
        # Active hand coordinates are shared preshape plus f1j2/f2j1/f3j2.
        # Source closure directions are all positive; retain the preshape.
        interface_final_hand_targets[1:]+=np.deg2rad(args.interface_extra_closure_deg)
        recipe=yaml.safe_load((repo/'src/kcg_connector/config/te_transport_grasp_relation_q70_4p5_v1.yaml').read_text())
        hand=np.asarray(recipe['hand_contract']['pregrasp_joint_positions_rad'],float)
        hand[0]=sensor_sample['active_positions_rad'][7]
        if interface_grip_recipe is not None or fourbar_couplings:
            geometry=json.loads(args.interface_grasp_relation.read_text())
            hand=np.asarray(geometry['open_hand_positions_rad'],float)
            interface_final_hand_targets=np.asarray(geometry['first_contact_hand_positions_rad'],float)
            interface_final_hand_targets[1:]+=.002
        sensor_sample=copy.deepcopy(sensor_sample);physical_sample=copy.deepcopy(physical_sample)
        sensor_sample['active_positions_rad'][7:]=hand.tolist();sensor_sample['active_targets_rad'][7:]=hand.tolist()
        for name,value in expand_active_hand_positions(hand).items():
            physical_sample['arm_control']['hand_joint_diagnostic']['joints'][name]['position_rad']=float(value)
        (args.output/'interface_open_start.json').write_text(json.dumps({'open_positions_rad':hand.tolist(),
            'final_source_drive_targets_rad':interface_source_hand_targets.tolist(),
            'final_commanded_drive_targets_rad':interface_final_hand_targets.tolist(),
            'extra_closure_deg':args.interface_extra_closure_deg,
            'finite_closure_duration_s':.8,'connector_and_robot_arm_initial_pose_unchanged':True})+'\n')
    if args.probe_open_grasp_recipe is not None:
        import copy
        from kcg_connector.robot_model import expand_active_hand_positions
        opening=json.loads(args.probe_open_grasp_recipe.read_text())
        hand=np.asarray(opening['open_hand_positions_rad'],float)
        sensor_sample=copy.deepcopy(sensor_sample);physical_sample=copy.deepcopy(physical_sample)
        sensor_sample['active_positions_rad'][7:]=hand.tolist()
        sensor_sample['active_targets_rad'][7:]=hand.tolist()
        for name,value in expand_active_hand_positions(hand).items():
            physical_sample['arm_control']['hand_joint_diagnostic']['joints'][name]['position_rad']=float(value)
        (args.output/'declared_open_hand_start.json').write_text(json.dumps({
            'recipe':str(args.probe_open_grasp_recipe),'hand_positions_rad':hand.tolist(),
            'scope':'PRE_PHYSICS_INITIAL_CONDITION_FOR_FRESH_GRASP_NO_POST_START_OBJECT_POSE_WRITE'},indent=2)+'\n')
    if args.mounted_grasp_recipe:
        import copy
        from kcg_connector.robot_model import expand_active_hand_positions
        lab_recipe=json.loads(args.mounted_grasp_recipe.read_text())
        if args.standard_render_steps:lab_recipe["standard_render_steps"]=True
        H=np.eye(4);H[:3,:3]=np.asarray(sensor_sample["handbase_rotation_world_row_major"]).reshape(3,3)
        H[:3,3]=sensor_sample["handbase_position_world_m"]
        lab_fixture_pose=H@np.linalg.inv(np.asarray(lab_recipe["canonical_body_from_hand_for_nut_grasp"]))
        sensor_sample=copy.deepcopy(sensor_sample);physical_sample=copy.deepcopy(physical_sample)
        hand=np.asarray(lab_recipe["open_hand_positions_rad"],float)
        sensor_sample["active_positions_rad"][7:]=hand.tolist()
        sensor_sample["active_targets_rad"][7:]=hand.tolist()
        for name,value in expand_active_hand_positions(hand).items():
            physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"][name]["position_rad"]=float(value)
    dt=1/float(args.physics_hz)
    half_second=round(.5/dt)
    if source_stage_recipe and (source_stage_recipe.get('serial_physx_dispatcher',False)
                               or 'physics_worker_threads' in source_stage_recipe):
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SETTING_NUM_THREADS,SETTING_PHYSX_DISPATCHER
        if args.physics_device!='cpu':raise ValueError('the local dispatcher configuration requires CPU physics')
        workers=int(source_stage_recipe.get('physics_worker_threads',0))
        if not 0<=workers<=28:raise ValueError('bounded CPU simulation worker count required')
        settings_store=carb.settings.get_settings()
        previous_physics_dispatch_settings={SETTING_NUM_THREADS:settings_store.get(SETTING_NUM_THREADS),
            SETTING_PHYSX_DISPATCHER:settings_store.get(SETTING_PHYSX_DISPATCHER)}
        if source_stage_recipe.get('serial_physx_dispatcher',False):settings_store.set_bool(SETTING_PHYSX_DISPATCHER,True)
        settings_store.set_int(SETTING_NUM_THREADS,workers)
        get_physx_interface().set_thread_count(workers)
        (args.output/'physics_dispatcher_comparison.json').write_text(json.dumps({
            'previous':previous_physics_dispatch_settings,
            'actual':{k:settings_store.get(k) for k in previous_physics_dispatch_settings},
            'restore_before_application_close':True,'physics_model_timestep_iterations_unchanged':True},indent=2)+'\n')
    SimulationManager.set_physics_sim_device(args.physics_device)
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=dt if args.standard_render_steps else 1/60,
                backend="numpy",device=args.physics_device,
                sim_params={"use_gpu_pipeline":args.physics_device!="cpu","use_fabric":args.cpu_fabric_output})
    if args.gpu_host_readback:
        # SimulationManager.set_device('cuda') has already enabled GPU
        # dynamics. Change only suppression of host output, before reset;
        # do not call set_device('cpu'), which would disable GPU dynamics.
        carb.settings.get_settings().set_bool("/physics/suppressReadback",False)
    stage=omni.usd.get_context().get_stage()
    scene=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    if args.physics_device=='cpu':
        scene.CreateEnableGPUDynamicsAttr(False);scene.CreateBroadphaseTypeAttr('MBP')
    scene.CreateSolverTypeAttr(args.solver_type);scene.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))
    scene.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    scene.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    robot_root=UsdGeom.Xform.Define(stage,"/World/HandArm")
    robot_asset_path=args.robot_asset.resolve() if args.robot_asset else Path(metadata['robot_asset'])
    robot_asset_path=robot_asset_path.resolve()
    if not robot_asset_path.is_file():raise FileNotFoundError(robot_asset_path)
    production_robot_asset=(repo/'artifacts/kcg_connector/isaac/te_nail_tip_body_grasp_v1/handarm_original_nails_source_decomposition.usda').resolve()
    if args.hand_role=='production-nails' and robot_asset_path!=production_robot_asset:
        raise ValueError('Production requires the verified original nail-present asset; declare nail-free only as comparison')
    if args.interface_grasp_relation is not None:
        selected_geometry=json.loads(args.interface_grasp_relation.read_text())
        if args.hand_role=='production-nails' and selected_geometry.get('hand_variant')!='LEGACY_NAIL_PRESENT':
            raise ValueError('Production grasp geometry must be checked with the nail-present source meshes')
    robot_root.GetPrim().GetReferences().AddReference(str(robot_asset_path))
    (args.output/'robot_asset_selection.json').write_text(json.dumps({'selected_asset':str(robot_asset_path),
        'source_record_asset':metadata['robot_asset'],'explicit_asset_override':args.robot_asset is not None,
        'hand_role':args.hand_role,'production_acceptance_eligible':args.hand_role=='production-nails' and not fourbar_couplings,
        'finger_mechanism_id':None if not fourbar_couplings else finger_mechanism_document['mechanism_id'],
        'authored_before_physics':True})+'\n')
    if args.mimic_natural_frequency is not None:
        mimic_rows=[]
        for follower,leader in controller.MIMIC_HAND_JOINTS.items():
            if follower in fourbar_couplings:continue
            if args.palm_layout_mechanism=='self-lock' and follower=='f3j1':continue
            prim=stage.GetPrimAtPath('/World/HandArm/Physics/'+follower)
            old_reference=prim.GetRelationship('newton:mimicJoint').GetTargets()
            coefficient=float(prim.GetAttribute('newton:mimicCoef1').Get())
            offset=float(prim.GetAttribute('newton:mimicCoef0').Get())
            if coefficient!=1. or offset!=0. or len(old_reference)!=1 or old_reference[0].name!=leader:
                raise ValueError('Unexpected source mimic relation; refusing to replace it')
            prim.RemoveAPI('NewtonMimicAPI')
            prim.GetAttribute('newton:mimicEnabled').Set(False)
            mimic=PhysxSchema.PhysxMimicJointAPI.Apply(prim,UsdPhysics.Tokens.rotX)
            mimic.CreateReferenceJointRel().SetTargets(old_reference)
            mimic.CreateReferenceJointAxisAttr(UsdPhysics.Tokens.rotX)
            mimic.CreateGearingAttr(-coefficient);mimic.CreateOffsetAttr(-offset)
            mimic.CreateNaturalFrequencyAttr(args.mimic_natural_frequency)
            mimic.CreateDampingRatioAttr(2.)
            mimic_rows.append({'follower':follower,'leader':leader,'nominal_multiplier':coefficient,
                'nominal_offset':offset,'natural_frequency':float(mimic.GetNaturalFrequencyAttr().Get()),
                'damping_ratio':float(mimic.GetDampingRatioAttr().Get())})
        (args.output/'hand_mimic_compliance_authoring.json').write_text(json.dumps({
            'scope':'FINITE_TRANSMISSION_COMPLIANCE_REFERENCE_NOT_HARDWARE_CALIBRATION',
            'geometry_mass_joint_limits_and_nominal_coupling_unchanged':True,
            'physx_single_dof_joint_axis_is_implicit':True,'joints':mimic_rows},indent=2)+'\n')
    if args.independent_robot_rigid_frames:
        from te_hand_rigid_transform_scene import author_independent_robot_rigid_frames
        frame_report=author_independent_robot_rigid_frames(stage,"/World/HandArm",
            physics_started=bool(world.is_playing() or world.current_time>0.))
        (args.output/"robot_rigid_frame_authoring.json").write_text(json.dumps(frame_report,indent=2)+"\n")
    hand_friction_report = None
    if args.hand_friction_effort_from_urdf:
        from te_hand_joint_friction import author_urdf_hand_friction_efforts
        hand_friction_report = author_urdf_hand_friction_efforts(repo,stage,"/World/HandArm")
        (args.output/"hand_joint_friction_authoring.json").write_text(json.dumps(hand_friction_report,indent=2)+"\n")
    worm_friction_authoring=[]
    if args.finger_worm_self_lock:
        from omni.physx.bindings._physx import (JOINT_AXIS_API,JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
            JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)
        for name in dict.fromkeys(n for follower,coupling in fourbar_couplings.items() for n in (coupling.source_joint,follower)):
            joint=stage.GetPrimAtPath('/World/HandArm/Physics/'+name)
            api=PhysxSchema.PhysxJointAPI.Apply(joint)
            before=api.GetJointFrictionAttr().Get();api.CreateJointFrictionAttr(0.)
            joint.ApplyAPI(JOINT_AXIS_API,UsdPhysics.Tokens.angular)
            for attr in (JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
                         JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR):
                joint.CreateAttribute(attr,Sdf.ValueTypeNames.Float).Set(0.)
            worm_friction_authoring.append({'joint':name,'previous_legacy_load_coefficient':before,
                'preserved_source_urdf_friction':joint.GetAttribute('urdf:dynamics:friction').Get(),
                'additional_native_friction':0.,'replacement':'PASSIVE_SPLIT_MOTOR_FRICTION_AND_OUTPUT_DAMPING'})
        (args.output/'worm_friction_authoring.json').write_text(json.dumps(worm_friction_authoring,indent=2)+'\n')
    native_velocity_bounds=None
    if args.native_model_velocity_limits:
        from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
        from kcg_connector.grasp.robust.collision_roster import load_authoritative_collision_link_roster
        from kcg_connector.grasp.carts_v2.models import _build_verified_robot_model
        velocity_model=_build_verified_robot_model(
            load_carts_hand_contract('src/kcg_connector/config/carts_hand_contact_v1.yaml',repository_root=repo),
            load_authoritative_collision_link_roster('src/kcg_connector/config/carts_collision_roster_v1.yaml',repository_root=repo),
            finger_mechanism_path=args.finger_mechanism)
        native_velocity_bounds={}
        for prim in stage.Traverse():
            name=prim.GetName()
            if str(prim.GetPath()).startswith('/World/HandArm/') and name in velocity_model.joints and prim.IsA(UsdPhysics.RevoluteJoint):
                speed=float(velocity_model.joints[name].limit.velocity)
                if not math.isfinite(speed) or speed<=0:raise ValueError('Invalid declared joint velocity reference')
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(float(np.degrees(speed)))
                native_velocity_bounds[name]=speed
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            a=PhysxSchema.PhysxArticulationAPI(prim);a.CreateSolverPositionIterationCountAttr(args.position_iterations)
            a.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            if args.wrist_reference_loads:a.CreateSleepThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            body_api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body_api.CreateSolverPositionIterationCountAttr(args.position_iterations)
            body_api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            if args.wrist_reference_loads:body_api.CreateSleepThresholdAttr(0.)
    material=UsdShade.Material.Define(stage,"/World/FixtureMaterial")
    m=UsdPhysics.MaterialAPI.Apply(material.GetPrim());m.CreateStaticFrictionAttr(.45)
    m.CreateDynamicFrictionAttr(.45);m.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("min")
    fingertip=stage.GetPrimAtPath("/World/HandArm/PhysicsMaterials/fingertip_pad")
    source_pad_parameters={name:fingertip.GetAttribute(name).Get() for name in
        ('physics:staticFriction','physics:dynamicFriction','physxMaterial:frictionCombineMode')}
    fm=UsdPhysics.MaterialAPI(fingertip);fm.CreateStaticFrictionAttr(.45);fm.CreateDynamicFrictionAttr(.45)
    PhysxSchema.PhysxMaterialAPI.Apply(fingertip).CreateFrictionCombineModeAttr("min")

    if args.free_plug_in_socket:
        import yaml
        from isaacsim.core.utils.stage import add_reference_to_stage
        from run_grasp_lift import prepare_dynamic_scene, _apply_contact_friction_perturbation
        from te_body_assembly_scene import prepare_body_assembly_scene
        from te_grounding_band_scene import install_grounding_band_contact_model
        launch=None
        if not source_stage_recipe:
            launch_path=base_run.with_suffix('.launch.json')
            if not launch_path.is_file():launch_path=base_run.parent/(base_run.name+'_command.json')
            launch_document=json.loads(launch_path.read_text())
            launch=launch_document['argv'] if isinstance(launch_document,dict) else launch_document
        base_config=(repo/source_stage_recipe['base_config'] if source_stage_recipe
                     else repo/launch[launch.index("--config")+1])
        config=yaml.safe_load(base_config.read_text())
        assembly_path=(repo/source_stage_recipe['assembly_config'] if source_stage_recipe else
                       args.run/"local_assembly_control.yaml" if (args.run/"local_assembly_control.yaml").exists()
                       else Path(local_metadata["body_assembly_control_config"])
                       if "body_assembly_control_config" in local_metadata
                       else repo/launch[launch.index("--body-assembly-collision-config")+1])
        if args.assembly_config is not None:
            assembly_path=args.assembly_config.resolve()
        entry=dict(config["dynamic"]["object_scenes"][metadata["object_id"]])
        def saved_pose(i):
            T=np.eye(4);q=np.array(physical_sample["object_part_orientations_wxyz"][i])
            T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            T[:3,3]=physical_sample["object_part_positions_m"][i]
            return T
        body_T,nut_T=saved_pose(0),saved_pose(1)
        entry["frozen_settled_world_from_object_row_major"]=body_T.ravel().tolist()
        assembly=prepare_dynamic_scene(repo,stage,entry,add_reference_to_stage,
                                       metadata["robustness_perturbation"])
        nut_xform=UsdGeom.Xformable(stage.GetPrimAtPath(assembly["part_prim_paths"][1]))
        nut_xform.ClearXformOpOrder()
        nut_xform.AddTransformOp().Set(Gf.Matrix4d(*(np.linalg.inv(body_T)@nut_T).T.ravel().tolist()))
        prepared=prepare_body_assembly_scene(repo,stage,assembly,assembly_path)
        assembly=prepared["scene"]
        if args.diagnostic_center_socket_before_start:
            socket_prim=stage.GetPrimAtPath('/World/TEVisualHandoff/FixedReceptaclePose')
            socket_xform=UsdGeom.Xformable(socket_prim)
            before=np.asarray(socket_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.copy()
            axis=before[:3,2]/np.linalg.norm(before[:3,2])
            shift=(np.eye(3)-np.outer(axis,axis))@(body_T[:3,3]-before[:3,3])
            if np.linalg.norm(shift)>.001:
                raise ValueError('diagnostic fixture-centering offset exceeds 1 mm')
            after=before.copy();after[:3,3]+=shift
            parent=np.asarray(UsdGeom.Xformable(socket_prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            local=np.linalg.inv(parent)@after
            socket_xform.ClearXformOpOrder()
            socket_xform.AddTransformOp(opSuffix='DeclaredColdCentering').Set(Gf.Matrix4d(*local.T.ravel().tolist()))
            centering={'scope':'PRE_PHYSICS_FIXTURE_PLACEMENT_COUNTERFACTUAL_NOT_ROBOT_ALIGNMENT',
                'socket_position_before_world_m':before[:3,3].tolist(),
                'socket_position_after_world_m':after[:3,3].tolist(),
                'socket_shift_world_m':shift.tolist(),
                'robot_and_body_nut_initial_poses_changed':False,
                'geometry_materials_mass_inertia_changed':False,'post_start_pose_write':False}
            prepared['report']['diagnostic_initial_socket_centering']=centering
            prepared['report']['socket_initial_position_world_m']=after[:3,3].tolist()
            (args.output/'diagnostic_initial_socket_centering.json').write_text(json.dumps(centering,indent=2)+'\n')
        passive_joint=stage.GetPrimAtPath(prepared["report"]["passive_joint_solver"]["internal_joint_path"])
        source_joint_audit=physical_sample.get("object_internal_joint_audit")
        if source_joint_audit is not None:
            source_passive_angle=float(source_joint_audit["positions_rad"][0])
        elif args.frozen_connector_model is not None:
            # The delivered maximal joint has no articulation coordinate
            # reader. Its declared cold initial angle follows the recorded
            # native Body/Nut poses and original joint frames, modulo 2 pi.
            j=UsdPhysics.RevoluteJoint(passive_joint)
            def local_rotation(q):
                return Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
            relative=(local_rotation(j.GetLocalRot0Attr().Get()).T @ body_T[:3,:3].T
                      @ nut_T[:3,:3] @ local_rotation(j.GetLocalRot1Attr().Get()))
            q=Rotation.from_matrix(relative).as_quat()
            axis_index=("X","Y","Z").index(j.GetAxisAttr().Get())
            source_passive_angle=float(2*np.arctan2(q[axis_index],q[3]))
            (args.output/'passive_angle_from_saved_native_poses.json').write_text(json.dumps({
                'source_step':sample_step,'angle_rad_modulo_two_pi':source_passive_angle,
                'source':'DECLARED_COLD_NATIVE_BODY_NUT_POSES_AND_ORIGINAL_JOINT_FRAMES',
                'used_after_physics_start':False})+'\n')
        else:
            raise ValueError('source has no passive-joint coordinate or declared maximal-joint model')
        initial_joint_state=PhysxSchema.JointStateAPI.Apply(passive_joint,UsdPhysics.Tokens.angular)
        initial_joint_state.CreatePositionAttr(float(np.degrees(source_passive_angle)))
        authored_passive_angle = float(np.radians(initial_joint_state.GetPositionAttr().Get()))
        authored_phase_error = float(np.arctan2(np.sin(authored_passive_angle-source_passive_angle),
                                                np.cos(authored_passive_angle-source_passive_angle)))
        if abs(authored_phase_error) > 2e-6:
            raise RuntimeError("the authored passive-joint angle does not represent the declared source value")
        # This is a static local diagnostic. Reported TGS velocity is not a
        # reliable derivative of the saved pose and is deliberately not copied.
        initial_joint_state.CreateVelocityAttr(0.)
        _apply_contact_friction_perturbation(stage,assembly,Usd,UsdPhysics,UsdShade,PhysxSchema)
        band=yaml.safe_load(assembly_path.read_text())["grounding_band_contact_model"]
        body_path,fixture_path=assembly["part_prim_paths"]
        other=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
               and str(p.GetPath())!=body_path]
        prepared["report"]["grounding_band_contact_model"]=install_grounding_band_contact_model(
            stage,body_path,repo/band["geometry_manifest"],
            stiffness_n_m=band["native_per_contact_stiffness_n_m"],damping_ns_m=band["native_per_contact_damping_ns_m"],
            socket_collision_path=prepared["receptacle_collision_path"],
            non_socket_contact_paths=[*other,assembly["roots"]["table"],assembly["roots"]["fixture"]])
        mating=yaml.safe_load(assembly_path.read_text()).get("mating_contact_model")
        if mating and mating.get("enabled"):
            from te_mating_contact_scene import install_socket_contact_interior
            install_socket_contact_interior(repo,stage,prepared,mating["manifest"])
        if args.frozen_connector_model is not None:
            import importlib.util
            installer=args.frozen_connector_model.resolve().parent/'install_model.py'
            spec=importlib.util.spec_from_file_location('validated_connector_installer',installer)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            frozen_install=module.install_model(stage,model_path=args.frozen_connector_model,prepared=prepared)
            (args.output/'frozen_model_installation.json').write_text(json.dumps(frozen_install,indent=2)+'\n')
            if args.connector_initial_pose is not None:
                observed=json.loads(args.connector_initial_pose.read_text())
                bench_parts=[]
                for p,q in zip(observed['positions_world_m'],observed['quaternions_wxyz']):
                    T=np.eye(4);T[:3,3]=p;T[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();bench_parts.append(T)
                if args.interface_grasp_relation is not None:
                    grasp=json.loads(args.interface_grasp_relation.read_text())
                    from te_three_finger_wrench_observer import source_grasp_contact_allowed
                    if not grasp.get('terminal_original_surface_geometry') or any(
                            not source_grasp_contact_allowed(grasp,c) for c in grasp['first_contacts']):
                        raise ValueError('The declared relation must use verified original fingertip surfaces')
                    hand_T=np.eye(4)
                    hand_T[:3,:3]=np.asarray(sensor_sample['handbase_rotation_world_row_major']).reshape(3,3)
                    hand_T[:3,3]=sensor_sample['handbase_position_world_m']
                    nut_T=hand_T@np.linalg.inv(np.asarray(grasp['canonical_body_from_hand_for_nut_grasp']))
                placement=nut_T@np.linalg.inv(bench_parts[1])
                frozen_stage=Usd.Stage.Open(str(args.frozen_connector_model.resolve()))
                socket_root='/World/TEVisualHandoff/FixedReceptaclePose'
                bench_socket=np.asarray(UsdGeom.Xformable(frozen_stage.GetPrimAtPath(socket_root)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                def cold_world_pose(path,T):
                    prim=stage.GetPrimAtPath(path);parent=np.asarray(UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.AddTransformOp().Set(Gf.Matrix4d(*(np.linalg.inv(parent)@T).T.ravel().tolist()))
                body_T,nut_T=(placement@T for T in bench_parts)
                cold_world_pose(body_path,body_T);cold_world_pose(fixture_path,nut_T);cold_world_pose(socket_root,placement@bench_socket)
                for path in (body_path,fixture_path):
                    rb=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path));rb.CreateVelocityAttr(Gf.Vec3f(0.));rb.CreateAngularVelocityAttr(Gf.Vec3f(0.))
                (args.output/'declared_connector_initialization.json').write_text(json.dumps({
                    'source_pose':str(args.connector_initial_pose.resolve()),
                    'saved_hand_and_nut_grasp_relation_preserved':args.interface_grasp_relation is None,
                    'declared_cad_grasp_relation':str(args.interface_grasp_relation) if args.interface_grasp_relation else None,
                    'world_from_benchmark':placement.tolist(),'body_world':body_T.tolist(),'nut_world':nut_T.tolist(),
                    'socket_world':(placement@bench_socket).tolist(),'all_parts_and_fixture_relocated_before_physics':True,
                    'post_start_object_pose_writes':False,'socket_placement_is_test_initial_condition_not_robot_alignment':True},indent=2)+'\n')
        (args.output/"assembly_scene.json").write_text(json.dumps(prepared["report"],indent=2)+"\n")
        fixture_sensor_path=body_path
        contact_filters=list(dict.fromkeys([body_path,fixture_path,prepared["receptacle_collision_path"],
            *prepared["contact_recording"]["additional_required_contact_filter_paths"]]))
    else:
        fixture_path="/World/DeclaredNutFixture"
        fixture=UsdGeom.Xform.Define(stage,fixture_path)
        position=physical_sample["object_part_positions_m"][1];quat=physical_sample["object_part_orientations_wxyz"][1]
        if lab_fixture_pose is not None:
            position=lab_fixture_pose[:3,3].tolist()
            q=Rotation.from_matrix(lab_fixture_pose[:3,:3]).as_quat();quat=q[[3,0,1,2]].tolist()
        fixture.AddTranslateOp().Set(Gf.Vec3d(*position));fixture.AddOrientOp().Set(Gf.Quatf(quat[0],Gf.Vec3f(*quat[1:])))
        UsdPhysics.RigidBodyAPI.Apply(fixture.GetPrim())
        mass_stage=Usd.Stage.Open(str(repo/"artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"))
        original_mass=mass_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/CouplingNut")
        UsdPhysics.MassAPI.Apply(fixture.GetPrim())
        for name in ("physics:mass","physics:centerOfMass","physics:diagonalInertia","physics:principalAxes"):
            a=original_mass.GetAttribute(name);fixture.GetPrim().CreateAttribute(name,a.GetTypeName()).Set(a.Get())
        mesh=UsdGeom.Mesh.Define(stage,fixture_path+"/SourceThreadedNut")
        geometry=repo/"artifacts/kcg_connector/isaac/te_full_assembly_20260905/representative_thread_geometry_clean_01/coupling_nut_standard_inner_thread.npz"
        data=np.load(geometry);v=data["vertices_m"].astype(np.float32);f=data["faces"].astype(np.int32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f),3,np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()));mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
        sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim());sdf.CreateSdfResolutionAttr(1024)
        sdf.CreateSdfSubgridResolutionAttr(6);sdf.CreateSdfNarrowBandThicknessAttr(.002);sdf.CreateSdfTriangleCountReductionFactorAttr(1.)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh.GetPrim()).CreateWeldToleranceAttr(0.)
        col=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());col.CreateContactOffsetAttr(.00005);col.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose="physics")
        if lab_recipe is not None and lab_recipe.get("render_original_nut_visual",False):
            from build_te_free_split_plug import NUT_VISUAL
            # The capped thread SDF remains the same collision-only diagnostic
            # representation. Display the original supplier-derived visual
            # layer instead of exposing its hidden collision closure faces.
            UsdGeom.Imageable(mesh.GetPrim()).GetVisibilityAttr().Set("invisible")
            visual=UsdGeom.Xform.Define(stage,fixture_path+"/OriginalNutCadVisual")
            visual.GetPrim().GetReferences().AddReference(str(NUT_VISUAL))
            if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                   for p in Usd.PrimRange(visual.GetPrim())):
                raise RuntimeError("the selected original Nut visual contains physics APIs")
            if not UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get():
                raise RuntimeError("display selection disabled the required Nut collider")
            (args.output/"mounted_nut_visual_authoring.json").write_text(json.dumps({
                "original_visual_source":str(NUT_VISUAL),"collision_mesh_path":str(mesh.GetPath()),
                "collision_enabled":True,"collision_geometry_or_material_changed":False,
                "visual_source_contains_physics":False,"source_mass_and_inertia_changed":False,
                "authored_before_physics":True,"not_a_formal_connector_model_delivery":True},indent=2)+"\n")
        clamp=UsdPhysics.FixedJoint.Define(stage,"/World/DeclaredFixtureMount")
        clamp.CreateBody1Rel().SetTargets([fixture_path]);clamp.CreateLocalPos0Attr(Gf.Vec3f(*position))
        clamp.CreateLocalRot0Attr(Gf.Quatf(quat[0],Gf.Vec3f(*quat[1:])))
        UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
        fixture_sensor_path=fixture_path
        contact_filters=[fixture_path]
    if args.interface_source_pad_material:
        for name,value in source_pad_parameters.items():fingertip.GetAttribute(name).Set(value)
    if args.interface_twist_deg is not None:
        (args.output/'pad_material_mode.json').write_text(json.dumps({
            'mode':'UNVALIDATED_LEGACY_PAD_SENSITIVITY' if args.interface_source_pad_material else 'DECLARED_UNCALIBRATED_DEVELOPMENT_VALUE',
            'source_asset_parameters':source_pad_parameters,
            'actual_parameters':{name:fingertip.GetAttribute(name).Get() for name in source_pad_parameters},
            'hardware_calibrated':False,'real_pad_material_applicability_verified':False,
            'formal_material_acceptance':False,'connector_material_changed':False},indent=2)+'\n')
    if args.nut_fingertip_sdf:
        from te_nut_fingertip_sdf import author_nut_only_fingertip_sdf
        fingertip_sdf_report=author_nut_only_fingertip_sdf(stage,"/World/HandArm",fixture_path,
            links=tuple(args.nut_fingertip_sdf_link or ('f1Link3','f2Link2','f3Link3')))
        (args.output/"nut_fingertip_sdf_authoring.json").write_text(json.dumps(fingertip_sdf_report,indent=2)+"\n")
    hand_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                and ("/handbase_link/" in str(p.GetPath()) or str(p.GetPath()).endswith("/handbase_link"))]
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI(prim).CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    if source_stage_recipe and source_stage_recipe.get('balanced_cpu_iteration_budget',False):
        if not balanced_iterations:raise ValueError('the declared balanced iteration comparison must be CPU240Hz/255/16')
        scene.CreateMinPositionIterationCountAttr(args.position_iterations)
        scene.CreateMaxPositionIterationCountAttr(args.position_iterations)
        counts={'rigid_bodies':0,'articulations':0}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api=PhysxSchema.PhysxRigidBodyAPI(prim);counts['rigid_bodies']+=1
                api.CreateSolverPositionIterationCountAttr(args.position_iterations)
                api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                api=PhysxSchema.PhysxArticulationAPI(prim);counts['articulations']+=1
                api.CreateSolverPositionIterationCountAttr(args.position_iterations)
                api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        (args.output/'balanced_iteration_comparison.json').write_text(json.dumps({
            'baseline_hz_position_velocity':[960,64,4],'actual_hz_position_velocity':[240,255,16],
            'baseline_position_iterations_per_second':61440,'actual_position_iterations_per_second':61200,
            'velocity_iterations_per_second':3840,'configured_actor_counts':counts,
            'geometry_material_inertia_effort_boundaries_changed':False,
            'accuracy_requires_current_physical_review':True},indent=2)+'\n')
    contact_capacity=(max(4096,int(prepared["contact_recording"].get("minimum_contact_records",4096)))
                      if args.free_plug_in_socket else 4096)
    contact_filter_arguments={} if args.raw_contact_only else {'contact_filter_paths':contact_filters}
    (args.output/'contact_recording_layout.json').write_text(json.dumps({
        'raw_unfiltered_records_only':args.raw_contact_only,'sensor_count':len(hand_paths),
        'legacy_filter_count':len(contact_filters),'record_capacity':contact_capacity,
        'collision_filters_and_contact_materials_changed':False},indent=2)+'\n')
    efficient_probe_views=bool(source_stage_recipe and (
        source_stage_recipe.get('omit_unused_contact_collectors',False) or args.contact_audit_mode=='native-report'))
    contacts=RigidPrim(hand_paths,resolve_paths=False,
        **({} if efficient_probe_views else {**contact_filter_arguments,'max_contact_count':contact_capacity}))
    probe_contact_paths=([str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                          and str(p.GetPath()).startswith("/World/HandArm/")] + [body_path,fixture_path]
                         if args.probe_additional_turn_deg is not None or args.source_stage_probe else [])
    probe_contacts=(RigidPrim(probe_contact_paths,resolve_paths=False,
                    **({} if efficient_probe_views else {**contact_filter_arguments,'max_contact_count':contact_capacity}))
                    if probe_contact_paths else None)
    (args.output/'contact_collector_usage.json').write_text(json.dumps({
        'tensor_contact_collectors_enabled':not efficient_probe_views,
        'native_contact_report_apis_retained':True,'all_native_points_required':True,
        'pose_views_retained':True,'joint_and_wrist_force_sensors_unchanged':True,
        'legacy_filter_count':len(contact_filters)},indent=2)+'\n')
    static_part_contact_recording = args.free_plug_in_socket and args.probe_additional_turn_deg is None and not efficient_probe_views
    parts=RigidPrim([body_path,fixture_path] if args.free_plug_in_socket else [fixture_path],resolve_paths=False,
        **({**contact_filter_arguments,'max_contact_count':contact_capacity}
           if static_part_contact_recording else {}))
    tree=(None if args.robot_state_before_reset else world.scene.add(
        SingleArticulation("/World/HandArm/Geometry/world",name="wrist",reset_xform_properties=False)))
    fixture_tree = (world.scene.add(SingleArticulation(fixture_sensor_path,name="fixture_sensor",reset_xform_properties=False))
                    if args.main_read_sequence and not args.free_plug_in_socket else None)
    if scene.GetSolverTypeAttr().Get() != args.solver_type:
        raise RuntimeError("scene preparation changed the declared solver type")
    robot_initialization_report=None
    if args.robot_state_before_reset:
        # World.reset advances physics before tensor setters become available.
        # Supply the saved physical hand support during those ticks as well.
        # No connector pose or friction/collision parameter is changed here.
        source_positions={f"iiwa_joint_{i+1}":v for i,v in enumerate(sensor_sample["active_positions_rad"][:7])}
        source_positions.update({k:v["position_rad"] for k,v in physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"].items()})
        active_names=controller.ARM_JOINT_NAMES+controller.ACTIVE_HAND_JOINT_NAMES
        source_targets=dict(zip(active_names,np.r_[sensor_sample["arm_control"]["drive_target_rad"],sensor_sample["active_targets_rad"][7:]]))
        fourbar_initialization=None
        if fourbar_couplings:
            finger_joint_names={name for follower,coupling in fourbar_couplings.items()
                                for name in (follower,coupling.source_joint)}
            fourbar_joint_limits={}
            for name in finger_joint_names:
                joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/HandArm/Physics/'+name))
                fourbar_joint_limits[name]=(math.radians(float(joint.GetLowerLimitAttr().Get())),
                                           math.radians(float(joint.GetUpperLimitAttr().Get())))
            original_positions=dict(source_positions)
            source_positions,position_rows=initialize_fourbar_pose(source_positions,fourbar_couplings,
                fourbar_joint_limits,float(finger_mechanism_document['geometry_uncertainty_m']))
            full_targets={**source_positions,**source_targets}
            fourbar_initial_targets,target_rows=initialize_fourbar_pose(full_targets,fourbar_couplings,
                fourbar_joint_limits,float(finger_mechanism_document['geometry_uncertainty_m']))
            source_targets={name:fourbar_initial_targets[name] for name in source_targets}
            for index,name in enumerate(active_names):
                sensor_sample['active_positions_rad'][index]=source_positions[name]
                sensor_sample['active_targets_rad'][index]=fourbar_initial_targets[name]
            for name,row in physical_sample['arm_control']['hand_joint_diagnostic']['joints'].items():
                row['position_rad']=source_positions[name]
            fourbar_initialization={'scope':'NEW_MECHANISM_PRE_RESET_INITIAL_CONDITION_NOT_HISTORICAL_STATE_REPLAY',
                'mechanism_id':finger_mechanism_document['mechanism_id'],
                'original_source_record_positions_rad':original_positions,
                'position_resolution':position_rows,'target_resolution':target_rows,
                'geometry_alignment_uncertainty_m':finger_mechanism_document['geometry_uncertainty_m'],
                'native_original_joint_limits_rad':fourbar_joint_limits,
                'after_reset_physical_position_or_velocity_writes':False}
        source_arm_damping=(float(physical_sample["arm_control"]["effective_arm_damping_nm_s_rad"])
                            if args.arm_damping is None else args.arm_damping)
        authored=[]
        for prim in stage.Traverse():
            name=prim.GetName()
            if not str(prim.GetPath()).startswith("/World/HandArm/") or name not in source_positions or not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            state=PhysxSchema.JointStateAPI.Apply(prim,UsdPhysics.Tokens.angular)
            state.CreatePositionAttr(float(np.degrees(source_positions[name])))
            state.CreateVelocityAttr(0.)
            drive=UsdPhysics.DriveAPI.Apply(prim,"angular")
            is_active=name in source_targets;is_arm=name in controller.ARM_JOINT_NAMES
            kp=(args.arm_stiffness_nm_rad if is_arm else args.hand_stiffness_nm_rad) if is_active else 0.
            kd=(source_arm_damping if is_arm else args.hand_damping_nm_s_rad) if is_active else 0.
            cap=(100. if is_arm else args.closing_drive_cap_nm if name in ('f1j2','f2j1','f3j2') else 1.) if is_active else 0.
            if args.palm_layout_mechanism=='self-lock' and name in ('f1j1','f3j1'):
                kp=kd=cap=0.
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(np.radians(kp)))
            drive.CreateDampingAttr(float(np.radians(kd)))
            drive.CreateMaxForceAttr(cap)
            target=(fourbar_initial_targets[name] if fourbar_initial_targets is not None else
                    source_targets.get(name,source_positions[name]))
            drive.CreateTargetPositionAttr(float(np.degrees(target)))
            drive.CreateTargetVelocityAttr(0.)
            authored.append(name)
        if set(authored)!=set(source_positions):
            raise RuntimeError("source robot state did not map to every original joint")
        robot_initialization_report={"authored_before_world_reset":True,"source_positions_rad":source_positions,
            "source_applied_drive_targets_rad":source_targets,"all_fifteen_joints_authored":True,
            "angular_drive_usd_conversion":"Nm/rad to Nm/degree by pi/180; finite caps retained",
            "legacy_robot_reset_writer_attached":False,"object_pose_modified":False,
            "finger_mechanism_initialization":fourbar_initialization}
        if fourbar_couplings:
            fourbar_authoring=author_fourbar_rods(stage,'/World/HandArm/Physics',args.finger_mechanism,
                                                 representation='tangent')
            fourbar_authoring.update(
                scope='MECHANISM_CANDIDATE_NOT_PRODUCTION_ASSEMBLY_ACCEPTANCE',
                reset_warmup_uses_initial_tangent_without_per_tick_relinearization=True,
                main_loop_relinearizes_from_robot_encoders_before_every_step=True)
            (args.output/'finger_fourbar_authoring.json').write_text(json.dumps(fourbar_authoring,indent=2)+'\n')
        (args.output/"robot_initialization_before_reset.json").write_text(json.dumps(robot_initialization_report,indent=2)+"\n")
        if args.palm_layout_mechanism=='self-lock':
            # The user confirmed the adjustable palm is mechanically locked by
            # its worm drive. Lock only the two hand layout joints at setup;
            # preserve every finger-closing DOF and all connector freedoms.
            palm_rows=[]
            for name in ('f1j1','f3j1'):
                joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/HandArm/Physics/'+name))
                prim=joint.GetPrim();angle_deg=float(np.degrees(source_positions[name]))
                before=[float(joint.GetLowerLimitAttr().Get()),float(joint.GetUpperLimitAttr().Get())]
                if not before[0]<=angle_deg<=before[1]:raise ValueError('Palm lock angle outside source mechanism range')
                if prim.HasAPI('NewtonMimicAPI'):
                    prim.RemoveAPI('NewtonMimicAPI');prim.GetAttribute('newton:mimicEnabled').Set(False)
                if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI,UsdPhysics.Tokens.rotX):
                    prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,UsdPhysics.Tokens.rotX)
                joint.CreateLowerLimitAttr(angle_deg);joint.CreateUpperLimitAttr(angle_deg)
                palm_rows.append({'joint':name,'source_limits_deg':before,'locked_angle_deg':angle_deg,
                    'limits_deg':[joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()],
                    'body0':[str(x) for x in joint.GetBody0Rel().GetTargets()],
                    'body1':[str(x) for x in joint.GetBody1Rel().GetTargets()]})
            if abs(palm_rows[0]['locked_angle_deg']-palm_rows[1]['locked_angle_deg'])>1e-4:
                raise ValueError('Source palm layout leader and follower disagree before locking')
            (args.output/'palm_self_lock_authoring.json').write_text(json.dumps({
                'basis':'USER_CONFIRMED_WORM_DRIVE_MECHANICAL_SELF_LOCK_20260912',
                'representation':'ZERO_WIDTH_REVOLUTE_LIMITS_AT_SELECTED_LAYOUT_NO_ACTIVE_SERVO',
                'elasticity_backlash_and_failure_load_unmeasured':True,
                'only_hand_layout_joints_changed':True,'authored_before_physics':True,'joints':palm_rows},indent=2)+'\n')
    if args.sdf_resolution_override is not None:
        sdf_rows=[]
        for prim in stage.Traverse():
            attr=prim.GetAttribute('physxSDFMeshCollision:sdfResolution')
            if prim.HasAPI(UsdPhysics.CollisionAPI) and attr and attr.Get():
                before=int(attr.Get());attr.Set(args.sdf_resolution_override)
                pts=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                sdf_rows.append({'path':str(prim.GetPath()),'before':before,'after':int(attr.Get()),
                    'maximum_grid_spacing_m':float(np.ptp(pts,axis=0).max()/args.sdf_resolution_override)})
        (args.output/'sdf_resolution_comparison.json').write_text(json.dumps({
            'scope':'UNVALIDATED_COLLISION_DISCRETIZATION_COMPARISON','source_CAD_vertices_unchanged':True,
            'rows':sdf_rows},indent=2)+'\n')
    if args.shared_hand_mechanism:
        from te_hand_mechanism_runtime import author_hand_mechanism
        shared_hand_setup=author_hand_mechanism(stage,repo,args.shared_hand_mechanism.resolve(),
            '/World/HandArm/Physics',source_positions)
        if source_stage_recipe and source_stage_recipe.get('motor_input_state'):
            for name,state in source_stage_recipe['motor_input_state'].items():
                drive=UsdPhysics.DriveAPI(stage.GetPrimAtPath('/World/HandArm/Physics/'+name),'angular')
                drive.CreateTargetPositionAttr(float(np.degrees(state['input_angle'])))
    if args.experimental_connector_position_convergence:
        # Apply after all model/hand authoring, before the first physical step.
        # The frozen USD and its validated64-iteration contract stay unchanged.
        before={'scene_min':scene.GetMinPositionIterationCountAttr().Get(),
                'scene_max':scene.GetMaxPositionIterationCountAttr().Get()}
        scene.CreateMinPositionIterationCountAttr(args.position_iterations)
        scene.CreateMaxPositionIterationCountAttr(args.position_iterations)
        actors=[]
        for prim in stage.Traverse():
            for schema in (PhysxSchema.PhysxRigidBodyAPI,PhysxSchema.PhysxArticulationAPI):
                if prim.HasAPI(schema):
                    api=schema(prim)
                    api.CreateSolverPositionIterationCountAttr(args.position_iterations)
                    api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
                    actors.append({'path':str(prim.GetPath()),'schema':schema.__name__,
                                   'position':api.GetSolverPositionIterationCountAttr().Get(),
                                   'velocity':api.GetSolverVelocityIterationCountAttr().Get()})
        if not actors or any(a['position']!=args.position_iterations or a['velocity']!=4 for a in actors):
            raise RuntimeError('Explicit numerical iteration override was not authored consistently')
        (args.output/'position_iteration_comparison.json').write_text(json.dumps({
            'scope':'PRE_RESET_NUMERICAL_CONVERGENCE_COMPARISON_NOT_TRANSFERRED_MODEL_ACCEPTANCE',
            'baseline_hz_position_velocity':[960,64,4],
            'actual_hz_position_velocity':[args.physics_hz,args.position_iterations,args.velocity_iterations],
            'scene_before':before,'scene_after':{'min':scene.GetMinPositionIterationCountAttr().Get(),
                'max':scene.GetMaxPositionIterationCountAttr().Get()},'actors':actors,
            'geometry_material_mass_inertia_effort_limits_changed':False,
            'source_model_file_changed':False,'original_key_review_tolerance_um':2.,
            'requires_new_physical_review':True},indent=2)+'\n')
    contact_order_audit = None
    if source_stage_recipe is not None and 'solve_articulation_contact_last' in source_stage_recipe:
        requested_order = source_stage_recipe['solve_articulation_contact_last']
        previous_order = scene.GetSolveArticulationContactLastAttr().Get()
        scene.CreateSolveArticulationContactLastAttr(requested_order)
        if scene.GetSolveArticulationContactLastAttr().Get() != requested_order:
            raise RuntimeError('The explicit articulation contact solve order was not authored')
        contact_order_audit = {
            'scope': 'LOCAL_PRE_RESET_CONTACT_SOLVER_ORDER_COMPARISON',
            'scene_path': str(scene.GetPrim().GetPath()), 'before': previous_order,
            'requested': requested_order, 'usd_readback_before_reset': scene.GetSolveArticulationContactLastAttr().Get(),
            'requested_solver_hz_position_velocity': [args.solver_type, args.physics_hz, args.position_iterations, args.velocity_iterations],
            'scene_position_iteration_min_max': [scene.GetMinPositionIterationCountAttr().Get(), scene.GetMaxPositionIterationCountAttr().Get()],
            'scene_velocity_iteration_min_max': [scene.GetMinVelocityIterationCountAttr().Get(), scene.GetMaxVelocityIterationCountAttr().Get()],
            'geometry_material_mass_inertia_effort_limits_changed': False,
            'source_model_file_changed': False, 'key_review_tolerance_um_unchanged': 2.,
            'validation_transfer_from_original_solver_claimed': False,
            'documentation': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order',
        }
    if source_stage_recipe is not None and 'nut_external_sdf_bits' in source_stage_recipe:
        if (args.physics_device,args.physics_hz,args.position_iterations,args.velocity_iterations)!=('cpu',960,64,4):
            raise ValueError('The external SDF storage comparison retainsCPU960Hz64/4')
        from nut_sdf_storage import configure_nut_external_sdf_bits
        report=configure_nut_external_sdf_bits(stage,source_stage_recipe['nut_external_sdf_bits'])
        (args.output/'nut_sdf_storage.json').write_text(json.dumps(report,indent=2)+'\n')
    if source_stage_recipe is not None and 'pin_contact_offset_m' in source_stage_recipe:
        declared_iterations=(args.position_iterations==64 or (
            args.experimental_connector_position_convergence and args.position_iterations in (16,32)))
        if ((args.physics_device,args.physics_hz,args.velocity_iterations)!=('cpu',960,4)
                or not declared_iterations
                or source_stage_recipe.get('fuse_convex_pin_quarters',False)):
            raise ValueError('The source pin margin comparison requiresCPU960Hz, explicit iteration comparison and original four-quarter geometry')
        from pin_contact_margin import configure_pin_contact_margin
        report=configure_pin_contact_margin(stage,source_stage_recipe['pin_contact_offset_m'])
        report['actual_hz_position_velocity']=[args.physics_hz,args.position_iterations,args.velocity_iterations]
        (args.output/'pin_contact_margin.json').write_text(json.dumps(report,indent=2)+'\n')
    if source_stage_recipe is not None and source_stage_recipe.get('fuse_convex_pin_quarters',False):
        if args.physics_device!='cpu':raise ValueError('The exact180vertex pin hull is a CPU-only candidate')
        from fuse_pin_collision_quarters import fuse_pin_quarters, fuse_pin_shafts, inspect_cooked_pin_hulls
        mode=source_stage_recipe.get('pin_fusion_mode','whole')
        if mode not in ('whole','axial_shaft'):raise ValueError('Unknown explicit pin decomposition')
        report=(fuse_pin_shafts(stage) if mode=='axial_shaft' else fuse_pin_quarters(stage))
        (args.output/'pin_collision_fusion.json').write_text(json.dumps(report,indent=2)+'\n')
        cooked=inspect_cooked_pin_hulls(stage,report)
        (args.output/'pin_cooked_geometry.json').write_text(json.dumps(cooked,indent=2)+'\n')
        if max(cooked['maximum_source_outside_cooked_plane_m'],
               cooked['maximum_cooked_outside_source_plane_m'])>1e-8:
            raise RuntimeError('Fused pin native cooking changed its source envelope by more than10nm')
    if source_stage_recipe is not None and source_stage_recipe.get('omit_disabled_collision_apis',False):
        from runtime_collision_pruning import omit_disabled_collision_apis
        report=omit_disabled_collision_apis(stage, deactivate_invisible_leaves=bool(
            source_stage_recipe.get('deactivate_disabled_invisible_collision_leaves',False)))
        (args.output/'disabled_collision_api_omission.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.physics_device!='cpu':
        buffers={a.GetName():a.Get() for a in scene.GetPrim().GetAttributes()
                 if 'gpu' in a.GetName().lower()}
        (args.output/'gpu_scene_buffers_before_reset.json').write_text(json.dumps(buffers,indent=2)+'\n')
    if source_stage_recipe is not None and source_stage_recipe.get('compile_leaf_collision_pairs',False):
        from runtime_collision_groups import compile_leaf_collision_pairs
        report=compile_leaf_collision_pairs(stage, include_actor_pairs=bool(
            source_stage_recipe.get('compile_actor_collision_pairs',False)),
            ownership_parser_workdir=args.output/'collision_filter_ownership')
        (args.output/'collision_pair_compilation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'begin_physics_reset','wall_seconds':time.monotonic()-diagnostic_started,
                      'device':args.physics_device}),flush=True)
    world.reset();world.pause()
    if contact_order_audit is not None:
        contact_order_audit['usd_readback_after_reset'] = scene.GetSolveArticulationContactLastAttr().Get()
        if contact_order_audit['usd_readback_after_reset'] != contact_order_audit['requested']:
            raise RuntimeError('Articulation contact solve order changed during reset')
        contact_order_audit['actor_solver_iterations'] = []
        for prim in stage.Traverse():
            for schema in (PhysxSchema.PhysxRigidBodyAPI, PhysxSchema.PhysxArticulationAPI):
                if prim.HasAPI(schema):
                    api = schema(prim)
                    contact_order_audit['actor_solver_iterations'].append({
                        'path': str(prim.GetPath()), 'schema': schema.__name__,
                        'position': api.GetSolverPositionIterationCountAttr().Get(),
                        'velocity': api.GetSolverVelocityIterationCountAttr().Get(),
                    })
        (args.output/'contact_order_comparison.json').write_text(json.dumps(contact_order_audit, indent=2)+'\n')
    print(json.dumps({'stage':'physics_reset_complete','wall_seconds':time.monotonic()-diagnostic_started}),flush=True)
    if args.physics_device=='cpu' and args.frozen_connector_model is not None:
        import carb.logging
        carb.logging.acquire_logging().set_level_threshold_for_source(
            'omni.physx.plugin',carb.logging.LogSettingBehavior.OVERRIDE,carb.logging.LEVEL_ERROR)
        (args.output/'logging_scope.json').write_text(json.dumps({
            'source':'omni.physx.plugin','after_reset_level':'ERROR',
            'reason':'Previously recorded repeated CPU SDF material-face report warnings; error logging retained',
            'physics_or_contact_processing_changed':False})+'\n')
    if args.robot_state_before_reset:
        tree=SingleArticulation("/World/HandArm/Geometry/world",name="wrist",reset_xform_properties=False)
        tree.initialize()
        native_initial=tree.get_joint_positions()
        native_initial=(native_initial.detach().cpu().numpy() if hasattr(native_initial,"detach") else np.asarray(native_initial)).reshape(-1)
        robot_initialization_report["native_positions_after_warmup_rad"]=dict(zip(tree.dof_names,map(float,native_initial)))
        robot_initialization_report["reset_physics_time_s"]=float(world.current_time)
        (args.output/"robot_initialization_before_reset.json").write_text(json.dumps(robot_initialization_report,indent=2)+"\n")
    passive_initialization=None
    if args.free_plug_in_socket and args.frozen_connector_model is None:
        # A legacy reader present during World.reset resets joint positions to
        # its cached drive targets (zero for the passive joint). Construct this
        # read-only observer after the initial reset to retain the USD state.
        fixture_tree=SingleArticulation(fixture_sensor_path,name="fixture_sensor",reset_xform_properties=False)
        fixture_tree.initialize()
        native_joint=fixture_tree.get_joint_positions()
        if hasattr(native_joint,"detach"):native_joint=native_joint.detach().cpu().numpy()
        else:native_joint=np.asarray(native_joint)
        observed_passive_angle=float(native_joint.reshape(-1)[0])
        phase_error=float(np.arctan2(np.sin(observed_passive_angle-source_passive_angle),
                                     np.cos(observed_passive_angle-source_passive_angle)))
        passive_initialization={"source_angle_rad":source_passive_angle,
                                "authored_before_world_reset":True,
                                "authored_angle_rad":authored_passive_angle,
                                "authored_phase_error_rad":authored_phase_error,
                                "native_angle_after_reset_rad":observed_passive_angle,
                                "read_only_observer_constructed_after_reset":True,
                                "phase_error_modulo_full_revolution_rad":phase_error,
                                "initial_velocity_rad_s":0.,"contact_warm_start_restored":False,
                                "scope":"LOCAL_STATIC_INITIAL_CONDITION_NOT_EXACT_FULL_EPISODE_REPLAY"}
        # The exact source-state authoring was checked before simulation above.
        # After reset's two physical ticks, a free joint may legitimately move.
        # In static diagnosis check the current joint against current body poses;
        # do not misclassify physical settling as a failed zero-time write.
        native_phase_bound=float(np.deg2rad(.01))
        passive_initialization.update(reset_physics_time_s=float(world.current_time),
            native_phase_check_bound_rad=native_phase_bound,
            native_phase_check_role="BOUNDED_INITIAL_SETTLING_AFTER_RESET_NOT_ZERO_TIME_IDENTITY")
        if args.probe_additional_turn_deg is None:
            with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
                current_p,current_q=parts.get_world_poses()
            current_rot=Rotation.from_quat(current_q.numpy()[:,[1,2,3,0]]).as_matrix()
            joint=UsdPhysics.RevoluteJoint(passive_joint)
            local_rot=joint.GetLocalRot0Attr().Get()
            joint_frame=Rotation.from_quat([*local_rot.GetImaginary(),local_rot.GetReal()]).as_matrix()
            axis=joint_frame@np.eye(3)[:,("X","Y","Z").index(joint.GetAxisAttr().Get())]
            source_relative=body_T[:3,:3].T@nut_T[:3,:3]
            current_relative=current_rot[0].T@current_rot[1]
            predicted_relative=Rotation.from_rotvec(axis*phase_error).as_matrix()@source_relative
            pose_error=float(Rotation.from_matrix(current_relative@predicted_relative.T).magnitude())
            zero_reset=bool(abs(np.arctan2(np.sin(observed_passive_angle),np.cos(observed_passive_angle)))<2e-6 and abs(np.arctan2(np.sin(source_passive_angle),np.cos(source_passive_angle)))>native_phase_bound)
            passive_initialization.update(
                native_phase_check_role="POST_WARMUP_JOINT_AND_NATIVE_POSE_CONSISTENCY_FOR_STATIC_DIAGNOSIS",
                old_phase_drift_reference_exceeded=abs(phase_error)>native_phase_bound,
                source_to_current_motion_is_recorded_not_an_assembly_pass=True,
                joint_pose_rotation_consistency_error_rad=pose_error,
                current_relative_rotation=current_relative.tolist(),
                source_relative_rotation=source_relative.tolist(),
                expected_body_frame_joint_axis=axis.tolist(),
                joint_pose_consistency_bound_rad=5e-5,
                unexpected_zero_reset_detected=zero_reset)
            if pose_error>5e-5 or zero_reset or abs(float(world.current_time)-2*dt)>1e-6:
                (args.output/"passive_initialization_readback.json").write_text(json.dumps(passive_initialization,indent=2)+"\n")
                raise RuntimeError("static initialization has inconsistent joint/pose state, a zero reset, or unexpected warmup time")
        (args.output/"passive_initialization_readback.json").write_text(json.dumps(passive_initialization,indent=2)+"\n")
        if args.probe_additional_turn_deg is not None and abs(phase_error)>native_phase_bound:
            raise RuntimeError(f"local passive-joint initial phase differs: source={source_passive_angle}, native={observed_passive_angle}")
    elif args.free_plug_in_socket:
        fixture_tree=None
        passive_initialization={'representation':stage.GetPrimAtPath(prepared['report']['passive_joint_solver']['internal_joint_path']).GetTypeName(),
            'frozen_model':str(args.frozen_connector_model.resolve()),'object_articulation_reader_used':False,
            'source_pose_is_declared_cold_initial_state_not_restored_contact_history':True}
    settings={"arm_control_law":"NATIVE_FORCE_DRIVE_GRAVITY_EQUIVALENT_POSITION_BIAS_V1",
              "arm_stiffness":args.arm_stiffness_nm_rad,"arm_damping":60.,"hand_stiffness":args.hand_stiffness_nm_rad,"hand_damping":args.hand_damping_nm_s_rad,
              "arm_drive_maximum_effort_nm":100.,"hand_drive_maximum_effort_nm":1.}
    settings['closing_drive_maximum_effort_nm']=args.closing_drive_cap_nm
    settings["arm_damping"] = (float(physical_sample["arm_control"]["effective_arm_damping_nm_s_rad"])
                               if args.arm_damping is None else args.arm_damping)
    names=metadata["controller_outcome"]["native_drive_audit"]["dof_names"]
    robot_data=controller.create_native_gravity_compensated_robot(
        "/World/HandArm/Geometry/world",names,settings,
        initial_arm_positions=sensor_sample["active_positions_rad"][:7],
        initial_hand_positions=sensor_sample["active_positions_rad"][7:],
        preserve_authored_state=bool(fourbar_couplings),initial_named_positions=fourbar_initial_targets)
    robot,active,*_=robot_data
    if shared_hand_setup is not None:
        from te_hand_mechanism_runtime import HandMechanismRuntime
        shared_hand_runtime=HandMechanismRuntime(world,robot,shared_hand_setup,args.output,
            active_effort_caps=[1.,args.closing_drive_cap_nm,args.closing_drive_cap_nm,args.closing_drive_cap_nm])
        if source_stage_recipe and source_stage_recipe.get('motor_input_state'):
            for name,state in source_stage_recipe['motor_input_state'].items():
                shared_hand_runtime.drives[name].input_angle=float(state['input_angle'])
                shared_hand_runtime.drives[name].input_velocity=float(state['input_velocity'])
            (args.output/'initial_internal_motor_state.json').write_text(json.dumps({
                'scope':'DECLARED_DIAGNOSTIC_INITIAL_CONTROLLER_STATE_ONLY',
                'source_step':args.source_step,'state':source_stage_recipe['motor_input_state'],
                'physical_joint_pose_written_after_reset':False},indent=2)+'\n')
    palm_lock_native_angle=None
    if args.palm_layout_mechanism=='self-lock':
        lock_indices=[robot.dof_names.index(n) for n in ('f1j1','f3j1')]
        zeros=np.zeros((1,2),dtype=np.float32)
        robot.set_dof_gains(zeros,zeros,indices=0,dof_indices=lock_indices)
        robot.set_dof_max_efforts(zeros,indices=0,dof_indices=lock_indices)
        lock_lo,lock_hi=robot.get_dof_limits(indices=0,dof_indices=lock_indices)
        locked=lock_lo.numpy()[0]
        if not np.allclose(locked,lock_hi.numpy()[0],rtol=0,atol=1e-7):
            raise RuntimeError('Native palm joint limits do not implement the authored mechanical lock')
        palm_lock_native_angle=float(locked[0])
        lock_kp,lock_kd=robot.get_dof_gains(indices=0,dof_indices=lock_indices)
        (args.output/'palm_self_lock_native_readback.json').write_text(json.dumps({
            'joint_names':['f1j1','f3j1'],'lower_rad':locked.tolist(),'upper_rad':lock_hi.numpy()[0].tolist(),
            'stiffness':lock_kp.numpy()[0].tolist(),'damping':lock_kd.numpy()[0].tolist(),
            'drive_caps_nm':robot.get_dof_max_efforts(indices=0,dof_indices=lock_indices).numpy()[0].tolist(),
            'active_servo_disabled':True,'loaded_motion_still_requires_measurement':True},indent=2)+'\n')
    if native_velocity_bounds is not None:
        before=robot.get_dof_max_velocities(indices=0).numpy()[0].copy()
        specified=np.asarray([native_velocity_bounds[n] for n in robot.dof_names])
        robot.set_dof_max_velocities(specified[None,:],indices=0)
        after=robot.get_dof_max_velocities(indices=0).numpy()[0].copy()
        if not np.allclose(after,specified,rtol=1e-6,atol=1e-7):raise RuntimeError('Native joint velocity readback differs from the model references')
        (args.output/'native_velocity_limit_readback.json').write_text(json.dumps({
            'joint_names':list(robot.dof_names),'before_rad_s':before.tolist(),'after_rad_s':after.tolist(),
            'scope':'DECLARED_MODEL_VELOCITY_REFERENCE_NOT_NEW_HARDWARE_CALIBRATION',
            'source':'Existing verified kinematic model; hand4rad/s remains an uncalibrated reference',
            'motor_force_caps_and_position_limits_unchanged':True},indent=2)+'\n')
    if args.interface_twist_deg is not None:
        native_kp,native_kd=robot.get_dof_gains(indices=0,dof_indices=active)
        native_caps=robot.get_dof_max_efforts(indices=0,dof_indices=active)
        (args.output/'active_drive_readback.json').write_text(json.dumps({
            'stiffness_nm_rad':native_kp.numpy()[0].tolist(),'damping_nm_s_rad':native_kd.numpy()[0].tolist(),
            'effort_caps_nm':native_caps.numpy()[0].tolist(),'force_drives_not_kinematic_constraints':True},indent=2)+'\n')
    if args.gpu_host_readback:
        readback={"gpu_dynamics_enabled":bool(scene.GetEnableGPUDynamicsAttr().Get()),
            "broadphase":str(scene.GetBroadphaseTypeAttr().Get()),
            "suppress_readback":carb.settings.get_settings().get("/physics/suppressReadback"),
            "data_device":str(SimulationManager.get_physics_sim_device()),
            "solver":args.solver_type,"physics_dt_s":dt,
            "scope":"GPU_DYNAMICS_WITH_HOST_RESULT_READBACK_NOT_CPU_SOLVER_MIGRATION"}
        (args.output/"physics_readback_mode.json").write_text(json.dumps(readback,indent=2)+"\n")
        if not readback["gpu_dynamics_enabled"] or readback["broadphase"]!="GPU" or readback["suppress_readback"]:
            raise RuntimeError("the requested data-path check did not retain GPU dynamics with host readback")
    if hand_friction_report is not None:
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            native_friction = [value.numpy()[0] for value in robot.get_dof_friction_properties(indices=0)]
        readback=[]
        for row in hand_friction_report["joints"]:
            index=robot.dof_names.index(row["joint"])
            values=[float(value[index]) for value in native_friction]
            if not np.allclose(values,[row["source_urdf_friction_nm"],row["source_urdf_friction_nm"],0.],rtol=1e-6,atol=1e-9):
                raise RuntimeError("native hand joint-friction tensor differs from the authored effort model")
            readback.append({"joint":row["joint"],"static_nm":values[0],"dynamic_nm":values[1],"viscous_nm_s_rad":values[2]})
        hand_friction_report["native_tensor_readback_after_reset"]=readback
        (args.output/"hand_joint_friction_authoring.json").write_text(json.dumps(hand_friction_report,indent=2)+"\n")
    actual={f"iiwa_joint_{i+1}":x for i,x in enumerate(sensor_sample["active_positions_rad"][:7])}
    actual.update({k:x["position_rad"] for k,x in physical_sample["arm_control"]["hand_joint_diagnostic"]["joints"].items()})
    if not fourbar_couplings:
        robot.set_dof_positions(np.array([[actual[n] for n in robot.dof_names]]))
    targets=np.r_[sensor_sample["arm_control"]["drive_target_rad"],sensor_sample["active_targets_rad"][7:]]
    if palm_lock_native_angle is not None:targets[7]=palm_lock_native_angle
    if interface_grip_recipe and interface_grip_recipe.get('recompute_initial_arm_bias'):
        targets[:7],bias_readback=controller.gravity_biased_arm_target(robot,robot_data[2],
            sensor_sample['active_positions_rad'][:7],robot_data[3],robot_data[4],settings)
        (args.output/'initial_native_gravity_compensation.json').write_text(json.dumps({
            'scope':'NEW_DECLARED_ARM_POSTURE_NATIVE_GRAVITY_BIAS_BEFORE_MEASUREMENT',
            'connector_pose_changed':False,'readback':bias_readback},indent=2)+'\n')
    robot.set_dof_position_targets(targets[None,:],indices=0,dof_indices=active)
    replay_targets=np.asarray([np.r_[s["arm_control"]["drive_target_rad"],s["active_targets_rad"][7:]] for s in replay_records])
    if replay_records:
        lo,hi=robot.get_dof_limits(indices=0,dof_indices=active)
        if not np.isfinite(replay_targets).all() or np.any(replay_targets<lo.numpy()[0]) or np.any(replay_targets>hi.numpy()[0]):
            raise ValueError("a recorded replay target is outside the existing joint limits")
    hb=next(i for i,p in enumerate(hand_paths) if p.endswith("/handbase_link"))
    row_index=tree._articulation_view._metadata.joint_indices["hand2arm"]+1
    with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
        masses=contacts.get_masses().numpy().reshape(-1);coms=contacts.get_coms()[0].numpy()
    stage.GetRootLayer().Export(str(args.output/"calibration_after_initialization.usda"))
    if args.source_stage_probe:
        from te_source_stage_probe import run_source_stage_probe
        import copy
        (args.output/'trace_metadata.json').write_text(json.dumps({
            'scope':'LOCAL_SOURCE_DIAGNOSTIC_METADATA_POINTERS_NOT_VISUAL_ASSEMBLY',
            'source_run':str(args.run.resolve()),'base_run':str(base_run.resolve()),
            'object_id':metadata['object_id'],'robot_asset':metadata['robot_asset'],
            'physics_dt_s':1./args.physics_hz},indent=2)+'\n')
        probe_sensor=copy.deepcopy(sensor_sample)
        # The initializer's fourbar target map contains native gravity-biased
        # arm drive targets. A JointSignalStepper needs the nominal references
        # so gravity compensation is not applied a second time.
        probe_sensor['active_targets_rad'][:7]=source_nominal_arm_targets
        result=run_source_stage_probe(repository=repo,args=args,world=world,robot_data=robot_data,
            ft_tree=tree,contact_view=probe_contacts,contact_paths=probe_contact_paths,prepared=prepared,
            metadata=metadata,sensor_sample=probe_sensor,source_rotation=control_record,recipe=source_stage_recipe,
            source_visual_context_run=base_run)
        print(json.dumps(result,indent=2),flush=True)
        passed=(result.get('free_return_completed') if 'free_joint7_return' in result
                else bool(result.get('rotation',{}).get('completed')))
        requested_exit_code=0 if passed and not result.get('error') else 2
        raise SystemExit(requested_exit_code)
    if args.wrist_reference_loads:
        from te_robot_wrist_reference_loads import run_robot_wrist_reference_loads
        result=run_robot_wrist_reference_loads(repository=repo,world=world,robot_data=robot_data,
            ft_tree=tree,contact_view=contacts,hand_paths=hand_paths,recipe=lab_recipe,
            source_sensor=sensor_sample,source_metadata=metadata,base_run=base_run,
            settings=settings,output=args.output)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    if lab_recipe is not None:
        from te_mounted_nut_torque_probe import run_mounted_nut_torque_probe
        result=run_mounted_nut_torque_probe(repository=repo,world=world,robot_data=robot_data,
            ft_tree=tree,contact_view=contacts,hand_paths=hand_paths,recipe=lab_recipe,
            fixture_pose=lab_fixture_pose,source_sensor=sensor_sample,source_metadata=metadata,
            base_run=base_run,settings=settings,output=args.output)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    if args.probe_additional_turn_deg is not None:
        from te_saved_nut_probe import run_probe
        result=run_probe(repository=repo, args=args, world=world, robot_data=robot_data,
                         ft_tree=tree, plug_tree=fixture_tree, contact_view=probe_contacts,
                         contact_paths=probe_contact_paths, prepared_scene=prepared,
                         source_metadata=metadata, source_sensor=sensor_sample,
                         source_rotation=control_record, passive_initialization=passive_initialization,
                         base_run=base_run, assembly_config_path=assembly_path)
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(0)
    def host(v):return v.detach().cpu().numpy() if hasattr(v,"detach") else v.numpy() if hasattr(v,"numpy") else np.asarray(v)
    interface_images=[];interface_movie=None;interface_abort=None;interface_following=None
    if args.interface_twist_deg is not None:
        import cv2
        import omni.replicator.core as rep
        from pxr import UsdLux
        from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
        camera='/World/HandInterfaceEvidenceCamera';focus=nut_T[:3,3]+[0.,0.,.035]
        _author_camera(stage,camera,_camera_cv_pose_from_eye_target(focus+[.15,-.18,.12],focus),
                       resolution=(800,600),focal_length_mm=32.,horizontal_aperture_mm=36.,clipping_range_m=(.01,5.),Gf=Gf,UsdGeom=UsdGeom)
        UsdLux.DomeLight.Define(stage,'/World/HandInterfaceEvidenceLight').CreateIntensityAttr(1000.)
        product=rep.create.render_product(camera,(800,600));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product.path])
        interface_movie=cv2.VideoWriter(str(args.output/'hand_interface.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(800,600))
        # The source URDF has coaxial joint7/link_ee/hand2arm Z axes. Check
        # actual frame alignment before commanding the finite joint stroke.
        _,hq=contacts.get_world_poses();hand_axis=Rotation.from_quat(host(hq)[hb,[1,2,3,0]]).as_matrix()[:,2]
        if not args.interface_force_following and float(hand_axis@nut_T[:3,2]) < .999:
            raise RuntimeError('Saved hand wrist axis does not align with declared Nut axis')
        target_end=targets.copy()
        if not args.interface_force_following:target_end[6]+=np.deg2rad(args.interface_twist_deg)
        if args.interface_start_open:target_end[7:]=interface_final_hand_targets
        if palm_lock_native_angle is not None:target_end[7]=palm_lock_native_angle
        lo,hi=robot.get_dof_limits(indices=0,dof_indices=active)
        if np.any(target_end<host(lo)[0]) or np.any(target_end>host(hi)[0]):
            raise RuntimeError('Interface target exceeds original joint limits')
        if args.interface_force_following:
            from te_local_interface_following import LocalInterfaceFollowing
            interface_following=LocalInterfaceFollowing(repo,
                host(robot.get_dof_positions(indices=0,dof_indices=active))[0],
                nut_T,body_T,placement@bench_socket,dt*args.interface_control_decimation,
                grip_recipe=interface_grip_recipe,finger_mechanism_path=args.finger_mechanism,
                hand_mechanism=shared_hand_runtime,arm_position_stiffness=float(settings['arm_stiffness']))
    rows=[];read_deltas=[];joint_rows=[];part_rows=[];normal_load_rows=[];part_contact_rows=[];render_audit=None
    interface_command_rows=[];projected_joint_reaction_rows=[]
    fourbar_step_rows=[]
    fourbar_follower_order=tuple(fourbar_couplings)
    fourbar_source_indices=[robot.dof_names.index(fourbar_couplings[name].source_joint) for name in fourbar_follower_order]
    fourbar_follower_indices=[robot.dof_names.index(name) for name in fourbar_follower_order]
    worm_drives={};worm_records=[];worm_stream=None
    if args.finger_worm_self_lock:
        from te_worm_drive import WormDrive,WormReference
        reference=WormReference(transmission_stiffness=args.hand_stiffness_nm_rad,transmission_damping=0.,
            output_viscosity=args.hand_damping_nm_s_rad,output_active_effort_reference=args.closing_drive_cap_nm,
            transmission_effort_boundary=args.closing_drive_cap_nm)
        motor_position_kp=(1+reference.load_friction_ratio)*args.hand_stiffness_nm_rad
        motor_position_kd=(1+reference.load_friction_ratio)*args.hand_damping_nm_s_rad
        initial_q=host(robot.get_dof_positions(indices=0))[0]
        for follower,index in zip(fourbar_follower_order,fourbar_source_indices):
            worm_drives[follower]=WormDrive(float(initial_q[index]),reference=reference,integration='passive_split')
        friction=[host(x)[0] for x in robot.get_dof_friction_properties(indices=0)]
        finger_indices=fourbar_source_indices+fourbar_follower_indices
        if any(np.max(np.abs(x[finger_indices]))>1e-12 for x in friction):
            raise RuntimeError('The integrated worm has duplicate native joint friction')
        (args.output/'finger_worm_reference.json').write_text(json.dumps({
            'scope':'WHOLE_ROBOT_INTEGRATION_CANDIDATE_NOT_HARDWARE_IDENTIFICATION','reference':vars(reference),
            'input_effort_cap_nm':reference.input_effort_boundary,
            'motor_position_kp_input_referred':motor_position_kp,'motor_position_kd_input_referred':motor_position_kd,
            'position_reference_is_input_angle_referred_to_output':True,'follower_order':list(fourbar_follower_order),
            'native_friction_readback':[[float(v[i]) for i in finger_indices] for v in friction],
            'position_state_written_after_reset':False,'online_contact_or_object_truth_used':False},indent=2)+'\n')
        worm_stream=(args.output/'finger_worm_samples.jsonl').open('w')
    hand_normal_wrench_rows=[];native_joint_wrench_rows=[]
    hand_link_pose_rows=[];hand_contact_point_records=[]
    contact_component_rows=[];contact_actor_pairs=set();world.play()
    actor_path_cache={}
    performance={'physics_s':0.,'readback_s':0.,'evidence_s':0.}
    def recorded_actor_paths(ids):
        ids=ids.to('cpu');array=ids.numpy();key=(tuple(array.shape),array.tobytes())
        if key not in actor_path_cache:
            actor_path_cache[key]=tuple(map(str,contacts.get_actor_paths_from_ids(ids)))
        return actor_path_cache[key]
    turn_start=3.0 if interface_grip_recipe else 1.6 if args.interface_start_open else .3
    if interface_grip_recipe and interface_grip_recipe.get('validated_grip_helical_following'):
        turn_start=float(interface_grip_recipe.get('preindex_start_s',3.))
        if not 3.<=turn_start<=4.:raise ValueError('the capacity-tested grip uses a bounded3..4second preparation')
    turn_duration=1.875 if interface_grip_recipe else 1.
    interface_duration=5.275 if interface_grip_recipe else 2.9 if args.interface_start_open else 2.
    if args.interface_motion_duration_s is not None:
        turn_duration=args.interface_motion_duration_s
        interface_duration=turn_start+turn_duration+.4
    side_sequence=bool(interface_grip_recipe and interface_grip_recipe.get('sidewall_sequence'))
    if side_sequence:
        interface_duration=float(interface_grip_recipe.get('probe_duration_s',6.9))
        turn_start=float(interface_grip_recipe.get('preindex_start_s',3.))
    seating_alignment_schedule=(interface_grip_recipe or {}).get('seating_alignment_stage')
    if seating_alignment_schedule and args.interface_regrasp_stroke_deg is None:
        if side_sequence:
            raise ValueError('Local seating alignment has one explicit two-part turn')
        first_angle=float(seating_alignment_schedule['first_turn_deg'])
        first_duration=float(seating_alignment_schedule['first_turn_duration_s'])
        align_duration=float(seating_alignment_schedule['alignment_hold_s'])
        final_duration=float(seating_alignment_schedule['final_turn_duration_s'])
        if (not np.isfinite([first_angle,first_duration,align_duration,final_duration]).all()
                or not 0<first_angle<args.interface_twist_deg or min(first_duration,align_duration,final_duration)<=0):
            raise ValueError('Finite positive seating-alignment phases required')
        turn_duration=first_duration+align_duration+final_duration
        interface_duration=turn_start+turn_duration+.4
        (args.output/'seating_alignment_schedule.json').write_text(json.dumps(seating_alignment_schedule,indent=2)+'\n')
    last_following_arm_drive=None
    release_base_command=None;release_start=interface_duration
    release_relax_duration=float((interface_grip_recipe or {}).get('release_relax_duration_s',0.))
    if not math.isfinite(release_relax_duration) or not 0<=release_relax_duration<=.5:
        raise ValueError('Release load-relaxation interval must be finite and at most0.5s')
    if args.interface_release_at_end:
        release_start=interface_duration;interface_duration+=1.3+release_relax_duration
    stroke_schedule=None
    if args.interface_regrasp_stroke_deg is not None:
        from te_interface_stroke_schedule import InterfaceStrokeSchedule
        stroke_schedule=InterfaceStrokeSchedule(args.interface_twist_deg,args.interface_regrasp_stroke_deg,
                                               turn_duration,release=True,
                                               reindex_speed_rad_s=float(interface_grip_recipe.get('reindex_speed_rad_s',.8)),
                                               preload_duration_s=float(interface_grip_recipe.get('regrasp_preload_duration_s',2.)),
                                               grip_duration_s=turn_start,relax_duration_s=release_relax_duration,
                                               opening_duration_s=.8 if interface_grip_recipe.get('friction_aware_release') else .6,
                                               open_hold_duration_s=.5 if interface_grip_recipe.get('friction_aware_release') else 0.,
                                               seating_alignment=seating_alignment_schedule,
                                               post_turn_hold_s=float(interface_grip_recipe.get('post_stroke_loaded_hold_s',.2)))
        interface_duration=stroke_schedule.duration
        release_start=next(p.start for p in stroke_schedule.phases if p.name in ('torque_relaxation','opening'))
        (args.output/'stroke_schedule.json').write_text(json.dumps(stroke_schedule.as_dict(),indent=2)+'\n')
    previous_scheduled_phase=None;regrasp_base_command=None;remembered_grip_goal=None
    release_readiness_records=[];open_moment_samples=[]
    phase_rows=[];final_contact_snapshots=[]
    release_open_target=(np.asarray(json.loads(args.interface_grasp_relation.read_text())['open_hand_positions_rad']) if args.interface_release_at_end else None)
    final_release_open_target=None
    if args.interface_release_at_end:
        extra=float((interface_grip_recipe or {}).get('final_release_extra_opening_rad',0.))
        if not math.isfinite(extra) or not 0<=extra<=.1:
            raise ValueError('Final release opening adjustment must stay within0to0.1rad')
        final_release_open_target=release_open_target.copy();final_release_open_target[1:]-=extra
        lo,hi=robot.get_dof_limits(indices=0,dof_indices=active)
        if np.any(final_release_open_target[1:]<host(lo)[0,8:]) or np.any(final_release_open_target[1:]>host(hi)[0,8:]):
            raise ValueError('Final release target exceeds original finger travel')
        (args.output/'final_release_plan.json').write_text(json.dumps({
            'extra_opening_rad':extra,'final_open_target_rad':final_release_open_target.tolist(),
            'regrasp_open_targets_unchanged':True,'finite_drive_caps_unchanged':True,
            'basis':'SOURCE_CAD_CLEARANCE_AT_OBSERVED_LOADED_POSTURE' if extra else 'ORIGINAL_PREGRASP_POSTURE'},indent=2)+'\n')
    def finite_output_posture(command,opening,*,allow_closing=False):
        """Use the same finite motor inverse for final and intermediate release."""
        if shared_hand_runtime is None or shared_hand_runtime.last_native_state is None:
            raise RuntimeError('Friction-aware posture needs the current finite hand runtime')
        from te_worm_drive import position_reference_for_input_velocity
        measured=shared_hand_runtime.last_native_state[0]
        for hi,name in enumerate(('f1j2','f2j1','f3j2'),start=1):
            output_angle=float(measured[shared_hand_runtime.indices[hi]])
            error=float(opening[hi]-output_angle)
            speed=0. if abs(error)<.0002 else float(np.clip(4.*error,-.15,.15 if allow_closing else 0.))
            low,high=shared_hand_runtime.setup['intervals'][name]
            command[7+hi],_=position_reference_for_input_velocity(
                shared_hand_runtime.drives[name],output_angle,speed,dt,
                shared_hand_runtime.settings['motor_position_kp'],shared_hand_runtime.settings['motor_position_kd'],
                low,high,clip_to_feasible=True)
    for step in range(round(interface_duration/dt)):
        if (interface_grip_recipe and interface_grip_recipe.get('validated_grip_helical_following')
                and args.interface_release_at_end and step*dt>=release_start+1.3+release_relax_duration):
            break
        if step%240==0 and (args.output/'STOP_REQUEST.json').exists():
            requested=json.loads((args.output/'STOP_REQUEST.json').read_text())
            interface_abort='EXTERNAL_EXPERIMENT_STOP: '+str(requested.get('reason','requested for physical review'))
            break
        if args.interface_twist_deg is not None and time.monotonic()-diagnostic_started>args.interface_wall_limit_s:
            interface_abort='DECLARED_WALL_TIME_LIMIT';break
        if args.interface_twist_deg is not None:
            elapsed=step*dt;releasing=args.interface_release_at_end and elapsed>=release_start
            phase=('free_hold' if releasing and elapsed>=release_start+release_relax_duration+.8
                   else 'torque_relaxation' if releasing and elapsed<release_start+release_relax_duration else 'opening' if releasing
                   else 'loaded_hold' if elapsed>=turn_start+turn_duration else 'turn' if elapsed>=turn_start else 'grip')
            phase_rows.append(phase)
            u=np.clip((step*dt-turn_start)/turn_duration,0.,1.);blend=10*u**3-15*u**4+6*u**5
            angular_rate=np.deg2rad(args.interface_twist_deg)*30*u**2*(1-u)**2/turn_duration
            angle_rad=np.deg2rad(args.interface_twist_deg)*blend
            if seating_alignment_schedule and stroke_schedule is None:
                t=elapsed-turn_start
                if t<first_duration:
                    u=np.clip(t/first_duration,0.,1.);b=10*u**3-15*u**4+6*u**5
                    angle_rad=np.deg2rad(first_angle)*b
                    angular_rate=np.deg2rad(first_angle)*30*u**2*(1-u)**2/first_duration
                elif t<first_duration+align_duration:
                    angle_rad=np.deg2rad(first_angle);angular_rate=0.
                    if not releasing:phase='alignment_hold';phase_rows[-1]=phase
                else:
                    u=np.clip((t-first_duration-align_duration)/final_duration,0.,1.);b=10*u**3-15*u**4+6*u**5
                    angle_rad=np.deg2rad(first_angle+(args.interface_twist_deg-first_angle)*b)
                    angular_rate=np.deg2rad(args.interface_twist_deg-first_angle)*30*u**2*(1-u)**2/final_duration
                blend=angle_rad/np.deg2rad(args.interface_twist_deg)
            scheduled_phase=None;grip_clock=None;grip_enabled=True;force_follow_enabled=True
            progress_angle_rad=angle_rad;check_release_now=False
            if args.interface_grip_only:
                force_follow_enabled=False
                if not releasing:
                    phase='loaded_hold' if elapsed>=turn_start else 'grip';phase_rows[-1]=phase
            if stroke_schedule is not None:
                scheduled_phase,angle_rad,angular_rate,phase_blend=stroke_schedule.sample(elapsed)
                progress_angle_rad=stroke_schedule.commanded_progress(scheduled_phase,phase_blend)
                if not releasing:phase=scheduled_phase.name
                phase_rows[-1]=phase
                grip_enabled=phase in ('grip','regrasp_preload','turn','alignment_hold','loaded_hold')
                force_follow_enabled=phase in ('turn','alignment_hold','loaded_hold')
                if phase=='regrasp_preload':
                    grip_clock=(2.5 if interface_grip_recipe.get('reuse_grip_targets') else 1.)+elapsed-scheduled_phase.start
                elif scheduled_phase.stroke>0:grip_clock=turn_start
                if phase!=previous_scheduled_phase:
                    regrasp_base_command=None
                    if phase in ('regrasp_relax','regrasp_open'):
                        remembered_grip_goal=interface_following.hand_goal.copy();open_moment_samples=[]
                    check_release_now=phase=='reindex'
                    if phase=='regrasp_preload':
                        if interface_grip_recipe.get('reuse_grip_targets'):
                            interface_following.hand_goal=remembered_grip_goal.copy()
                        else:
                            interface_following.hand_goal=interface_following.contact_q.copy()
                            interface_following.hand_goal[1:]+=.002
                    previous_scheduled_phase=phase
            if side_sequence:
                v=np.clip(step*dt-turn_start,0.,1.);v2=np.clip(step*dt-5.5,0.,1.)
                second_duration=1.
                if interface_grip_recipe.get('sidewall_measured_balance') or interface_grip_recipe.get('coupled_contact_forces'):
                    second_duration=float(interface_grip_recipe.get('post_index_turn_duration_s',2.5))
                    second_start=float(interface_grip_recipe.get('sidewall_start_s',4.))
                    v2=np.clip((step*dt-second_start)/second_duration,0.,1.)
                blend=.5*(10*v**3-15*v**4+6*v**5+10*v2**3-15*v2**4+6*v2**5)
                angular_rate=np.deg2rad(args.interface_twist_deg)*.5*(30*v**2*(1-v)**2+30*v2**2*(1-v2)**2/second_duration)
                angle_rad=np.deg2rad(args.interface_twist_deg)*blend
            command=targets.copy();command[6]+=np.deg2rad(args.interface_twist_deg)*blend
            if args.interface_start_open:
                g=np.clip((step*dt-.2)/.8,0.,1.);g=10*g**3-15*g**4+6*g**5
                command[7:]=targets[7:]+g*(interface_final_hand_targets-targets[7:])
            capacity_motion=bool(interface_grip_recipe and interface_grip_recipe.get('validated_grip_helical_following'))
            if interface_following is not None and (not releasing or capacity_motion) and step%args.interface_control_decimation==0:
                q_online=host(robot.get_dof_positions(indices=0,dof_indices=active))[0]
                raw_online=host(tree.get_measured_joint_forces())[row_index]
                current_reaction=host(robot.get_dof_projected_joint_forces(indices=0))[0]
                grip_reaction=current_reaction[[robot.dof_names.index(n) for n in ('f1j2','f2j1','f3j2')]]
                lower_now,upper_now=robot.get_dof_limits(indices=0,dof_indices=active)
                lower_now=host(lower_now)[0];upper_now=host(upper_now)[0]
                if (not np.isfinite(q_online).all() or np.any(q_online<lower_now-.02)
                        or np.any(q_online>upper_now+.02)):
                    interface_abort='ACTUAL_ROBOT_JOINT_STATE_OUTSIDE_FINITE_MODEL_LIMITS'
                    (args.output/'abnormal_joint_state.json').write_text(json.dumps({
                        'step':step,'time_s':elapsed,'positions_rad':q_online.tolist(),
                        'lower_rad':lower_now.tolist(),'upper_rad':upper_now.tolist(),
                        'wrist_raw':raw_online.tolist(),'projected_reactions':grip_reaction.tolist()},indent=2)+'\n')
                    break
                try:
                    if phase in ('regrasp_open','regrasp_open_hold') and interface_following.bias is not None:
                        observer=interface_following.grip_observer
                        moments=grip_reaction-observer.tare_reaction-(observer._system(q_online)[2]-observer.tare_gravity)
                        open_moment_samples.append(moments.copy())
                    if check_release_now:
                        from te_nut_motion import released_grip_readiness
                        if not open_moment_samples:raise RuntimeError('NO_MEASURED_RELEASE_HISTORY')
                        moments=np.mean(open_moment_samples[-max(2,round(.1/dt)):],axis=0)
                        clear=bool(np.all(q_online[8:11]<=release_open_target[1:]+.002))
                        ready=released_grip_readiness(clear,moments,.02)
                        ready.update(time_s=elapsed,stroke=scheduled_phase.stroke,
                            output_positions_rad=q_online[8:11].tolist(),open_output_target_rad=release_open_target[1:].tolist(),
                            geometry_basis='ACTUAL_OUTPUT_ENCODERS_AT_SOURCE_OPEN_POSTURE_WITH_0.002_RAD_TOLERANCE; SOURCE_CLEARANCE_REVIEW_AFTER_RUN')
                        release_readiness_records.append(ready)
                        (args.output/'regrasp_release_readiness.json').write_text(json.dumps(release_readiness_records,indent=2)+'\n')
                        if not ready['ready']:raise RuntimeError('REGRASP_OUTPUT_OPENING_OR_UNLOADING_NOT_CONFIRMED')
                    nominal=interface_following.update(q_online,raw_online,step*dt,
                        angle_rad,
                        angular_rate,
                        projected_reactions=grip_reaction,grip_elapsed=grip_clock,
                        grip_enabled=grip_enabled and not releasing,force_follow_enabled=force_follow_enabled and not releasing,
                        arm_speed_override=(float(interface_grip_recipe.get('reindex_speed_rad_s',.8)) if scheduled_phase is not None and phase=='reindex' else None),
                        entry_assist_enabled=not args.interface_grip_only and (scheduled_phase is None or scheduled_phase.stroke==0),
                        hold_pose=capacity_motion and (releasing or phase in
                            ('regrasp_relax','regrasp_open','regrasp_open_hold','regrasp_close','regrasp_preload')),
                        regulate_static_grip=(phase=='regrasp_preload' or bool(seating_alignment_schedule and phase=='alignment_hold'
                            and interface_grip_recipe.get('regulate_grip_while_aligning'))),
                        progress_angle_rad=progress_angle_rad,
                        stroke_index=scheduled_phase.stroke if scheduled_phase else 0,
                        preserve_stroke_pose=stroke_schedule is not None)
                    if nominal is not None:
                        velocity_reference=interface_following.last_joint_velocity if capacity_motion else None
                        last_following_arm_drive,load_audit=controller.gravity_biased_arm_target(robot,robot_data[2],nominal,
                            robot_data[3],robot_data[4],settings,
                            velocity_reference_rad_s=velocity_reference,
                            load_feedforward_nm=interface_following.last_contact_compensation)
                        if capacity_motion and load_audit['saturated']:raise RuntimeError('ORIGINAL_ARM_DRIVE_EFFORT_BOUNDARY')
                        if velocity_reference is not None:
                            robot.set_dof_velocity_targets(np.asarray(velocity_reference)[None,:],indices=0,dof_indices=robot_data[2])
                        if interface_following.seating_torque_candidate is not None and args.interface_release_at_end:
                            release_start=min(release_start,elapsed+dt)
                            (args.output/'seating_torque_candidate.json').write_text(json.dumps(interface_following.seating_torque_candidate,indent=2)+'\n')
                        if side_sequence and interface_following.side_grip is not None:
                            cap=interface_following.side_grip['third_actuator_cap_nm']
                            robot.set_dof_max_efforts(np.array([[cap]]),indices=0,
                                dof_indices=[robot.dof_names.index('f3j2')])
                except (RuntimeError,ValueError) as error:
                    interface_abort='FORCE_FOLLOWING: '+str(error);break
            if interface_following is not None and last_following_arm_drive is not None:
                command[:7]=last_following_arm_drive
                if interface_grip_recipe is not None:command[7:]=interface_following.hand_goal
            if stroke_schedule is not None and phase in ('regrasp_open','regrasp_open_hold','reindex','regrasp_close'):
                if regrasp_base_command is None:regrasp_base_command=command.copy()
                if phase=='regrasp_open':
                    command[7:]=(1.-phase_blend)*regrasp_base_command[7:]+phase_blend*release_open_target
                elif phase in ('reindex','regrasp_open_hold'):command[7:]=release_open_target
                else:
                    if interface_grip_recipe.get('reuse_grip_targets'):
                        contact_goal=remembered_grip_goal.copy()
                    else:
                        contact_goal=interface_following.contact_q.copy();contact_goal[1:]+=.002
                    command[7:]=(1.-phase_blend)*release_open_target+phase_blend*contact_goal
                if interface_grip_recipe.get('friction_aware_release'):
                    try:
                        finite_output_posture(command,contact_goal if phase=='regrasp_close' else release_open_target,
                                              allow_closing=phase=='regrasp_close')
                    except (ValueError,RuntimeError) as error:
                        interface_abort='REGRASP_MOTOR_INTERFACE: '+str(error);break
            if releasing:
                if release_base_command is None:release_base_command=command.copy()
                current_loaded_arm=command[:7].copy()
                command=release_base_command.copy();opening=final_release_open_target
                if capacity_motion:command[:7]=current_loaded_arm
                x=np.clip((elapsed-release_start-release_relax_duration)/.8,0.,1.);x=10*x**3-15*x**4+6*x**5
                command[7:]=(1.-x)*release_base_command[7:]+x*opening
                if (interface_grip_recipe and interface_grip_recipe.get('friction_aware_release')
                        and elapsed>=release_start+release_relax_duration):
                    try:
                        finite_output_posture(command,opening)
                    except (ValueError,RuntimeError) as error:
                        interface_abort='RELEASE_MOTOR_INTERFACE: '+str(error);break
            if palm_lock_native_angle is not None:
                command[7]=palm_lock_native_angle
            if fourbar_couplings:
                command_by_name=dict(zip(controller.ARM_JOINT_NAMES+controller.ACTIVE_HAND_JOINT_NAMES,command))
                for follower,coupling in fourbar_couplings.items():
                    commanded_follower,_=coupling.position_and_derivative(command_by_name[coupling.source_joint])
                    lower,upper=fourbar_joint_limits[follower]
                    if not lower-1e-12<=commanded_follower<=upper+1e-12:
                        raise ValueError('Four-bar command exceeds the original follower travel: '+follower)
            robot.set_dof_position_targets(command[None,:],indices=0,dof_indices=active)
            interface_command_rows.append(command.copy())
        if replay_records and step<len(replay_records):
            robot.set_dof_position_targets(replay_targets[step][None,:],indices=0,dof_indices=active)
        if step==round(1./dt) and args.paused_world_render:
            def snapshot():
                return {"positions":robot.get_dof_positions(indices=0).numpy().copy(),
                        "velocities":robot.get_dof_velocities(indices=0).numpy().copy(),
                        "wrist":host(tree.get_measured_joint_forces())[row_index].copy()}
            world.pause();before=snapshot();world.render();after=snapshot();world.play()
            render_audit={k:{"before":before[k].tolist(),"after":after[k].tolist(),
                             "maximum_change":float(np.max(np.abs(after[k]-before[k])))} for k in before}
        if args.main_read_sequence:
            robot.get_dof_positions(indices=0).numpy()
            robot.get_dof_velocities(indices=0).numpy()
            robot.get_dof_gravity_compensation_forces(indices=0).numpy()
        if fourbar_couplings:
            # Run at the physics rate, including controller-decimation skips.
            # Only constraint tangent coefficients change; no physical q/qd or
            # object/contact truth is used to overwrite the realized motion.
            measured_joints=host(robot.get_dof_positions(indices=0))[0]
            measured_sources={fourbar_couplings[name].source_joint:float(measured_joints[index])
                              for name,index in zip(fourbar_follower_order,fourbar_source_indices)}
            tangent_rows=update_fourbar_tangents(stage,'/World/HandArm/Physics',fourbar_couplings,measured_sources)
            if len(tangent_rows)!=len(fourbar_couplings):raise RuntimeError('Incomplete four-bar tangent update')
            for finger_index,(follower,source_index,follower_index,row) in enumerate(zip(
                    fourbar_follower_order,fourbar_source_indices,fourbar_follower_indices,tangent_rows)):
                q_source=float(measured_joints[source_index]);q_follower=float(measured_joints[follower_index])
                fourbar_step_rows.append([step,float(world.current_time),finger_index,q_source,q_follower,
                    fourbar_couplings[follower].closure_error(q_source,q_follower),row['slope'],row['offset_rad']])
        if worm_drives:
            measured_velocity=host(robot.get_dof_velocities(indices=0))[0]
            worm_targets=[]
            try:
                for follower,index in zip(fourbar_follower_order,fourbar_source_indices):
                    source=fourbar_couplings[follower].source_joint
                    target=float(command[7+controller.ACTIVE_HAND_JOINT_NAMES.index(source)])
                    law=worm_drives[follower].prepare_position(float(measured_joints[index]),float(measured_velocity[index]),
                        target,dt,stiffness=motor_position_kp,damping=motor_position_kd)
                    worm_targets.append(law['position_target'])
                # These replace the three output PD targets before the one
                # physical step. Arm, locked palm and passive followers retain
                # their existing control/constraint roles.
                robot.set_dof_position_targets(np.asarray([worm_targets]),indices=0,dof_indices=fourbar_source_indices)
            except (ValueError,RuntimeError) as error:
                interface_abort='WORM_MOTOR_COMMAND: '+str(error);break
        if shared_hand_runtime is not None:
            shared_hand_runtime.submit(command[7:],phase)
        tick=time.monotonic()
        try:
            world.step(render=False)
        except RuntimeError as error:
            if shared_hand_runtime is None:raise
            interface_abort='SHARED_HAND_RUNTIME: '+str(error)
        performance['physics_s']+=time.monotonic()-tick
        if worm_drives:
            worm_q=host(robot.get_dof_positions(indices=0))[0];worm_velocity=host(robot.get_dof_velocities(indices=0))[0]
            projected=host(robot.get_dof_projected_joint_forces(indices=0))[0]
            for follower,pi,di,tangent in zip(fourbar_follower_order,fourbar_source_indices,fourbar_follower_indices,tangent_rows):
                observed=float(projected[pi]+tangent['slope']*projected[di])
                row=worm_drives[follower].complete(float(worm_q[pi]),float(worm_velocity[pi]),observed_drive_effort=observed)
                row.update(step=step,time_s=float(world.current_time),follower_joint=follower,phase=phase)
                worm_records.append(row);worm_stream.write(json.dumps(row)+'\n')
                if row['elastic_effort_boundary_exceeded'] or row['drive_saturation'] or row['friction_heat_j'] < -1e-9:
                    interface_abort='WORM_PHYSICAL_PORT_BOUNDARY: '+follower
            if step%240==0:worm_stream.flush()
        readback_started=time.monotonic()
        if args.main_read_sequence:
            before_queries=host(tree.get_measured_joint_forces())[row_index].copy()
            robot.get_dof_positions(indices=0).numpy()
            robot.get_dof_velocities(indices=0).numpy()
            robot.get_dof_projected_joint_forces(indices=0).numpy()
            after_projected=host(tree.get_measured_joint_forces())[row_index].copy()
            fixture_tree.get_measured_joint_forces()
            after_fixture=host(tree.get_measured_joint_forces())[row_index].copy()
            read_deltas.append([*(after_projected-before_queries),*(after_fixture-after_projected)])
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            pos,ori=contacts.get_world_poses();vel,angvel=contacts.get_velocities()
        pos=pos.numpy();rot=Rotation.from_quat(ori.numpy()[:,[1,2,3,0]]).as_matrix();origin=pos[hb]
        hand_link_pose_rows.append(np.c_[pos,ori.numpy()])
        centers=pos+np.einsum("nij,nj->ni",rot,coms);gravity=masses[:,None]*np.array([0.,0.,-9.81])
        expected=np.r_[gravity.sum(0),np.cross(centers-origin,gravity).sum(0)]
        gravity_wrench=expected.copy();normal_wrench=np.zeros(6);friction_wrench=np.zeros(6)
        normal,points,normals,hand_separation,counts,starts,actor_ids=contacts.get_raw_contact_data()
        normal=normal.numpy().ravel();points=points.numpy();normals=normals.numpy()
        hand_separation=hand_separation.numpy().ravel()
        normal_load=np.zeros(len(hand_paths))
        hand_normal_wrenches=np.zeros((len(hand_paths),6))
        # Starts/counts have one row per sensor and one column per filter.
        # A free connector uses multiple filters; flattened indices are not
        # hand-link indices and must not be used to label force records.
        contact_starts=starts.numpy().reshape(len(hand_paths),-1)
        contact_counts=counts.numpy().reshape(len(hand_paths),-1)
        for sensor_index in range(len(hand_paths)):
            for start,count in zip(contact_starts[sensor_index],contact_counts[sensor_index]):
                sl=slice(int(start),int(start+count));F=normal[sl,None]*normals[sl]/dt
                wrench=np.r_[F.sum(0),np.cross(points[sl]-origin,F).sum(0)]
                expected+=wrench;normal_wrench+=wrench
                hand_normal_wrenches[sensor_index]+=wrench
                normal_load[sensor_index]+=float(np.linalg.norm(F,axis=1).sum())
                partner_paths=recorded_actor_paths(actor_ids[sl]) if count else ()
                if count and step%args.contact_record_stride==0:
                    hand_contact_point_records.append({'step':step,'sensor_path':hand_paths[sensor_index],
                        'points_world_m':points[sl].tolist(),'normal_world':normals[sl].tolist(),
                        'normal_force_n':(normal[sl]/dt).tolist(),
                        'separation_m':hand_separation[sl].tolist(),
                        'partner_paths':list(partner_paths)})
                if count:
                    for partner in partner_paths:
                        contact_actor_pairs.add((hand_paths[sensor_index],partner))
        if not args.raw_contact_only:
            friction,points,counts,starts=contacts.get_friction_data();friction=friction.numpy()/dt;points=points.numpy()
            for start,count in zip(starts.numpy().ravel(),counts.numpy().ravel()):
                if not count:continue
                sl=slice(int(start),int(start+count));F=friction[sl]
                wrench=np.r_[F.sum(0),np.cross(points[sl]-origin,F).sum(0)]
                expected+=wrench;friction_wrench+=wrench
        else:
            # Missing friction measurements are not zero physical friction.
            # NaN is retained in the raw NPZ; JSON summaries use null.
            friction_wrench[:]=np.nan;expected[:]=np.nan
        net_force=contacts.get_net_contact_forces(dt=dt).numpy().sum(axis=0)
        filtered_force=(np.full(3,np.nan) if args.raw_contact_only else contacts.get_contact_force_matrix(dt=dt).numpy().sum(axis=(0,1)))
        contact_component_rows.append(np.r_[gravity_wrench,normal_wrench,friction_wrench,net_force,filtered_force])
        native_wrenches=host(tree.get_measured_joint_forces()).copy()
        raw=native_wrenches[row_index];canonical=np.r_[-rot[hb]@raw[:3],-rot[hb]@raw[3:]]
        hand_normal_wrench_rows.append(hand_normal_wrenches)
        native_joint_wrench_rows.append(native_wrenches)
        momentum=(masses[:,None]*vel.numpy()).sum(0)
        rows.append([*canonical,*expected,*(canonical-expected),*momentum,*origin])
        joint_rows.append([*robot.get_dof_positions(indices=0).numpy()[0],
                           *robot.get_dof_velocities(indices=0).numpy()[0]])
        with use_backend("tensor",raise_on_unsupported=True,raise_on_fallback=True):
            part_pos,part_ori=parts.get_world_poses();part_vel,part_angvel=parts.get_velocities()
        part_rows.append(np.c_[part_pos.numpy(),part_ori.numpy(),part_vel.numpy(),part_angvel.numpy()])
        normal_load_rows.append(normal_load)
        if static_part_contact_recording:
            f,_,_,separation,counts,starts,_=parts.get_raw_contact_data()
            f=f.numpy().ravel();separation=separation.numpy().ravel();stats=[]
            for row_starts,row_counts in zip(starts.numpy().reshape(2,-1),counts.numpy().reshape(2,-1)):
                force=0.;minimum=0.;number=0
                for start,count in zip(row_starts,row_counts):
                    if not count:continue
                    sl=slice(int(start),int(start+count));force+=float(np.maximum(0.,f[sl]).sum()/dt)
                    minimum=min(minimum,float(separation[sl].min()));number+=int(count)
                stats.append([force,minimum,number])
            part_contact_rows.append(stats)
        if args.interface_twist_deg is not None and (step==round(interface_duration/dt)-1 or (args.interface_release_at_end and step==round(release_start/dt)-1)):
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            from te_fast_contact_reading import read_shape_contact_pairs_fast
            decoded={}
            def decode_id(value):
                key=int(value)
                if key not in decoded:decoded[key]=str(PhysicsSchemaTools.intToSdfPath(key))
                return decoded[key]
            class ContactSnapshot:
                def __init__(self,data):self.data=data
                def get_full_contact_report(self):return self.data
            snapshot=ContactSnapshot(get_physx_simulation_interface().get_full_contact_report())
            contact_rows=[read_shape_contact_pairs_fast(snapshot,dt,p,host(part_pos)[i],decode_path=decode_id) for i,p in enumerate([body_path,fixture_path])]
            for actor_rows in contact_rows:
                for row in actor_rows:row.pop('friction_wrench_n_nm',None)
            final_contact_snapshots.append({'step':step,'phase':phase,'shape_contacts':contact_rows,'raw_sdk_friction_not_used':True})
        performance['readback_s']+=time.monotonic()-readback_started
        if args.interface_twist_deg is not None:
            if step % 240 == 0 or step == round(interface_duration/dt)-1:
                evidence_started=time.monotonic()
                world.pause();before_time=float(world.current_time)
                timeline=omni.timeline.get_timeline_interface();auto=timeline.is_auto_updating()
                settings_api=carb.settings.get_settings();play=settings_api.get('/app/player/playSimulations')
                try:
                    timeline.set_auto_update(False);timeline.commit_silently();settings_api.set('/app/player/playSimulations',False)
                    for _ in range(3):omni.kit.app.get_app().update()
                    settings_api.set('/app/player/playSimulations',play)
                    rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
                finally:
                    settings_api.set('/app/player/playSimulations',play);timeline.set_auto_update(auto);timeline.commit_silently()
                frame=np.asarray(rgb.get_data()).copy();world.play()
                if abs(float(world.current_time)-before_time)>1e-9:raise RuntimeError('Evidence capture advanced physics')
                if frame.size:
                    image=cv2.cvtColor(frame[:,:,:3],cv2.COLOR_RGB2BGR);path=args.output/f'interface_{step:04d}.png';cv2.imwrite(str(path),image);interface_movie.write(image);interface_images.append(str(path))
                (args.output/'interface_progress.json').write_text(json.dumps({'step':step,'time_s':float(world.current_time),
                    'phase':phase,
                    'positions_world_m':host(part_pos).tolist(),'quaternions_wxyz':host(part_ori).tolist(),
                    'hand_normal_load_n':normal_load.tolist(),'hand_wrist_position_world_m':origin.tolist()})+'\n')
                performance['evidence_s']+=time.monotonic()-evidence_started
            qforce=host(robot.get_dof_projected_joint_forces(indices=0))[0]
            projected_joint_reaction_rows.append(qforce.copy())
            # Preserve the source check's1.2Nm finger protection and finite
            # model excursion stop. These measurements only abort the run.
            closing_indices=[robot.dof_names.index(n) for n in ('f1j2','f2j1','f3j2')]
            if not args.closing_reaction_record_only and np.max(np.abs(qforce[closing_indices]))>args.closing_reaction_stop_nm:
                interface_abort='ORIGINAL_FINGER_REACTION_LIMIT';break
            delta=host(part_pos)-np.array([body_T[:3,3],nut_T[:3,3]])
            axial_axis=(placement@bench_socket)[:3,2] if args.connector_initial_pose else nut_T[:3,2]
            along=delta@axial_axis;lateral=delta-along[:,None]*axial_axis
            travel=max(.003,abs(args.interface_twist_deg)*.00762/360.+.002)
            if not np.isfinite(host(part_pos)).all() or np.max(np.linalg.norm(lateral,axis=1))>.003 or np.max(np.abs(along))>travel:
                interface_abort='FINITE_CONNECTOR_DISPLACEMENT_LIMIT';break
        if interface_abort is not None:break
    if (args.interface_twist_deg is not None and rows
            and not any(r['step']==len(rows)-1 for r in final_contact_snapshots)):
        # An observation stop can precede the scheduled final snapshot. Record
        # the last actual contacts after motion ends; never feed them to control.
        try:
            from types import SimpleNamespace
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            from te_fast_contact_reading import read_shape_contact_pairs_fast
            terminal_report=get_physx_simulation_interface().get_full_contact_report()
            terminal_reader=SimpleNamespace(get_full_contact_report=lambda:terminal_report)
            terminal_contacts=[read_shape_contact_pairs_fast(terminal_reader,dt,p,host(part_pos)[i],
                decode_path=lambda value:str(PhysicsSchemaTools.intToSdfPath(int(value))))
                for i,p in enumerate([body_path,fixture_path])]
            for actor_rows in terminal_contacts:
                for contact_row in actor_rows:contact_row.pop('friction_wrench_n_nm',None)
            final_contact_snapshots.append({'step':len(rows)-1,'phase':phase_rows[len(rows)-1],
                'shape_contacts':terminal_contacts,'raw_sdk_friction_not_used':True,
                'postrun_terminal_snapshot':True,'abort':interface_abort})
        except Exception as error:
            (args.output/'terminal_contact_snapshot_error.json').write_text(json.dumps({'error':str(error)})+'\n')
    if worm_stream is not None:worm_stream.flush();worm_stream.close();worm_stream=None
    if interface_movie is not None:interface_movie.release()
    if interface_following is not None:
        (args.output/'force_following.json').write_text(json.dumps(interface_following.report(),indent=2)+'\n')
        (args.output/'force_following_samples.json').write_text(json.dumps(interface_following.records)+'\n')
        (args.output/'release_wrench_samples.json').write_text(json.dumps(interface_following.release_records)+'\n')
    half_second=min(half_second,len(rows))
    a=np.array(rows);result={"scope":("LOCAL_FREE_BODY_NUT_SOCKET_CONTACT_WITH_ORIGINAL_HAND_NOT_FULL_ASSEMBLY"
        if args.free_plug_in_socket else "STATIC_ORIGINAL_HAND_WITH_EXPLICITLY_MOUNTED_NUT_FIXTURE_NOT_ASSEMBLY"),
      "source_run":str(args.run),"source_step":sample_step,"velocity_iterations":args.velocity_iterations,
      "physics_device":args.physics_device,
      "solver_type_readback":scene.GetSolverTypeAttr().Get(),
      "physics_dt_s":dt,"physics_hz":args.physics_hz,
      "position_iterations":args.position_iterations,"external_forces_every_iteration":bool(args.external_forces_every_iteration),"hand_paths":hand_paths,
      "native_drive_settings":settings,
      "robot_dof_names":list(robot.dof_names),
      "native_joint_wrench_indices":dict(tree._articulation_view._metadata.joint_indices),
      "additional_force_logging_role":"OFFLINE_EVALUATION_ONLY_NO_CONTROL_INPUT",
      "hand_joint_friction_authoring":hand_friction_report,
      "robot_initialization_before_reset":robot_initialization_report,
      "passive_joint_initialization":passive_initialization,
      "native_articulation_velocity_iterations": {str(p.GetPath()):PhysxSchema.PhysxArticulationAPI(p).GetSolverVelocityIterationCountAttr().Get()
          for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxArticulationAPI)},
      "main_native_read_sequence":args.main_read_sequence,
      "paused_world_render_audit":render_audit,
      "maximum_query_wrist_change": (np.max(np.abs(read_deltas),axis=0).tolist() if read_deltas else None),
      "native_hand_mass_kg":float(masses.sum()),"robot_pose_initialized_while_paused_before_measurement":True,
      "nut_pose_or_drive_targets_changed_after_measurement_start":bool(replay_records),
      "connector_pose_changed_after_measurement_start":False,
      "nut_fixture_mount_added":not args.free_plug_in_socket,
      "last_half_second_mean":a[-half_second:].mean(0).tolist(),"columns":"canonical_wrist_world[6],gravity_plus_contact_world[6],difference[6],hand_linear_momentum[3],wrist_position[3]",
      "half_second_before_midpoint_mean":a[half_second:2*half_second].mean(0).tolist() if len(a)>half_second else None,
      "last_half_second_wrist_position_range_m":np.ptp(a[-half_second:,-3:],axis=0).tolist(),
      "native_momentum_change_per_second_n":((a[-1,18:21]-a[-half_second,18:21])/(max(1,half_second-1)*dt)).tolist()}
    if args.interface_twist_deg is not None:
        result['interface_check']={'requested_wrist_joint7_deg':args.interface_twist_deg,'abort':interface_abort,
            'additional_closing_target_deg':args.interface_extra_closure_deg,
            'unvalidated_legacy_pad_sensitivity':args.interface_source_pad_material,
            'duration_s':len(rows)*dt,'images':interface_images,'no_connector_fixture_or_external_tool_added':True,
            'complete_robot_assembly_verified':False,'declared_bench_initial_pose':str(args.connector_initial_pose)}
    result["last_half_second_hand_normal_load_n"]=np.mean(normal_load_rows[-half_second:],axis=0).tolist()
    result['performance_timing_s']=performance
    result['diagnostic_actor_decode_cache_entries']=len(actor_path_cache)
    result['physical_release_requested']=args.interface_release_at_end
    (args.output/'phase_records.json').write_text(json.dumps(phase_rows)+'\n')
    (args.output/'final_normal_contact_snapshots.json').write_text(json.dumps(final_contact_snapshots)+'\n')
    result['force_limit_interpretation']={
        'user_measurement':'approximately16N weight lifted through a tip rope; nominal15N directional operating force',
        'measurement_record':'src/kcg_connector/config/hand_tip_pull_measurement_20260911.yaml',
        'contact_normal_force_limit_n':None,
        'normal15N_is_calibrated_hardware_capacity':False,
        'tip_test_to_current_contact_actuator_mapping_complete':False,
        'contact_force_exceedance_alone_establishes_hardware_overload':False,
        'contact_normal15Nrestriction_removed_by_explicit_user_request':True,
        'projected_reaction_reference_record_only':args.closing_reaction_record_only,
        'finite_closing_drive_cap_nm':args.closing_drive_cap_nm}
    if replay_records:
        result["scope"]="BOUNDED_SEALED_ROBOT_COMMAND_REPLAY_THEN_STATIC_HOLD_NOT_ASSEMBLY"
        result["source_command_replay"]={"source_step_ids":[s["step"] for s in replay_records],
            "replayed_command_count":len(replay_records),"then_hold_last_target":True,
            "original_drive_gains_caps_and_joint_limits_retained":True,
            "online_feedback_or_contact_truth_used":False,
            "initial_velocity_and_contact_history_not_restored":True}
        np.savez_compressed(args.output/"replayed_drive_targets.npz",targets_rad=replay_targets,
            source_steps=np.asarray([s["step"] for s in replay_records]))
    result["last_half_second_part_position_range_m"]=np.ptp(np.array(part_rows)[-half_second:,:,:3],axis=0).tolist()
    result["all_raw_hand_contact_actor_pairs"]=sorted(contact_actor_pairs)
    result["contact_component_columns"]="gravity[6],raw_normal_wrench[6],filtered_friction_wrench[6],native_net_contact_force[3],native_filtered_contact_force_matrix_sum[3]"
    result["last_half_second_contact_component_mean"]=np.mean(contact_component_rows[-half_second:],axis=0).tolist()
    if args.frozen_connector_model is not None:
        result['contact_readback_validity']={
            'raw_sdk_multi_patch_friction_is_validated_ground_truth':False,
            'gravity_plus_raw_contact_balance_is_validated_ground_truth':False,
            'reason':'Installed PhysX multi-patch friction writeback can overwrite patch records; preserve raw arrays for diagnostics only',
            'canonical_wrist_and_projected_joint_sensor_data_modified':False,
            'online_control_uses_raw_contact_friction':False}
    np.savez_compressed(args.output/"samples.npz",values=a,joints=np.array(joint_rows),
                        parts=np.array(part_rows),hand_normal_load=np.array(normal_load_rows),
                        contact_components=np.array(contact_component_rows),part_contact_stats=np.array(part_contact_rows),
                        interface_commands=np.array(interface_command_rows),projected_joint_reactions=np.array(projected_joint_reaction_rows),
                        hand_normal_wrenches_world_about_handbase=np.array(hand_normal_wrench_rows),
                        native_joint_wrenches=np.array(native_joint_wrench_rows),
                        hand_link_poses=np.array(hand_link_pose_rows))
    (args.output/'hand_contact_point_records.json').write_text(json.dumps(hand_contact_point_records)+'\n')
    if static_part_contact_recording:
        result['part_contact_stats_columns']='positive_normal_force_sum_n,minimum_separation_m,contact_count'
        result['maximum_part_contact_normal_sum_n']=np.max(np.array(part_contact_rows)[:,:,0],axis=0).tolist()
    if args.audit_source_key_sdf:
        import torch
        world.pause()
        core_path=prepared["report"]["grounding_band_contact_model"]["rigid_core_collision"]
        core=UsdGeom.Mesh.Get(stage,core_path)
        vertices=np.asarray(core.GetPointsAttr().Get(),dtype=np.float64)
        faces=np.asarray(core.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).reshape(-1,3)
        slab=(vertices[:,2]>=-.007646)&(vertices[:,2]<=-.000761)
        protruding=slab&(np.linalg.norm(vertices[:,:2],axis=1)>.0183)
        selected=np.flatnonzero(np.any(protruding[faces],axis=1)&np.all(slab[faces],axis=1))
        if len(selected)!=106:
            raise RuntimeError("source-key SDF query must retain the original 106 key triangles")
        triangles=vertices[faces[selected]]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        bary=np.asarray([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3,1/3,1/3]])
        surface=np.einsum("ij,kjl->kil",bary,triangles).reshape(-1,3)
        directions=np.repeat(normals,len(bary),axis=0)
        resolution=int(PhysxSchema.PhysxSDFMeshCollisionAPI(core.GetPrim()).GetSdfResolutionAttr().Get())
        spacing=float(np.ptp(vertices,axis=0).max()/resolution)
        points=np.concatenate([surface-spacing*directions,surface,surface+spacing*directions])
        before=robot.get_dof_positions(indices=0).numpy().copy()
        view=world.physics_sim_view.create_sdf_shape_view(core_path,len(points))
        if view.count!=1:
            raise RuntimeError("SDF reader did not bind exactly the existing rigid core")
        queried=host(view.get_sdf_and_gradients(torch.as_tensor(points[None,:,:],dtype=torch.float32,device=args.physics_device))).copy()[0]
        after=robot.get_dof_positions(indices=0).numpy().copy()
        # This installed tensor returns gradient XYZ then SDF, contrary to the
        # current high-level docstring. Paired normal-offset queries retain a
        # direct unit/layout check; never treat a unit gradient as metres.
        count=len(surface);sdf=queried[count:2*count,3]
        derivative=(queried[2*count:,3]-queried[:count,3])/(2*spacing)
        centroid_indices=np.arange(6,count,7)
        gradient_norm=np.linalg.norm(queried[count:2*count,:3],axis=1)
        if not (.8<float(np.median(derivative[centroid_indices]))<1.2
                and .8<float(np.median(gradient_norm))<1.2):
            raise RuntimeError("native SDF result layout did not satisfy the source-normal offset check")
        np.savez_compressed(args.output/"source_key_native_sdf_query.npz",query_points_mesh_local_m=points,
            raw_sdf_and_gradient=queried,source_surface_points_m=surface,source_face_indices=selected,
            face_sample_barycentric=bary,offset_directions=directions)
        sdf_report={"scope":"POST_MEASUREMENT_NATIVE_SDF_VS_ORIGINAL_KEY_SURFACE_SAMPLES",
            "collision_path":core_path,"source_key_triangle_count":106,"sampled_surface_point_count":count,
            "sdf_resolution":resolution,"grid_spacing_m":spacing,
            "source_surface_sdf_min_median_max_m":[float(sdf.min()),float(np.median(sdf)),float(sdf.max())],
            "source_surface_max_absolute_sdf_m":float(np.max(np.abs(sdf))),
            "normal_offset_finite_difference_median":float(np.median(derivative)),
            "face_centroid_normal_offset_finite_difference_median":float(np.median(derivative[centroid_indices])),
            "maximum_robot_state_change_during_queries_rad":float(np.max(np.abs(after-before))),
            "field_layout":"LOCAL_GRADIENT_XYZ_THEN_SDF_METERS_VERIFIED_WITH_NORMAL_OFFSETS",
            "sampling_is_not_a_global_error_bound_or_assembly_pass":True,
            "no_physics_step_or_asset_authoring_during_query":True,"online_control_use":False}
        (args.output/"source_key_native_sdf_query.json").write_text(json.dumps(sdf_report,indent=2)+"\n")
        result["post_measurement_source_key_sdf_audit"]=sdf_report
    if read_deltas:
        np.savez_compressed(args.output/"native_query_changes.npz",values=np.array(read_deltas))
    result['raw_contact_only']=args.raw_contact_only
    result['hand_role']=args.hand_role
    result['production_acceptance_eligible']=args.hand_role=='production-nails' and not fourbar_couplings
    result['palm_layout_mechanism']=args.palm_layout_mechanism
    if shared_hand_runtime is not None:
        result['shared_hand_runtime']={'source':str(args.shared_hand_mechanism),'steps':shared_hand_runtime.steps,
            'failure':shared_hand_runtime.failure,'full_palm_travel_retained':True,
            'zero_width_palm_joint_limits_used':False,'motor_state_reset_during_episode':False,
            'stream':'hand_mechanism_samples.jsonl.gz'}
    result['interface_grip_only']=args.interface_grip_only
    if worm_drives:
        result['finger_worm_integration']={
            'mode':'PASSIVE_SPLIT_WITH_IMPLICIT_MOTOR_POSITION_PD','records':len(worm_records),
            'maximum_motor_substep_energy_residual_j':max((abs(r['motor_substep_energy_residual_j']) for r in worm_records),default=None),
            'motor_effort_saturated_steps':sum(r['motor_effort_at_boundary'] for r in worm_records),
            'maximum_input_effort_nm':max((abs(r['input_effort']) for r in worm_records),default=None),
            'maximum_transmission_effort_nm':max((abs(r['transmission_effort']) for r in worm_records),default=None),
            'physical_position_or_velocity_overwrites':False,'full_assembly_verified':False}
    if fourbar_couplings:
        samples=np.asarray(fourbar_step_rows,dtype=float).reshape(-1,8)
        np.savez_compressed(args.output/'finger_fourbar_step_samples.npz',values=samples)
        result['finger_mechanism_candidate']={
            'mechanism_id':finger_mechanism_document['mechanism_id'],
            'status':'MECHANISM_CANDIDATE_NOT_PRODUCTION_ASSEMBLY_ACCEPTANCE',
            'finger_self_lock_and_complete_assembly_still_require_verification':True,
            'physical_steps_with_tangent_update':len(samples)//len(fourbar_couplings),
            'reset_warmup_uses_fixed_initial_tangent':True,
            'maximum_recorded_rod_length_error_m':float(np.abs(samples[:,5]).max()) if len(samples) else None,
            'sample_follower_order':list(fourbar_follower_order),
            'sample_columns':['step','physical_time_s','finger_index','source_rad','follower_rad',
                              'rod_length_error_m','dp_dq','tangent_offset_rad'],
            'physical_position_and_velocity_overwrites_after_reset':False}
    if args.raw_contact_only:
        result['unavailable_record_fields']=['friction contact wrench','filtered contact force matrix','gravity plus complete contact support wrench']
        def finite_json(value):
            if isinstance(value,float) and not math.isfinite(value):return None
            if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
            if isinstance(value,list):return [finite_json(v) for v in value]
            return value
        result=finite_json(result)
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
except Exception:
    failed=True
    import traceback
    error=traceback.format_exc();(args.output/"error.txt").write_text(error);print(error,flush=True)
finally:
    if locals().get('shared_hand_runtime') is not None:shared_hand_runtime.close()
    if locals().get('worm_stream') is not None:worm_stream.flush();worm_stream.close()
    if previous_physics_dispatch_settings is not None:
        for name,value in previous_physics_dispatch_settings.items():
            if value is None:carb.settings.get_settings().destroy_item(name)
            else:carb.settings.get_settings().set(name,value)
    # The finite interface check saves its frames, video and samples itself.
    # Do not wait on stage teardown after those artifacts are complete.
    if args.interface_twist_deg is not None:
        if locals().get('interface_movie') is not None:interface_movie.release()
        app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=1 if failed else requested_exit_code)
    else:
        app.close(exit_code=1 if failed else requested_exit_code)
