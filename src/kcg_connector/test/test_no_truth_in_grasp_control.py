import ast
import inspect

from kcg_connector.grasp import finger_contact_detector
from kcg_connector.grasp import grasp_stability_monitor
from kcg_connector.grasp import lift_recovery
from kcg_connector.grasp import posthoc_wrench_analysis
from kcg_connector.grasp import randomization
from kcg_connector.grasp import realized_authoring
from kcg_connector.grasp import single_finger_contact_test
from kcg_connector.grasp import single_finger_gui_consistency
from kcg_connector.grasp import single_finger_posthoc_audit
from kcg_connector.grasp import terminal_evaluator
from kcg_connector.grasp import three_finger_sequential_grasp


CONTROL_MODULES = (
    finger_contact_detector,
    grasp_stability_monitor,
    lift_recovery,
    realized_authoring,
    three_finger_sequential_grasp,
    single_finger_contact_test,
)

LOG_ONLY_MODULES = (
    posthoc_wrench_analysis,
    single_finger_posthoc_audit,
    single_finger_gui_consistency,
    terminal_evaluator,
    randomization,
)
FORBIDDEN_CALLS = {
    "get_full_contact_report",
    "get_world_pose",
    "GetWorldTransform",
    "ComputeLocalToWorldTransform",
    "GetContactReport",
}
FORBIDDEN_IMPORT_ROOTS = {"omni", "pxr", "isaacsim"}


def test_control_modules_do_not_import_simulator_or_posthoc_evaluator():
    for module in CONTROL_MODULES:
        tree = ast.parse(inspect.getsource(module))
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
                assert "grasp_evidence" not in node.module
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    calls.add(function.id)
                elif isinstance(function, ast.Attribute):
                    calls.add(function.attr)
        assert imports.isdisjoint(FORBIDDEN_IMPORT_ROOTS)
        assert calls.isdisjoint(FORBIDDEN_CALLS)


def test_log_only_diagnostics_are_pure_and_never_touch_the_simulator():
    # The posthoc diagnostics and terminal evaluator handle truth-flavored
    # values, but they must stay pure Python: no simulator imports and no
    # simulator API calls.  Their outputs are dicts for the report only.
    for module in LOG_ONLY_MODULES:
        tree = ast.parse(inspect.getsource(module))
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    calls.add(function.id)
                elif isinstance(function, ast.Attribute):
                    calls.add(function.attr)
        assert imports.isdisjoint(FORBIDDEN_IMPORT_ROOTS)
        assert calls.isdisjoint(FORBIDDEN_CALLS)


def test_window_statistics_block_signature_has_no_truth_fragments():
    signature = inspect.signature(
        posthoc_wrench_analysis.window_statistics_block
    )
    names = " ".join(signature.parameters)
    for fragment in ("plug", "contact", "collider", "truth", "pose"):
        assert fragment not in names, (
            f"window_statistics_block parameter mentions {fragment}"
        )


def test_runtime_update_signatures_are_sensor_and_robot_state_only():
    signatures = (
        inspect.signature(finger_contact_detector.FingerContactDetector.update),
        inspect.signature(three_finger_sequential_grasp.ThreeFingerSequentialGrasp.update),
        inspect.signature(grasp_stability_monitor.GraspStabilityMonitor.update),
        inspect.signature(lift_recovery.plan_recovery_return),
        inspect.signature(lift_recovery.plan_recovery_open),
        inspect.signature(grasp_stability_monitor.wrist_payload_increment),
    )
    forbidden_fragments = (
        "plug",
        "receptacle",
        "contact_normal",
        "contact_point",
        "collider",
        "penetration",
        "truth",
    )
    for signature in signatures:
        names = " ".join(signature.parameters)
        assert not any(fragment in names for fragment in forbidden_fragments)
