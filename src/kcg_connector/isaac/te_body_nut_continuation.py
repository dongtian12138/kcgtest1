"""Bounded reuse of the existing release, regrasp and nut-turn controllers.

No measured object/contact truth enters this sequence. A terminal torque stop
may be followed by a current-vision guide check and controlled release, while
the original stop remains recorded. Final seating is a separate physical audit.
"""
import copy


def continue_nut_strokes_and_release(repository, runtime, stepper, dynamic, record,
        socket, assembly, collision_scene, obstacles, output, save_record):
    from te_body_nut_regrasp import run_body_nut_regrasp
    from te_body_nut_rotation import run_body_nut_rotation
    from te_body_nut_reindex import (
        run_nut_release_and_reindex, can_unload_after_torsional_pilot_stop,
        can_unload_after_transmission_reserve_stop)

    settings = assembly["continued_nut_strokes"]
    maximum = settings["maximum_additional_strokes"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 6:
        raise ValueError("continued nut assembly is bounded to at most six additional strokes")
    turn_settings = copy.deepcopy(assembly["nut_rotation_after_index"])
    angle = float(turn_settings["rotation_about_socket_plus_z_deg"])
    if not -120. <= angle < 0.:
        raise ValueError("continued strokes retain the bounded tightening direction")
    last_rotation = record.get("nut_rotation_after_index", record["nut_rotation"])
    last_grip = (record["nut_regrasp_after_index"] if "nut_rotation_after_index" in record
                 else record["nut_regrasp"])
    total_requested = sum(abs(float(record[name]["settings"]["rotation_about_socket_plus_z_deg"]))
        for name in ("nut_rotation", "nut_rotation_after_index") if name in record)
    series = record["continued_nut_strokes"] = {
        "maximum_additional_strokes": maximum, "attempts": [],
        "sum_requested_stroke_angles_deg": total_requested,
        "requested_angles_are_not_actual_rotation": True,
        "full_assembly_requires_postrun_physical_evaluation": True,
        "online_object_or_contact_truth_used": False,
    }
    for index in range(maximum):
        if not record.get("completed") or not last_rotation.get("completed"):
            break
        if total_requested + abs(angle) > 600.:
            series["termination"] = "REQUESTED_ROTATION_BUDGET_REACHED"
            break
        entry = {"index": index + 1, "requested_rotation_deg": angle}
        series["attempts"].append(entry)
        reindex_settings = copy.deepcopy(assembly["nut_reindex"])
        reindex_settings.update(release_only=False,
            rotation_about_socket_plus_z_deg=-float(last_rotation["settings"]["rotation_about_socket_plus_z_deg"]))
        record["stage"] = "CONTINUED_NUT_RELEASE_AND_OPEN_HAND_REINDEX"
        save_record()
        entry["reindex"] = run_nut_release_and_reindex(
            repository, runtime, stepper, dynamic, last_grip, last_rotation,
            socket, reindex_settings, output / f"nut_reindex_continued_{index+1:02d}")
        record["completed"] = bool(entry["reindex"].get("completed"))
        if not record["completed"]:
            record["failure_reason"] = entry["reindex"].get("failure_reason")
            save_record()
            break
        record["stage"] = "CONTINUED_CURRENT_VISION_NUT_REGRASP"
        save_record()
        entry["grip"] = run_body_nut_regrasp(
            repository, runtime, stepper, dynamic, entry["reindex"]["final_observation"],
            socket, collision_scene, obstacles, output / f"nut_regrasp_continued_{index+1:02d}")
        record["completed"] = bool(entry["grip"].get("completed"))
        if not record["completed"]:
            record["failure_reason"] = entry["grip"].get("failure_reason")
            save_record()
            break
        last_grip = entry["grip"]
        record["stage"] = "CONTINUED_BOUNDED_NUT_ROTATION"
        total_requested += abs(angle)
        series["sum_requested_stroke_angles_deg"] = total_requested
        save_record()
        entry["rotation"] = run_body_nut_rotation(
            repository, runtime, stepper, dynamic, last_grip, socket,
            copy.deepcopy(turn_settings), output / f"nut_rotation_continued_{index+1:02d}")
        last_rotation = entry["rotation"]
        record["completed"] = bool(last_rotation.get("completed"))
        if not record["completed"]:
            record["failure_reason"] = last_rotation.get("failure_reason")
        save_record()

    latest_phase = runtime["nail_body_ft_auditor"].samples[-1]["phase"]
    pilot_stop = can_unload_after_torsional_pilot_stop(
        last_rotation, latest_phase, int(stepper.step_index), stepper.abort_reason)
    pilot_stop = pilot_stop or can_unload_after_transmission_reserve_stop(
        last_rotation, latest_phase, int(stepper.step_index), stepper.abort_reason)
    release_eligible = (stepper.abort_reason is None
        and last_rotation.get("last_step") == int(stepper.step_index)
        and ((record.get("completed") and last_rotation.get("completed")) or pilot_stop))
    series["terminal_release_eligible"] = bool(release_eligible)
    if settings.get("release_at_end", True) and release_eligible:
        release_settings = copy.deepcopy(assembly["nut_reindex"])
        release_settings.update(release_only=True, open_hold_duration_s=3.,
            rotation_about_socket_plus_z_deg=0., allow_release_after_torsional_pilot_stop=True,
            allow_release_after_transmission_reserve_stop=True)
        record["stage"] = "TERMINAL_CURRENT_VISION_CHECK_AND_NUT_RELEASE"
        save_record()
        series["terminal_release"] = run_nut_release_and_reindex(
            repository, runtime, stepper, dynamic, last_grip, last_rotation,
            socket, release_settings, output / "nut_terminal_release")
        # The additional torque stop is retained, even if unloading succeeds.
        # It can coincide with seating, but only the physical audit decides.
        record["completed"] = bool(last_rotation.get("completed")
            and series["terminal_release"].get("completed"))
        if not series["terminal_release"].get("completed"):
            series["terminal_release_failure_reason"] = series["terminal_release"].get("failure_reason")
            if not record.get("failure_reason"):
                record["failure_reason"] = series["terminal_release_failure_reason"]
    record["completed_scope"] = "BOUNDED_TURN_SERIES_AND_RELEASE_REQUIRE_FULL_ASSEMBLY_PHYSICAL_EVALUATION"
    save_record()
