"""Deterministic policy used to validate the RL reset/step contract."""

import argparse
import json
import sys

import numpy as np

from .cylinder_env import (
    HAND_LOWER_BOUNDS,
    HAND_UPPER_BOUNDS,
    KcgCylinderEnv,
    VALIDATED_CLOSED_HAND_POSITIONS,
)


def run_smoke(env: KcgCylinderEnv, episode: int = 1) -> bool:
    observation, info = env.reset()
    print(
        "RL RESET OK: "
        + json.dumps(
            {
                "episode": episode,
                "observation_size": int(observation.size),
                "grasp_distance": info["metrics"].get("grasp_distance"),
                "object_position": info["metrics"].get("object_position"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    normalized_hand = (
        2.0
        * (VALIDATED_CLOSED_HAND_POSITIONS - HAND_LOWER_BOUNDS)
        / (HAND_UPPER_BOUNDS - HAND_LOWER_BOUNDS)
        - 1.0
    )
    grasp_action = np.concatenate((normalized_hand, [-1.0]))
    lift_action = np.concatenate((normalized_hand, [1.0]))

    # One complete five-second grasp macro, one preload observation step, then
    # the independently validated eight-second lift.  Remaining idle macros
    # let the unchanged grip satisfy the three-second hold criterion.
    policy = [grasp_action, grasp_action, lift_action]
    policy.extend([grasp_action] * 6)

    final_info = info
    reward = 0.0
    for action in policy:
        _, reward, terminated, truncated, final_info = env.step(action)
        if terminated or truncated:
            break

    metrics = final_info.get("metrics", {})
    passed = final_info.get("termination_reason") == "success"
    summary = {
        "episode": episode,
        "episode_steps": final_info.get("episode_steps"),
        "termination_reason": final_info.get("termination_reason"),
        "height_gain": metrics.get("height_gain"),
        "grasp_distance": metrics.get("grasp_distance"),
        "hold_seconds": metrics.get("hold_seconds"),
        "finger_torque_deltas": metrics.get("finger_torque_deltas"),
        "last_reward": round(float(reward), 6),
    }
    if passed:
        print("RL STEP SMOKE PASSED: " + json.dumps(summary, sort_keys=True), flush=True)
    else:
        print("RL STEP SMOKE FAILED: " + json.dumps(summary, sort_keys=True), flush=True)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one or more consecutive cylinder RL episodes."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="number of reset/grasp/lift/hold episodes to run",
    )
    arguments, _ = parser.parse_known_args()
    if arguments.episodes <= 0:
        parser.error("--episodes must be positive")

    env = KcgCylinderEnv()
    try:
        passed = True
        for episode in range(1, arguments.episodes + 1):
            if not run_smoke(env, episode=episode):
                passed = False
                break
        return 0 if passed else 1
    except Exception as error:
        print(f"RL STEP SMOKE ERROR: {error}", flush=True)
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
