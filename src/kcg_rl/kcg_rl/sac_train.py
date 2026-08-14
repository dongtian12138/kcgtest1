"""GPU SAC training and evaluation entry point for the cylinder task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import stable_baselines3
import torch

from .cylinder_env import GymnasiumKcgCylinderEnv


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, smoke-test, or evaluate SAC on the KCG cylinder task."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "train", "evaluate"),
        default="smoke",
    )
    parser.add_argument("--timesteps", type=positive_integer)
    parser.add_argument("--episodes", type=positive_integer, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", default="artifacts/kcg_rl/cylinder_sac"
    )
    parser.add_argument("--model-path")
    parser.add_argument(
        "--checkpoint-frequency", type=positive_integer, default=500
    )
    return parser.parse_args(argv)


def require_training_device(requested_device: str) -> torch.device:
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return device


def make_environment() -> Monitor:
    # Training CLI arguments are not ROS arguments, so initialize rclpy with an
    # empty argument list.  ROS topics and parameters still come from the
    # already-running cylinder launch.
    return Monitor(GymnasiumKcgCylinderEnv(ros_args=[]))


def model_device(model: SAC) -> str:
    return str(next(model.actor.parameters()).device)


def runtime_metadata(device: torch.device) -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }


def new_model(
    env: Monitor,
    arguments: argparse.Namespace,
    device: torch.device,
) -> SAC:
    smoke = arguments.mode == "smoke"
    return SAC(
        "MlpPolicy",
        env,
        device=device,
        seed=arguments.seed,
        verbose=1,
        learning_starts=0 if smoke else 100,
        buffer_size=128 if smoke else 100_000,
        batch_size=2 if smoke else 128,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64] if smoke else [256, 256]},
    )


def train(arguments: argparse.Namespace, device: torch.device) -> int:
    timesteps = arguments.timesteps
    if timesteps is None:
        timesteps = 2 if arguments.mode == "smoke" else 10_000

    output_directory = Path(arguments.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = make_environment()
    try:
        if arguments.model_path:
            model = SAC.load(
                str(Path(arguments.model_path).expanduser()),
                env=environment,
                device=device,
            )
        else:
            model = new_model(environment, arguments, device)

        callback = None
        if arguments.mode == "train":
            checkpoint_directory = output_directory / "checkpoints"
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            callback = CheckpointCallback(
                save_freq=arguments.checkpoint_frequency,
                save_path=str(checkpoint_directory),
                name_prefix="cylinder_sac",
                save_replay_buffer=True,
            )

        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=not bool(arguments.model_path),
        )
        model_name = (
            "cylinder_sac_smoke" if arguments.mode == "smoke" else "cylinder_sac"
        )
        model_path = output_directory / model_name
        model.save(str(model_path))

        replay_size = int(model.replay_buffer.size())
        metadata = runtime_metadata(device)
        metadata.update(
            {
                "mode": arguments.mode,
                "requested_timesteps": timesteps,
                "model_timesteps": int(model.num_timesteps),
                "replay_size": replay_size,
                "actor_device": model_device(model),
                "model_path": str(model_path.with_suffix(".zip")),
            }
        )
        metadata_path = output_directory / f"{model_name}_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        passed = (
            model.num_timesteps >= timesteps
            and replay_size > 0
            and (
                device.type != "cuda"
                or model_device(model).startswith("cuda")
            )
        )
        prefix = (
            "SAC TRAIN SMOKE PASSED"
            if arguments.mode == "smoke"
            else "SAC TRAINING COMPLETED"
        )
        print(prefix + ": " + json.dumps(metadata, sort_keys=True), flush=True)
        return 0 if passed else 1
    finally:
        environment.close()


def evaluate(arguments: argparse.Namespace, device: torch.device) -> int:
    if not arguments.model_path:
        raise ValueError("--model-path is required in evaluate mode")
    environment = make_environment()
    try:
        model = SAC.load(
            str(Path(arguments.model_path).expanduser()),
            env=environment,
            device=device,
        )
        results = []
        successes = 0
        for episode in range(1, arguments.episodes + 1):
            observation, _ = environment.reset(seed=arguments.seed + episode)
            terminated = False
            truncated = False
            episode_return = 0.0
            final_info: Dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = (
                    environment.step(action)
                )
                episode_return += float(reward)
            reason = final_info.get("termination_reason", "")
            successes += int(reason == "success")
            results.append(
                {
                    "episode": episode,
                    "return": round(episode_return, 6),
                    "termination_reason": reason,
                    "height_gain": final_info.get("metrics", {}).get("height_gain"),
                }
            )
        summary = {
            "episodes": arguments.episodes,
            "successes": successes,
            "success_rate": successes / arguments.episodes,
            "actor_device": model_device(model),
            "results": results,
        }
        print("SAC EVALUATION: " + json.dumps(summary, sort_keys=True), flush=True)
        return 0
    finally:
        environment.close()


def main(argv: Optional[List[str]] = None) -> int:
    arguments = parse_arguments(argv)
    device = require_training_device(arguments.device)
    metadata = runtime_metadata(device)
    print("RL TRAINING RUNTIME: " + json.dumps(metadata, sort_keys=True), flush=True)
    if arguments.mode == "evaluate":
        return evaluate(arguments, device)
    return train(arguments, device)


if __name__ == "__main__":
    sys.exit(main())
