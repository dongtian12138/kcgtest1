#!/usr/bin/env python3

"""Fail closed unless Isaac RL uses the verified CUDA PyTorch runtime."""

import importlib.metadata
import json
import sys

import cloudpickle
import gymnasium
import numpy
import pandas
import stable_baselines3
import torch


EXPECTED_VERSIONS = {
    "cloudpickle": "3.1.2",
    "gymnasium": "1.2.3",
    "isaacsim": "6.0.1.0",
    "pandas": "3.0.5",
    "stable_baselines3": "2.7.1",
}


def main():
    versions = {
        "cloudpickle": cloudpickle.__version__,
        "gymnasium": gymnasium.__version__,
        "isaacsim": importlib.metadata.version("isaacsim"),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }
    if versions["torch"] != "2.11.0+cu128":
        raise RuntimeError(
            "Isaac GPU torch changed: " + versions["torch"]
        )
    for package, expected in EXPECTED_VERSIONS.items():
        if versions[package] != expected:
            raise RuntimeError(
                f"unexpected {package} version: {versions[package]}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    value = torch.linspace(
        -1.0, 1.0, 4096, device=device, requires_grad=True
    )
    loss = (value.square().mean() + value.sin().mean())
    loss.backward()
    if value.grad is None or not torch.isfinite(value.grad).all():
        raise RuntimeError("CUDA backward pass produced invalid gradients")
    versions.update(
        {
            "cuda_backward_finite": True,
            "gpu": torch.cuda.get_device_name(device),
            "python": sys.version.split()[0],
        }
    )
    print(json.dumps(versions, sort_keys=True), flush=True)
    print("ISAAC GPU RL RUNTIME PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
