"""A requested CPU experiment must never be certified as GPU, or vice versa."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
from carts_v2.engine_health import physics_backend_record, physics_world_parameters


def context(device, gpu, broadphase):
    return SimpleNamespace(device=device, use_gpu_sim=gpu, use_gpu_pipeline=gpu,
                           is_gpu_dynamics_enabled=lambda: gpu,
                           get_broadphase_type=lambda: broadphase)


def test_backend_identity_matches_requested_execution_device():
    cpu = SimpleNamespace(backend="numpy", device="cpu")
    gpu = SimpleNamespace(backend="torch", device="cuda:0")
    cpu_context, gpu_context = context("cpu", False, "MBP"), context("cuda:0", True, "GPU")
    actual_cpu = physics_backend_record(cpu, cpu_context, "cpu")
    assert actual_cpu["pass"] and actual_cpu["cpu_backend_pass"]
    assert actual_cpu["gpu_backend_pass"] is False
    assert physics_backend_record(gpu, gpu_context, "cuda:0")["pass"]
    assert not physics_backend_record(cpu, cpu_context, "cuda:0")["pass"]
    assert not physics_backend_record(gpu, gpu_context, "cpu")["pass"]
    assert not physics_backend_record(cpu, gpu_context, "cpu")["pass"]
    assert not physics_backend_record(gpu, cpu_context, "cuda:0")["pass"]
    resources = {"gpu_found_lost_aggregate_pairs_capacity": 4096,
                 "gpu_total_aggregate_pairs_capacity": 8192}
    assert physics_world_parameters(resources, "cpu")["sim_params"]["use_gpu_pipeline"] is False
    assert physics_world_parameters(resources)["sim_params"]["use_gpu_pipeline"] is True
