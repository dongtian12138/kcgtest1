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


def test_gpu_solver_with_explicit_host_readback_is_not_misclassified_as_cpu():
    host = SimpleNamespace(backend='numpy', device='cpu')
    gpu_scene = context('cpu', False, 'GPU')
    gpu_scene.is_gpu_dynamics_enabled = lambda: True
    assert not physics_backend_record(host, gpu_scene, 'cuda:0')['pass']
    audited = physics_backend_record(host, gpu_scene, 'cuda:0',
        gpu_host_readback=True, observed_suppress_readback=False)
    assert audited['pass'] and audited['gpu_backend_pass']
    assert not audited['cpu_backend_pass']
    assert audited['world_device']=='cpu' and audited['gpu_dynamics_enabled']
    assert not physics_backend_record(host, gpu_scene, 'cuda:0',
        gpu_host_readback=True, observed_suppress_readback=True)['pass']
    assert not physics_backend_record(host, context('cpu',False,'MBP'), 'cuda:0',
        gpu_host_readback=True, observed_suppress_readback=False)['pass']
