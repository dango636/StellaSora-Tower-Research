"""Machine-local execution settings; never part of frozen learning config."""
from __future__ import annotations

import json
import os
from pathlib import Path

PROFILES = {
    'local_5070': {
        'STELLA_GPU_EVAL_BATCH': '8192',
        'STELLA_GPU_ROOT_BATCH': '128',
        'STELLA_GPU_SAMPLING_BATCH': '1024',
        'STELLA_GPU_INFERENCE_ROWS': '16384',
        'STELLA_FIT_DATA_RESIDENCY': 'auto',
        'STELLA_FIT_DATA_BUDGET_MB': '7168',
        'STELLA_FIT_MEMORY_RESERVE_MB': '3072',
        'STELLA_GPU_SINGLE_SYNC': '1',
        'STELLA_GPU_CUDA_GRAPH': '0',
        'STELLA_GPU_BOUND_LAUNCH': '1',
        'STELLA_GPU_CACHE_MB': '8192',
        'STELLA_EVAL_INPUT_CACHE_MB': '8192',
        'STELLA_EVAL_INPUT_CACHE_MODE': 'compact',
        'STELLA_EVAL_DISK_CACHE': '0',
    },
    'colab_blackwell96': {
        'STELLA_GPU_EVAL_BATCH': '32768',
        'STELLA_GPU_ROOT_BATCH': '512',
        'STELLA_GPU_SAMPLING_BATCH': '2048',
        'STELLA_GPU_INFERENCE_ROWS': '16384',
        'STELLA_FIT_DATA_RESIDENCY': 'auto',
        'STELLA_FIT_DATA_BUDGET_MB': '73728',
        'STELLA_FIT_MEMORY_RESERVE_MB': '8192',
        'STELLA_GPU_SINGLE_SYNC': '1',
        'STELLA_GPU_CUDA_GRAPH': '0',
        'STELLA_GPU_BOUND_LAUNCH': '1',
        # The legacy 10000 MiB compute limit was intended for 12 GiB desktop
        # devices; inheriting it repeatedly freed useful allocator blocks here.
        # This is a ceiling, not a reservation. Actual visible/free memory and
        # the phase-specific reserve still gate every allocation at runtime.
        'STELLA_GPU_CACHE_MB': '73728',
        # All 4960*256 immutable initial inputs fit comfortably; reuse public
        # state/belief arrays as well as MT states across the six candidates.
        'STELLA_EVAL_INPUT_CACHE_MB': '16384',
        'STELLA_EVAL_INPUT_CACHE_MODE': 'full',
        'STELLA_EVAL_DISK_CACHE': '1',
        'STELLA_EVAL_DISK_CACHE_MB': '16384',
        'STELLA_EVAL_DISK_RESERVE_MB': '16384',
    },
}


def profile_name(explicit=None):
    if explicit:
        name = explicit
    else:
        name = os.environ.get('STELLA_HARDWARE_PROFILE')
        path = Path(__file__).resolve().parents[1] / 'hardware_profile.json'
        if not name and path.exists():
            name = json.loads(path.read_text(encoding='utf-8'))['profile']
    if name and name not in PROFILES:
        raise ValueError('Unknown hardware profile: ' + str(name))
    return name


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def memory_limit_summary(*, allocator_cache_mb=None, device_free_bytes=None, device_total_bytes=None):
    """Describe phase budgets without treating their ceilings as allocations."""
    def configured_bytes(key):
        value = os.environ.get(key)
        return None if value is None else int(value)*2**20
    cache = (int(allocator_cache_mb)*2**20 if allocator_cache_mb is not None
             else configured_bytes("STELLA_GPU_CACHE_MB"))
    profile = profile_name()
    reserve = (2048 if profile == "colab_blackwell96" else 768)*2**20 if profile else None
    return dict(
        rollout_allocator_cache_ceiling_bytes=cache,
        rollout_activation_reserve_bytes=reserve,
        fit_resident_data_ceiling_bytes=configured_bytes("STELLA_FIT_DATA_BUDGET_MB"),
        fit_free_memory_reserve_bytes=configured_bytes("STELLA_FIT_MEMORY_RESERVE_MB"),
        evaluation_host_input_cache_ceiling_bytes=configured_bytes("STELLA_EVAL_INPUT_CACHE_MB"),
        evaluation_disk_input_cache_ceiling_bytes=configured_bytes("STELLA_EVAL_DISK_CACHE_MB")
            if os.environ.get("STELLA_EVAL_DISK_CACHE") == "1" else 0,
        evaluation_disk_free_reserve_bytes=configured_bytes("STELLA_EVAL_DISK_RESERVE_MB")
            if os.environ.get("STELLA_EVAL_DISK_CACHE") == "1" else 0,
        device_free_bytes_at_snapshot=None if device_free_bytes is None else int(device_free_bytes),
        device_total_bytes=None if device_total_bytes is None else int(device_total_bytes),
        budget_semantics="ceilings only; rollout service and fitting have separate lifetimes; do not add them as live allocations",
        allocation_guard="rollout uses actual free plus allocator slack; fitting uses min(data cap, actual free minus fit reserve)")


def apply_profile(name=None, *, override=False):
    """Set execution defaults before worker creation; explicit env wins normally."""
    name = profile_name(name)
    if not name:
        return {'profile': None, 'environment': {}, 'learning_config_changed': False}
    values = dict(PROFILES[name])
    values['STELLA_CPU_WORKERS'] = str(min(8 if name == 'local_5070' else 16, available_cpus()))
    for key, value in values.items():
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    os.environ['STELLA_HARDWARE_PROFILE'] = name
    return {'profile': name, 'environment': {key: os.environ[key] for key in values},
            'learning_config_changed': False,
            'memory_limits': memory_limit_summary(),
            'precision': 'FP32; learning batch size and objective unchanged'}


def runtime_workers(configured):
    value = int(os.environ.get('STELLA_CPU_WORKERS', configured))
    if value < 1:
        raise ValueError('STELLA_CPU_WORKERS must be positive')
    return min(value, available_cpus())


def validate_gpu_profile(name, total_memory_bytes, device_name=''):
    """Reject accidental use of a 96 GB budget on a fractional/other GPU."""
    if name not in PROFILES:
        raise ValueError('Unknown hardware profile: ' + str(name))
    gib = float(total_memory_bytes) / 1024**3
    minimum = 80 if name == 'colab_blackwell96' else 10
    if gib < minimum:
        raise ValueError(f'{name} needs at least {minimum} GiB visible GPU memory; '
                         f'actual {device_name}: {gib:.2f} GiB. Select the matching runtime/profile.')
    return {'profile': name, 'device_name': device_name, 'visible_memory_gib': gib}
