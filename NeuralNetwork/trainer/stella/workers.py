"""Spawn-safe per-process model cache; BLAS threading is set by the entrypoint."""
from .io_utils import Control
from .policy import QPolicy
from collections import OrderedDict
from pathlib import Path

POLICIES = {}
CONTROL = None
_MODEL_CACHE = OrderedDict()
INFERENCE = None
AFFINITY = None


def initialize(run_dir, model_paths, inference=None, affinity=None):
    global POLICIES, CONTROL, _MODEL_CACHE, INFERENCE, AFFINITY
    from .cpu_affinity import bind_worker
    AFFINITY = bind_worker(affinity, run_dir)
    CONTROL = Control(run_dir)
    POLICIES = {}
    _MODEL_CACHE = OrderedDict()
    INFERENCE = None
    if inference is not None:
        from .label_inference import InferenceClient
        INFERENCE = InferenceClient(inference, check)
    ensure_models(model_paths)


def ensure_models(model_paths):
    """Reuse immutable CPU models across stages and rounds, bounded to four."""
    for name, raw in model_paths.items():
        path = Path(raw)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, name if INFERENCE else None)
        if key not in _MODEL_CACHE:
            policy = QPolicy.load(path, name)
            if INFERENCE is not None and name in INFERENCE.spec["model_checksums"]:
                from .label_inference import RemoteQModel
                policy = QPolicy(RemoteQModel(INFERENCE, policy.model, name), name)
            _MODEL_CACHE[key] = policy
        _MODEL_CACHE.move_to_end(key)
        POLICIES[name] = _MODEL_CACHE[key]
    while len(_MODEL_CACHE) > 4:
        _MODEL_CACHE.popitem(last=False)
    # Drop obsolete aliases as well, otherwise they retain evicted weights.
    for name in tuple(POLICIES):
        if name not in model_paths:
            del POLICIES[name]


def invoke(function, task, model_paths):
    check()
    ensure_models(model_paths)
    return function(task)


def check():
    if CONTROL is not None:
        CONTROL.check()
