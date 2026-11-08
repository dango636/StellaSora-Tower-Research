"""Physical-core-aware affinity for simulation workers on Windows x64.

Discover SMT siblings through RelationProcessorCore, never by assuming that
logical CPU 0..N-1 are different cores. Only pool initializers apply affinity;
the coordinator, GPU service, and serial execution keep their original masks.
"""
from __future__ import annotations

import ctypes
import multiprocessing
import os
from pathlib import Path
import struct
import uuid

from .io_utils import atomic_json


def _api():
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise NotImplementedError("Physical-core affinity currently requires Windows x64")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.argtypes = []
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetActiveProcessorGroupCount.argtypes = []
    api.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
    api.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                                     ctypes.POINTER(ctypes.c_ulong)]
    api.GetLogicalProcessorInformationEx.restype = ctypes.c_int
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.POINTER(ctypes.c_size_t)]
    api.GetProcessAffinityMask.restype = ctypes.c_int
    api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    api.SetProcessAffinityMask.restype = ctypes.c_int
    return api


def process_mask():
    api = _api()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return allowed.value


def set_process_mask(mask):
    if type(mask) is not int or mask <= 0:
        raise ValueError("CPU affinity mask must be a positive integer")
    api = _api()
    if not api.SetProcessAffinityMask(api.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
    actual = process_mask()
    if actual != mask:
        raise RuntimeError("Windows did not apply the requested worker affinity")
    return actual


def parse_core_records(data):
    """Parse x64 SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX core records."""
    cores, offset, occupied = [], 0, set()
    while offset < len(data):
        if len(data)-offset < 8:
            raise ValueError("Truncated CPU topology header")
        relation, size = struct.unpack_from("<II", data, offset)
        if size < 48 or offset+size > len(data) or relation != 0:
            raise ValueError("Invalid processor-core topology record")
        groups = struct.unpack_from("<H", data, offset+30)[0]
        if groups != 1:
            raise ValueError("A physical core must belong to exactly one processor group")
        mask, group = struct.unpack_from("<QH", data, offset+32)
        if group != 0:
            raise NotImplementedError("Multiple processor groups require a different affinity API")
        cpus = [i for i in range(64) if mask & (1 << i)]
        if not cpus or occupied.intersection(cpus):
            raise ValueError("Empty or overlapping processor-core masks")
        occupied.update(cpus)
        cores.append({"physical_core": len(cores), "logical_cpus": cpus})
        offset += size
    if not cores:
        raise ValueError("Windows returned no physical cores")
    return cores


def topology():
    api = _api()
    if api.GetActiveProcessorGroupCount() != 1:
        raise NotImplementedError("Only single-processor-group hosts are supported")
    size = ctypes.c_ulong()
    api.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    if not size.value or ctypes.get_last_error() != 122:  # ERROR_INSUFFICIENT_BUFFER
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not api.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"cores": parse_core_records(buffer.raw[:size.value]), "allowed_mask": process_mask()}


def select_slots(cores, allowed_mask, count):
    if count < 1:
        raise ValueError("Worker count must be positive")
    available = [(core["physical_core"], [cpu for cpu in core["logical_cpus"]
                                         if allowed_mask & (1 << cpu)]) for core in cores]
    # First use one logical CPU from every physical core, then SMT siblings.
    slots = []
    for sibling in range(max((len(cpus) for _, cpus in available), default=0)):
        for physical, cpus in available:
            if sibling < len(cpus):
                cpu = cpus[sibling]
                slots.append({"physical_core": physical, "logical_cpu": cpu, "mask": 1 << cpu})
    if count > len(slots):
        raise ValueError("More workers requested than available logical CPUs")
    return slots[:count]


def create_plan(count, run_dir=None):
    """Allocate a fresh shared slot counter for this pool, without pinning its parent."""
    record = {"pool_id": uuid.uuid4().hex, "worker_count": count}
    try:
        detected = topology()
        slots = select_slots(detected["cores"], detected["allowed_mask"], count)
        record.update(mode="physical_core_first", topology=detected, slots=slots)
    except (OSError, NotImplementedError, ValueError) as exc:
        record.update(mode="unbound", reason=str(exc))
        if run_dir is not None:
            atomic_json(Path(run_dir) / "worker_affinity.json", record)
        return None
    if run_dir is not None:
        atomic_json(Path(run_dir) / "worker_affinity.json", record)
    record["next_slot"] = multiprocessing.get_context("spawn").Value("i", 0)
    return record


def bind_worker(plan, run_dir):
    if plan is None:
        return None
    with plan["next_slot"].get_lock():
        slot = plan["next_slot"].value
        plan["next_slot"].value += 1
    if slot >= len(plan["slots"]):
        raise RuntimeError("No free physical-core affinity slot in this worker pool")
    assigned = plan["slots"][slot]
    record = {"pool_id": plan["pool_id"], "slot": slot, "pid": os.getpid(),
              **assigned, "actual_mask": set_process_mask(assigned["mask"])}
    atomic_json(Path(run_dir) / "worker_affinity" / f"worker_{slot:02d}.json", record)
    return record
