"""Read-only hardware telemetry; sampled in a cached background thread.

No training-library imports or random generators are used here. CPU utilization
is system-wide, GPU utilization is device-wide, process count is the worker tree.
"""
from __future__ import annotations

import csv
import ctypes
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time


def find_nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        for path in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe",
                     Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe"):
            if path.is_file():
                return str(path)
    return None


def _cpu_identity():
    name = platform.processor() or "CPU"
    physical = None
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            size = ctypes.c_ulong(0)
            call = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
            call(0, None, ctypes.byref(size))  # RelationProcessorCore
            if size.value:
                buffer = ctypes.create_string_buffer(size.value)
                if call(0, buffer, ctypes.byref(size)):
                    offset = physical = 0
                    while offset + 8 <= size.value:
                        record_size = ctypes.c_ulong.from_buffer(buffer, offset + 4).value
                        if record_size < 8:
                            break
                        physical += 1
                        offset += record_size
        elif Path("/proc/cpuinfo").exists():
            cores = set()
            for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
                fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                name = fields.get("model name", name)
                if "physical id" in fields and "core id" in fields:
                    cores.add((fields["physical id"], fields["core id"]))
            physical = len(cores) or None
    except (OSError, ValueError, AttributeError):
        pass
    return {"name": name, "logical_cores": os.cpu_count(), "physical_cores": physical}


def _cpu_ticks():
    if os.name == "nt":
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError("GetSystemTimes failed")
        value = lambda x: (x.dwHighDateTime << 32) | x.dwLowDateTime
        total = value(kernel) + value(user)
        return total, value(idle)
    if Path("/proc/stat").exists():
        values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        # guest and guest_nice are already included in user and nice.
        return sum(values[:8]), values[3] + (values[4] if len(values) > 4 else 0)
    raise OSError("此系统未提供支持的 CPU 计数器")


def _process_parents():
    if os.name == "nt":
        from ctypes import wintypes
        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
        kernel = ctypes.windll.kernel32
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateToolhelp32Snapshot(2, 0)
        if handle == ctypes.c_void_p(-1).value:
            raise OSError("Process snapshot unavailable")
        parents, entry = {}, Entry()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            more = kernel.Process32FirstW(handle, ctypes.byref(entry))
            while more:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                more = kernel.Process32NextW(handle, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(handle)
        return parents
    if Path("/proc").exists():
        parents = {}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                # A process name may itself contain spaces and parentheses.
                content = (path / "stat").read_text()
                fields = content[content.rfind(")") + 2:].split()
                parents[int(path.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
        return parents
    raise OSError("此系统未提供支持的进程树计数器")


def _tree_size(root_pid):
    if root_pid is None:
        return 0
    parents = _process_parents()
    if root_pid not in parents:
        return 0
    descendants = {root_pid}
    while True:
        more = {pid for pid, parent in parents.items() if parent in descendants}
        fresh = more - descendants
        if not fresh:
            return len(descendants)
        descendants.update(fresh)


def query_gpu(smi=None):
    smi = smi or find_nvidia_smi()
    if not smi:
        return {"available": False, "reason": "未找到 nvidia-smi；可能没有 NVIDIA GPU 或驱动尚未安装", "devices": []}
    cmd = [smi, "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,driver_version,temperature.gpu", "--format=csv,noheader,nounits"]
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2.5, **flags)
        if run.returncode:
            return {"available": False, "reason": (run.stderr.strip() or "nvidia-smi 无法查询设备")[:250], "devices": []}
        def numeric(value):
            try:
                return float(value.strip())
            except ValueError:
                return None
        devices = []
        for row in csv.reader(io.StringIO(run.stdout)):
            if len(row) != 8:
                continue
            devices.append({"index": int(row[0]), "name": row[1].strip(), "uuid": row[2].strip(),
                            "utilization_percent": numeric(row[3]), "memory_used_mib": numeric(row[4]),
                            "memory_total_mib": numeric(row[5]), "driver_version": row[6].strip(),
                            "temperature_c": numeric(row[7])})
        return {"available": bool(devices), "reason": None if devices else "nvidia-smi 没有返回设备", "devices": devices}
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return {"available": False, "reason": f"GPU 监测不可用：{error}", "devices": []}


class HardwareMonitor:
    def __init__(self, interval=3.0):
        self.interval = max(2.0, float(interval))
        self.identity = _cpu_identity()
        self.smi = find_nvidia_smi()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._worker_pid = None
        self._previous = None
        self._snapshot = {"sampled_at": None, "cpu": {**self.identity, "utilization_percent": None, "reason": "正在采样 CPU 计数器"},
                          "processes": {"worker_pid": None, "training_process_count": 0, "reason": None},
                          "gpu": {"available": False, "reason": "正在检测 GPU", "devices": []}}
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="stella-hardware", daemon=True)
        self.thread.start()
        return self

    def set_worker_pid(self, pid):
        with self._lock:
            self._worker_pid = pid

    def snapshot(self):
        with self._lock:
            # Fresh nested containers are produced on every sample; callers do not mutate them.
            return dict(self._snapshot)

    def _run(self):
        while not self._stop.is_set():
            cpu = {**self.identity, "utilization_percent": None, "reason": None}
            try:
                current = _cpu_ticks()
                if self._previous and current[0] > self._previous[0]:
                    total, idle = current[0] - self._previous[0], current[1] - self._previous[1]
                    cpu["utilization_percent"] = max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))
                else:
                    cpu["reason"] = "首次 CPU 计数器采样；约 3 秒后显示实测利用率"
                self._previous = current
            except (OSError, ValueError, IndexError) as error:
                cpu["reason"] = str(error)
            with self._lock:
                pid = self._worker_pid
            processes = {"worker_pid": pid, "training_process_count": None, "reason": None}
            try:
                processes["training_process_count"] = _tree_size(pid)
            except (OSError, ValueError, AttributeError) as error:
                processes["reason"] = str(error)
            result = {"sampled_at": time.time(), "interval_seconds": self.interval, "cpu": cpu,
                      "processes": processes, "gpu": query_gpu(self.smi)}
            with self._lock:
                self._snapshot = result
            self._stop.wait(self.interval)

    def close(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=3)
