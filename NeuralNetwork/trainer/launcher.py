"""Launch the autonomous worker and its local progress window."""
from __future__ import annotations

import argparse
import json
import logging
import math
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
# Keep a tiny inference batch from spawning a BLAS thread per CPU core.
# Set these before either this process or its child imports NumPy.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(ROOT))


class InstanceLock:
    def __init__(self, path):
        self.stream = Path(path).open("a+b")
        self.stream.seek(0, 2)
        if not self.stream.tell():
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)

    def acquire(self):
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    def close(self):
        self.stream.close()


class BoundedLog:
    """Drain a child pipe continuously while bounding long-running disk usage."""
    def __init__(self, path, max_bytes=8*1024*1024, backups=3):
        self.logger = logging.Logger("stella-child-log", level=logging.INFO)
        self.handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.handler)
        self.thread = None

    def write(self, text):
        self.logger.info(str(text).rstrip("\r\n"))

    def start(self, stream):
        def drain():
            try:
                for chunk in iter(lambda: stream.readline(65536), ""):
                    self.write(chunk)
            finally:
                stream.close()
        self.thread = threading.Thread(target=drain, name="stella-log-writer", daemon=True)
        self.thread.start()
        return self

    def close(self):
        if self.thread:
            self.thread.join(timeout=3)
        self.handler.flush()
        self.handler.close()


def read_runtime_probe(path, returncode):
    """Accept readiness only from a successfully completed fresh probe."""
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Runtime report must be an object")
    except (OSError, ValueError) as error:
        return {"state": "unavailable", "ok": False,
                "reason": "Runtime probe produced no valid report: " + str(error)}
    if result.get("state") == "ready" and (
            returncode != 0 or result.get("ok") is not True
            or not isinstance(result.get("python_executable"), str)
            or not result["python_executable"]):
        return {"state": "unavailable", "ok": False,
                "reason": f"Runtime readiness rejected (probe exit {returncode})"}
    return result


def open_existing(run_dir, open_browser=True):
    for _ in range(20):
        try:
            meta = json.loads((run_dir / "launcher.json").read_text(encoding="utf-8"))
            url = meta["url"]
            with urllib.request.urlopen(url + "api/status", timeout=2) as response:
                status = json.load(response)
            if status.get("process", {}).get("running") is False:
                with urllib.request.urlopen(url + "api/session", timeout=2) as response:
                    session = json.load(response)
                req = urllib.request.Request(url + "api/control", data=b'{"action":"restart"}', headers={"Content-Type": "application/json", "X-Stella-Token": session["token"]}, method="POST")
                with urllib.request.urlopen(req, timeout=2) as response:
                    response.read()
            print("Existing task:", url, flush=True)
            if open_browser:
                webbrowser.open(url)
            return 0
        except (OSError, ValueError, KeyError):
            time.sleep(0.15)
    print("Another launcher owns this task. Please use its existing console window.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stella Sora continuous trainer")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "catalogue4960"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--once", action="store_true", help="Exit when the worker exits (for scripts/tests).")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--hours", type=float, default=0, help="Optional worker session limit in hours; default 0: unlimited.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Offline CPU mode; never install GPU packages.")
    parser.add_argument("--retry-gpu", action="store_true", help="Retry a previously failed optional GPU setup.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--profile", choices=["local_5070", "colab_blackwell96"])
    args = parser.parse_args(argv)
    from stella.hardware_profiles import apply_profile
    apply_profile(args.profile)
    if not math.isfinite(args.hours) or args.hours < 0:
        parser.error("--hours must be finite and nonnegative")
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    local_config = run_dir / "config.json"
    config = (Path(args.config).expanduser().resolve() if args.config else
              local_config if local_config.is_file() else ROOT / "config.json")
    lock = InstanceLock(run_dir / "launcher.lock")
    if not lock.acquire():
        lock.close()
        return open_existing(run_dir, not args.no_browser)
    dashboard = None
    child = None
    worker_log = None
    interrupted = False

    def request_stop(_signal=None, _frame=None):
        nonlocal interrupted
        if interrupted:
            print("Still waiting for a safe save point. The dashboard remains available.", flush=True)
            return
        interrupted = True
        (run_dir / "stop.flag").touch()
        (run_dir / "pause.flag").unlink(missing_ok=True)
        print("Stop requested. Waiting for a safe save point...", flush=True)

    try:
        import numpy as np
        from stella.dashboard import Dashboard
        # Fail visibly before spawning if the runtime is incomplete.
        if not np.isfinite(np.ones((2, 2), dtype=np.float32) @ np.ones((2, 2), dtype=np.float32)).all():
            raise RuntimeError("NumPy self-check failed")
        if not config.is_file():
            raise FileNotFoundError(f"Missing training config: {config}")
        if not local_config.exists():
            from stella.io_utils import atomic_json
            atomic_json(local_config, json.loads(config.read_text(encoding="utf-8")))
            config = local_config
        dashboard = Dashboard(run_dir, port=args.port)
        url = dashboard.start()
        (run_dir / "launcher.json").write_text(json.dumps({"pid": os.getpid(), "url": url, "run_dir": str(run_dir)}, ensure_ascii=False), encoding="utf-8")
        print("\nStella Sora Continuous", flush=True)
        print("Dashboard:", url, flush=True)
        print("Run folder:", run_dir, flush=True)
        print("Closing the browser tab does not stop training.", flush=True)
        print("Use the dashboard's Save and Stop button to checkpoint and stop.\n", flush=True)
        signal.signal(signal.SIGINT, request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, request_stop)
        if not args.no_browser:
            webbrowser.open(url)
        config_data = json.loads(config.read_text(encoding="utf-8"))
        configured_backend = str(config_data.get("compute", {}).get("backend", config_data.get("training", {}).get("backend", "auto"))).lower()
        force_cpu = args.cpu or configured_backend in {"numpy", "cpu"} or os.environ.get("STELLA_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        while True:
            for filename in ("pause.flag", "stop.flag", "restart.flag"):
                (run_dir / filename).unlink(missing_ok=True)
            # The dashboard and hardware monitor remain responsive throughout setup.
            setup_cmd = [sys.executable, "-u", "-m", "stella.gpu_setup", "--run-dir", str(run_dir)]
            if force_cpu:
                setup_cmd.append("--cpu")
            if args.retry_gpu:
                setup_cmd.append("--retry")
            dashboard.set_process(running=True, stage="gpu_setup")
            creation = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            setup_log = BoundedLog(run_dir / "gpu_setup.log", max_bytes=4*1024*1024, backups=2)
            setup_report = run_dir / "gpu_setup.json"
            # A crashed/killed probe must never reuse yesterday's ready report.
            setup_report.unlink(missing_ok=True)
            try:
                child = subprocess.Popen(setup_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(),
                                         text=True, encoding="utf-8", errors="replace", bufsize=1, **creation)
                setup_log.start(child.stdout)
                while child.poll() is None:
                    time.sleep(.25)
            finally:
                setup_log.close()
            setup = read_runtime_probe(setup_report, child.returncode)
            if interrupted or (run_dir / "stop.flag").exists() or setup.get("state") == "cancelled":
                dashboard.set_process(running=False, exit_code=0, stage="gpu_setup")
                if args.once or interrupted:
                    return 0
                print("GPU preparation stopped. Click Continue to resume.", flush=True)
                while not (run_dir / "restart.flag").exists() and not interrupted:
                    time.sleep(.25)
                if interrupted:
                    return 0
                continue
            interpreter = setup.get("python_executable", sys.executable) if setup.get("state") == "ready" else sys.executable
            if configured_backend == "cuda" and not force_cpu and setup.get("state") != "ready":
                raise RuntimeError("CUDA was explicitly configured, but the existing environment failed verification: "
                                   + str(setup.get("reason") or setup.get("message") or "unknown runtime error"))
            worker_env = os.environ.copy()
            resource_override = run_dir / "runtime_resources.json"
            if resource_override.exists():
                worker_env.update(json.loads(resource_override.read_text(encoding="utf-8"))["environment"])
            if setup.get("state") != "ready":
                worker_env["STELLA_FORCE_CPU"] = "1"
                print("Training backend: CPU. " + str(setup.get("reason") or setup.get("message") or ""), flush=True)
            else:
                worker_env.pop("STELLA_FORCE_CPU", None)
                print("CUDA backend verified: " + str(setup.get("device_name")) + "; Python: " + str(interpreter), flush=True)
            cmd = [str(interpreter), "-u", "-m", "stella.engine", "--run-dir", str(run_dir), "--config", str(config)]
            cmd.extend(["--hours", str(args.hours)])
            if args.rounds is not None:
                cmd.extend(["--rounds", str(args.rounds)])
            if args.smoke:
                cmd.append("--smoke")
            worker_log = BoundedLog(run_dir / "worker.log")
            worker_log.write("\n=== Worker launch " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=worker_env,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, **creation)
            worker_log.start(child.stdout)
            dashboard.set_process(running=True, pid=child.pid)
            while child.poll() is None:
                time.sleep(0.25)
            code = child.returncode
            worker_log.close()
            worker_log = None
            dashboard.set_process(running=False, pid=child.pid, exit_code=code)
            print(f"Worker ended (exit code {code}). Results: {url}", flush=True)
            if code:
                print("Details are in", run_dir / "worker.log", flush=True)
            if args.once or interrupted:
                return code
            print("Click Continue in the dashboard, or double-click the launcher, to resume.", flush=True)
            while not (run_dir / "restart.flag").exists() and not interrupted:
                time.sleep(0.25)
            if interrupted:
                return code
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        (run_dir / "launcher_error.log").write_text(error, encoding="utf-8")
        return 1
    finally:
        if child is not None and child.poll() is None:
            (run_dir / "stop.flag").touch()
            (run_dir / "pause.flag").unlink(missing_ok=True)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("Worker is still saving. See worker.log before closing its process.", flush=True)
        if worker_log:
            worker_log.close()
        if dashboard:
            dashboard.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
