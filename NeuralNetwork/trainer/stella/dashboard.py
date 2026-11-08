"""Small, local-only dashboard. No external web assets or framework required."""
from __future__ import annotations

import csv
import io
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _tail_jsonl(path: Path, count: int, max_bytes: int = 8_000_000):
    """Bound both memory and I/O for a training process that may run for months."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            offset = max(0, size - max_bytes)
            stream.seek(offset)
            raw = stream.read()
        lines = raw.splitlines()
        if offset and lines:
            lines = lines[1:]
        rows = []
        for line in lines[-count:]:
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                pass  # An in-progress append is retried on the next refresh.
        return rows
    except OSError:
        return []


class Dashboard:
    def __init__(self, run_dir, host="127.0.0.1", port=0):
        if host != "127.0.0.1":
            raise ValueError("Dashboard must bind to 127.0.0.1")
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir = Path(__file__).resolve().parent.parent / "web"
        self.token = secrets.token_urlsafe(32)
        self.process = {"running": None, "pid": None, "exit_code": None}
        from .hardware import HardwareMonitor
        self.hardware = HardwareMonitor(interval=3)
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, code, payload, content_type="application/json; charset=utf-8", headers=None):
                if not isinstance(payload, bytes):
                    payload = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def _local(self):
                return self.headers.get("Host", "") in {owner.address, "localhost:" + str(owner.server.server_port)}

            def do_GET(self):
                if not self._local():
                    return self._send(403, {"error": "Local host only"})
                path = urlsplit(self.path).path
                if path == "/api/session":
                    return self._send(200, {"token": owner.token, "run_dir": str(owner.run_dir)})
                if path == "/api/status":
                    status = _read_json(owner.run_dir / "status.json", {"status": "bootstrap", "message": "正在启动训练进程…"})
                    with owner._lock:
                        status["process"] = dict(owner.process)
                    status["hardware"] = owner.hardware.snapshot()
                    status["gpu_setup"] = _read_json(owner.run_dir / "gpu_setup.json", {})
                    status["pause_requested"] = (owner.run_dir / "pause.flag").exists()
                    status["stop_requested"] = (owner.run_dir / "stop.flag").exists()
                    status["server_time"] = time.time()
                    return self._send(200, status)
                if path == "/api/history":
                    return self._send(200, {"rows": _tail_jsonl(owner.run_dir / "history.jsonl", 2000), "limit": 2000})
                if path == "/api/presets":
                    return self._send(200, _read_json(owner.run_dir / "preset_results.json", {"round": None, "presets": []}))
                if path == "/api/events":
                    return self._send(200, {"rows": _tail_jsonl(owner.run_dir / "events.jsonl", 120, 400_000)})
                if path == "/api/history.csv":
                    rows = _tail_jsonl(owner.run_dir / "history.jsonl", 2000)
                    columns = ["round", "teacher_id", "candidate_id", "best_model_id", "promoted", "candidate_mean", "best_candidate_mean", "composite_teacher_mean", "champion_updated_count", "distinct_teacher_count", "primary_delta", "ci_low", "ci_high", "elapsed_seconds"]
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                    return self._send(200, ("\ufeff" + buf.getvalue()).encode("utf-8"), "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="training_history.csv"'})
                files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"), "/assets/stella-ui.woff": ("assets/stella-ui.woff", "font/woff")}
                if path in files:
                    filename, content_type = files[path]
                    try:
                        return self._send(200, (owner.web_dir / filename).read_bytes(), content_type)
                    except OSError:
                        return self._send(500, {"error": "Dashboard asset is missing"})
                return self._send(404, {"error": "Not found"})

            def do_POST(self):
                origin = self.headers.get("Origin")
                if not self._local() or (origin and origin not in {owner.url.rstrip("/"), "http://localhost:" + str(owner.server.server_port)}):
                    return self._send(403, {"error": "Origin rejected"})
                if not secrets.compare_digest(self.headers.get("X-Stella-Token", ""), owner.token):
                    return self._send(403, {"error": "Token rejected"})
                if urlsplit(self.path).path != "/api/control":
                    return self._send(404, {"error": "Not found"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("Invalid body length")
                    action = json.loads(self.rfile.read(size)).get("action")
                    if action not in ("pause", "resume", "stop", "restart"):
                        raise ValueError("Unknown control action")
                    if action == "restart":
                        if owner.process.get("running") is False:
                            (owner.run_dir / "restart.flag").touch()
                            return self._send(200, {"ok": True, "action": action})
                        return self._send(409, {"error": "任务仍在运行。"})
                    if owner.process.get("running") is False:
                        return self._send(409, {"error": "任务已结束。点击“继续运行”即可恢复。"})
                    if action == "pause":
                        (owner.run_dir / "pause.flag").touch()
                    elif action == "resume":
                        (owner.run_dir / "pause.flag").unlink(missing_ok=True)
                    else:
                        (owner.run_dir / "stop.flag").touch()
                        (owner.run_dir / "pause.flag").unlink(missing_ok=True)
                    return self._send(200, {"ok": True, "action": action})
                except (ValueError, OSError) as error:
                    return self._send(400, {"error": str(error)})

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.address = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.address}/"
        self.thread = None

    def start(self):
        self.hardware.start()
        self.thread = threading.Thread(target=self.server.serve_forever, name="stella-dashboard", daemon=True)
        self.thread.start()
        return self.url

    def set_process(self, *, running, pid=None, exit_code=None, stage="training"):
        with self._lock:
            self.process = {"running": running, "pid": pid, "exit_code": exit_code, "stage": stage}
        self.hardware.set_worker_pid(pid if running and stage == "training" else None)

    def close(self):
        self.hardware.close()
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=3)
