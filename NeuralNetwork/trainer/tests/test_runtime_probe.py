import json
import sys

from stella import gpu_setup
from launcher import read_runtime_probe


def test_managed_colab_interpreter_outside_source_tree_is_accepted(tmp_path, monkeypatch):
    external_python = "/usr/local/bin/python3"
    monkeypatch.setattr(gpu_setup, "probe_current_runtime", lambda: {
        "ok": True, "python_executable": external_python, "backend": "torch_cuda",
        "nvrtc_execution_verified": True, "packages_installed": False})
    assert gpu_setup.main(["--run-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "gpu_setup.json").read_text())
    assert report["state"] == "ready"
    assert report["python_executable"] == external_python
    assert report["nvrtc_execution_verified"]
    assert not report["packages_installed"]
    assert {p.name for p in tmp_path.iterdir()} == {"gpu_setup.json"}


def test_missing_cuda_is_visible_and_does_not_install_or_report_cpu_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_setup, "probe_current_runtime", lambda: {
        "ok": False, "python_executable": sys.executable,
        "reason": "No CUDA runtime", "packages_installed": False})
    assert gpu_setup.main(["--run-dir", str(tmp_path), "--retry"]) == 2
    report = json.loads((tmp_path / "gpu_setup.json").read_text())
    assert report["state"] == "unavailable"
    assert not report["packages_installed"]


def test_explicit_cpu_execution_never_probes_cuda(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("CPU request must not probe CUDA")
    monkeypatch.setattr(gpu_setup, "probe_current_runtime", forbidden)
    assert gpu_setup.main(["--run-dir", str(tmp_path), "--cpu"]) == 0
    assert json.loads((tmp_path / "gpu_setup.json").read_text())["state"] == "cpu"


def test_stop_request_skips_runtime_probe(tmp_path, monkeypatch):
    (tmp_path / "stop.flag").touch()
    monkeypatch.setattr(gpu_setup, "probe_current_runtime", lambda: (_ for _ in ()).throw(AssertionError()))
    assert gpu_setup.main(["--run-dir", str(tmp_path)]) == 10
    assert json.loads((tmp_path / "gpu_setup.json").read_text())["state"] == "cancelled"


def test_crashed_probe_cannot_reuse_a_ready_report(tmp_path):
    path = tmp_path / "gpu_setup.json"
    path.write_text(json.dumps({"state": "ready", "ok": True,
                                "python_executable": "/old/runtime/python"}))
    for status in (-9, 1, 2, None):
        report = read_runtime_probe(path, status)
        assert report["state"] == "unavailable"
        assert not report["ok"]
        assert "python_executable" not in report


def test_incomplete_readiness_and_malformed_report_are_rejected(tmp_path):
    path = tmp_path / "gpu_setup.json"
    assert read_runtime_probe(path, 0)["state"] == "unavailable"
    path.write_text(json.dumps({"state": "ready", "python_executable": sys.executable}))
    assert read_runtime_probe(path, 0)["state"] == "unavailable"
    path.write_text("[]")
    assert read_runtime_probe(path, 0)["state"] == "unavailable"


def test_successful_probe_accepts_external_managed_python(tmp_path):
    path = tmp_path / "gpu_setup.json"
    path.write_text(json.dumps({"state": "ready", "ok": True,
                                "python_executable": "/usr/local/bin/python3"}))
    assert read_runtime_probe(path, 0)["python_executable"] == "/usr/local/bin/python3"
