"""Tiny HTTP server for Koyeb LLVC device-test deployment.

Endpoints:
  GET  /health          -> 200 OK (used by Koyeb health checks)
  GET  /check           -> runs tt-smi, returns device presence
  POST /run             -> starts pytest device tests in background
  GET  /status          -> shows whether tests are running / done
  GET  /results         -> returns stdout + CSV contents if available
"""
import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

# Shared mutable state (single-process, single-worker is fine for Koyeb)
_state = {
    "running": False,
    "done": False,
    "stdout": "",
    "stderr": "",
    "returncode": None,
}

REPO_ROOT = Path("/app/tt-metal")
RESULT_DIR = Path("/tmp/llvc_results")
RESULT_DIR.mkdir(exist_ok=True)


def _run_tests():
    _state["running"] = True
    _state["done"] = False
    _state["stdout"] = ""
    _state["stderr"] = ""
    _state["returncode"] = None

    # 1. Device correctness test
    proc1 = subprocess.run(
        [sys.executable, "-m", "pytest",
         "models/demos/audio/llvc/tests/test_llvc.py",
         "-k", "ttnn", "-v", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "TT_METAL_HOME": str(REPO_ROOT)},
    )
    out = f"=== test_llvc.py (device) ===\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}\nRC: {proc1.returncode}\n\n"

    # 2. Device performance test
    proc2 = subprocess.run(
        [sys.executable, "-m", "pytest",
         "models/demos/audio/llvc/tests/test_perf_llvc.py::test_llvc_performance_device",
         "-v", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "TT_METAL_HOME": str(REPO_ROOT)},
    )
    out += f"=== test_perf_llvc.py (device) ===\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}\nRC: {proc2.returncode}\n"

    # 3. Collect any CSV sidecars
    csv_files = list(REPO_ROOT.rglob("perf_llvc_streaming_device_n300_*.csv"))
    csv_files += list(REPO_ROOT.rglob("perf_llvc_*.csv"))
    out += f"\n=== CSV files found ({len(csv_files)}) ===\n"
    for csv in csv_files:
        out += f"\n--- {csv.name} ---\n"
        try:
            out += csv.read_text()
        except Exception as e:
            out += f"Error reading {csv}: {e}\n"

    _state["stdout"] = out
    _state["stderr"] = ""
    _state["returncode"] = proc1.returncode or proc2.returncode
    _state["running"] = False
    _state["done"] = True


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/check")
def check():
    """Quick device sanity check."""
    try:
        smi = subprocess.run(
            ["tt-smi"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        smi = None
    except Exception as e:
        return jsonify({"tt-smi_found": False, "error": str(e)}), 200

    try:
        # Try a tiny ttnn import + device open
        check_py = REPO_ROOT / "models" / "demos" / "audio" / "llvc" / "_device_check.py"
        check_py.write_text(
            "import ttnn; d=ttnn.open_device(0); ttnn.close_device(d); print('OK')"
        )
        ttnn_check = subprocess.run(
            [sys.executable, str(check_py)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "TT_METAL_HOME": str(REPO_ROOT)},
        )
        ttnn_ok = ttnn_check.returncode == 0
        ttnn_out = ttnn_check.stdout.strip() + ttnn_check.stderr.strip()
    except Exception as e:
        ttnn_ok = False
        ttnn_out = str(e)

    return jsonify({
        "tt-smi_found": smi is not None,
        "tt-smi_returncode": smi.returncode if smi else None,
        "tt-smi_stdout": (smi.stdout[:2000] if smi else ""),
        "ttnn_device_open": ttnn_ok,
        "ttnn_output": ttnn_out[:2000],
    })


@app.route("/run", methods=["POST"])
def run():
    if _state["running"]:
        return jsonify({"status": "already_running", "done": _state["done"]})
    t = threading.Thread(target=_run_tests, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    return jsonify({
        "running": _state["running"],
        "done": _state["done"],
        "returncode": _state["returncode"],
    })


@app.route("/results")
def results():
    return jsonify({
        "done": _state["done"],
        "returncode": _state["returncode"],
        "stdout": _state["stdout"],
    })


if __name__ == "__main__":
    # Koyeb exposes whatever port we listen on; 8080 is conventional.
    app.run(host="0.0.0.0", port=8080, threaded=True)
