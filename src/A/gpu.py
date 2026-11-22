"""GPU lock helper: wraps `python bus/bus.py gpu acquire|release` (see PROTOCOL.md)."""
import contextlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _bus(*args):
    return subprocess.run([sys.executable, str(ROOT / "bus" / "bus.py"), "gpu", *args],
                          cwd=ROOT, capture_output=True, text=True)


@contextlib.contextmanager
def gpu_lock():
    while True:
        r = _bus("acquire")
        if r.returncode == 0:
            print(r.stdout.strip(), flush=True)
            break
        print("GPU busy, retrying:", r.stdout.strip(), flush=True)
        time.sleep(5)
    try:
        yield
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(_bus("release").stdout.strip(), flush=True)
