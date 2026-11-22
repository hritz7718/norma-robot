"""Fixed executable invocations, bounded lifetime, no shell interpolation."""
import os
import signal
import subprocess
import time
from concurrent.futures import CancelledError


def run_process(argv, *, cwd, input_text, timeout, env=None, cancel=None):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Worker cancelled.")
    with subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True, shell=False) as proc:
        deadline = time.monotonic() + timeout
        pending_input = input_text
        while True:
            cancelled = cancel is not None and cancel.is_set()
            remaining = deadline - time.monotonic()
            if cancelled or remaining <= 0:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                if cancelled:
                    raise CancelledError("Worker cancelled.")
                raise RuntimeError("Worker timed out; its process group was terminated.")
            try:
                out, err = proc.communicate(pending_input, timeout=min(.1, remaining) if cancel is not None else remaining)
                break
            except subprocess.TimeoutExpired:
                # communicate retains pending input and output across timeouts.
                pending_input = None
        if len(out) > 1_000_000:
            raise RuntimeError("Worker response exceeded the output limit.")
        if proc.returncode:
            # Do not return stderr: CLI/network errors may contain authentication details.
            raise RuntimeError(f"Worker exited with code {proc.returncode}. Check CLI login or local model dependencies.")
        return out
