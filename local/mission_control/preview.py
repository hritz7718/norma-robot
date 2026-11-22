import json
import math
import os
import tempfile
import time
from pathlib import Path
from .config import CHECKPOINT, CHECKPOINT_LABEL, HERE, ROOT, VLA_PYTHON
from .processes import run_process


def policy_instruction(task):
    target = task["target"]
    color = "" if target["color"] == "unspecified" else target["color"] + " "
    return ("pick up" if task["task_type"] == "pick_up" else "inspect") + " the " + color + target["object"]


def predict(task, station, mock=False, captured=None):
    # Only a pickup is in this local checkpoint's demonstrated action domain.
    if task["task_type"] != "pick_up":
        raise ValueError("VLA preview is available for pickup tasks only; inspection has no motor prediction.")
    raw, snapshot = captured if captured is not None else station.capture()
    start = time.time()
    if mock:
        current = [j["pos_norm"] for j in snapshot["joints"]]
        goals = [round(c + .01 * (-1 if i % 2 else 1), 4) for i, c in enumerate(current)]
        result = {"current": current, "predicted": goals, "delta": [g-c for g,c in zip(goals,current)],
                  "out_of_range": [False]*8, "inference_seconds": 0, "device": "simulation",
                  "commands_sent": 0, "source": "SIMULATED prediction; model not loaded"}
    else:
        if not CHECKPOINT.is_dir() or not VLA_PYTHON.exists():
            raise RuntimeError("Configured checkpoint or SmolVLA Python environment is missing.")
        with tempfile.TemporaryDirectory(prefix="elrobot-preview-") as folder:
            frame_path = Path(folder) / "observation.bin"
            frame_path.write_bytes(raw)
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            out = run_process([str(VLA_PYTHON), str(HERE / "preview_worker.py")], cwd=ROOT,
                              input_text=json.dumps({"frame_path": str(frame_path), "task": policy_instruction(task)}),
                              timeout=180, env=env)
            lines = [line.removeprefix("MISSION_PREVIEW_JSON=") for line in out.splitlines()
                     if line.startswith("MISSION_PREVIEW_JSON=")]
            if len(lines) != 1:
                raise RuntimeError("Preview worker did not return a prediction.")
            result = json.loads(lines[0])
    for name in ("current", "predicted", "delta"):
        if len(result[name]) != 8 or not all(type(x) in (int, float) and math.isfinite(x) for x in result[name]):
            raise RuntimeError("Malformed preview output.")
    if (result.get("commands_sent") != 0 or len(result.get("out_of_range", [])) != 8
            or any(type(flag) is not bool for flag in result["out_of_range"])):
        raise RuntimeError("Invalid preview worker response.")
    for current, goal, delta, out_of_range in zip(
            result["current"], result["predicted"], result["delta"], result["out_of_range"]):
        if not math.isclose(goal - current, delta, abs_tol=1e-6) or out_of_range != (not 0 <= goal <= 1):
            raise RuntimeError("Inconsistent preview worker response.")
    result.update({"task_id": task["id"], "frame_id": snapshot["frame_id"],
                   "captured_at": start, "capture_age_ms": snapshot["observation_age_ms"],
                   "checkpoint": CHECKPOINT_LABEL, "historical": True,
                   "instruction": policy_instruction(task), "observation": snapshot,
                   "camera_urls": {name: f"/api/preview/camera/{name}.jpg?task_id={task['id']}&frame_id={snapshot['frame_id']}"
                                   for name in ("workspace", "wrist")}})
    return result
