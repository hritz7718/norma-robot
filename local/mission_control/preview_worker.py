"""One offline prediction. No Station connection, command building, or motor writes.

Only this worker loads torch; the HTTP server stays responsive during model load.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "software/station/shared"), str(ROOT / "software/ai/smolvla_py")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from local.mission_control.config import CHECKPOINT


def deny(*args, **kwargs):
    raise PermissionError("Offline preview worker cannot connect to Station or send commands.")


def main():
    import torch
    import station_py
    # Defense in depth before importing the existing inference module.
    station_py.send_commands = deny
    station_py.new_station_client = deny
    from scripts import run_policy as rp
    from target.gen_python.protobuf.drivers.inferences import normvla
    rp.build_sync_write_command = deny
    rp.predict_commands = deny
    rp.run_one_tick = deny
    request = json.loads(sys.stdin.read(8192))
    checkpoint = CHECKPOINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.monotonic()
    stats = {k: v.to(device) for k, v in rp.load_stats(checkpoint / "stats.safetensors").items()}
    policy = rp.SmolVLAPolicy.from_pretrained(checkpoint,
        config_overrides={"load_vlm_weights": False}, strict=False).to(device)
    policy.eval()
    frame = normvla.FrameReader(memoryview(Path(request["frame_path"]).read_bytes()))
    joints = frame.get_joints() or []
    if len(joints) != 8 or len(frame.get_images() or []) != 2:
        raise RuntimeError("Expected exactly eight joints and two images.")
    batch, _ = rp.frame_to_batch(frame, stats, rp.IMAGE_KEYS, device)
    tokens, mask = policy.tokenize_task(request["task"], device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    with torch.inference_mode():
        # Match Episode Lab's read-only preview; preserve unclipped predictions.
        pred = policy.predict_action_chunk(batch)[0]
        goals = rp.unnormalize_action(pred, stats)[0].detach().cpu().tolist()
    current = [float(j.get_position_norm()) for j in joints]
    if len(goals) != 8 or not all(math.isfinite(g) for g in goals + current):
        raise RuntimeError("Invalid model prediction.")
    result = {"predicted": goals, "current": current,
              "delta": [g-c for g, c in zip(goals, current)],
              "out_of_range": [not 0 <= g <= 1 for g in goals],
              "inference_seconds": round(time.monotonic() - start, 2),
              "device": str(device), "commands_sent": 0, "source": "SmolVLA checkpoint"}
    print("MISSION_PREVIEW_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
