#!/usr/bin/env bash
# Install the detector into the existing VLA environment; no server or robot IO.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
vla_python="$repo_dir/software/ai/smolvla_py/.venv/bin/python"
assets_dir="$repo_dir/local/mission_control/assets"
mkdir -p "$assets_dir" "$repo_dir/local/mission_control/runtime/uv-cache"
if command -v uv >/dev/null 2>&1; then
  # MediaPipe requires the contrib wheel. Avoid two distributions owning cv2.
  UV_CACHE_DIR="$repo_dir/local/mission_control/runtime/uv-cache" uv pip uninstall --python "$vla_python" opencv-contrib-python-headless
  UV_CACHE_DIR="$repo_dir/local/mission_control/runtime/uv-cache" uv pip install --python "$vla_python" --requirements "$repo_dir/local/mission_control/requirements.txt"
else
  "$vla_python" -m pip uninstall --yes opencv-contrib-python-headless
  "$vla_python" -m pip install --no-cache-dir --requirement "$repo_dir/local/mission_control/requirements.txt"
fi
if [[ ! -s "$assets_dir/hand_landmarker.task" ]]; then
  curl --fail --location --connect-timeout 10 --max-time 120 \
    --output "$assets_dir/hand_landmarker.task.tmp" \
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
  mv "$assets_dir/hand_landmarker.task.tmp" "$assets_dir/hand_landmarker.task"
fi
