"""Mission Control defaults. Checkpoint and recording paths remain read-only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FOLLOWER = "5B8E113018"
LEADER = "5B91030264"
STATION = "localhost:8888"
MOTOR_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_RUN_SECONDS = 120.0
DEFAULT_MAX_DELTA_TICKS = 300
MAX_RUN_SECONDS = 300.0
MAX_DELTA_TICKS_LIMIT = 1000
PREDICTION_BOUND_TOLERANCE = 0.05
CHECKPOINT = ROOT / "software/ai/smolvla_py/checkpoints/block-local-20261122-004545/final"
VLA_PYTHON = ROOT / "software/ai/smolvla_py/.venv/bin/python"
CHECKPOINT_LABEL = str(CHECKPOINT.relative_to(ROOT))
EPISODE_STATE = ROOT / "local/episode_dashboard_state.json"
EPISODE_URL = "http://127.0.0.1:8891"
FRESH_SECONDS = 2.0
# Legacy camera list order for unassigned previews and calibration.
CAMERAS = {"workspace": 1, "wrist": 0}
# All five episodes used for this checkpoint have this model input order.
POLICY_CAMERAS = {"workspace": 0, "wrist": 1}
# Workspace identity is independent of whether the homography is usable.
WORKSPACE_CAMERA_ID = "11391:19040:3:53"
CAMERA_SOURCE = ROOT / "local/calibration/camera_source.json"
BENCH_MAT = HERE / "bench_mat.json"
CAMERA_CALIBRATION = HERE / "runtime/workspace_homography.json"
MANUAL_CALIBRATION = ROOT / "local/calibration/tabletop_homography.json"
TAUGHT_POINTS = MANUAL_CALIBRATION.with_name("robot_taught_points.json")
TAUGHT_POINT_COUNT = 6
TAUGHT_START_TOLERANCE_TICKS = 20
TAUGHT_VLA_MAX_OFFSET_TICKS = 500
ROBOT_ALIGNMENT = HERE / "robot_alignment.json"
REQUESTED_SAFE_HEIGHT_MM = 200.0
APPROACH_MAX_STEP_TICKS = 50
HANDOFF_MAX_STEP_TICKS = 100
HANDOFF_SECONDS = 20.0
HAND_MODEL = HERE / "assets/hand_landmarker.task"
HAND_MAX_AGE_SECONDS = 1.0
# Independent of homography: normalized camera-image x0, y0, x1, y1.
# The rest of the camera image is the yellow warning zone.
HAND_EXCLUSION_RECT = (0.1, 0.1, 0.9, 0.9)
