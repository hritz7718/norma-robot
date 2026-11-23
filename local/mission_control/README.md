# ElRobot Mission Control

Execute captures a fresh workspace JPEG, uses Codex image interpretation (or
labelled OpenCV blue fallback), then runs the existing persistent SmolVLA loop.
Valid camera/robot setup adds a six-pose or URDF approach before VLA. Missing,
incomplete or invalid setup selects direct VLA at startup, with no tabletop
approach or Cartesian height control. Object/hand detection works in image
pixels without homography. Both routes require live cameras, fault-free Follower
feedback, calibrated servo limits and disabled mirroring. Hand alerts are warning-only.
The existing UI, two camera streams, telemetry, logs, and STOP remain in use.
Mission Control assembles model input from the selected workspace and wrist
cameras and eight Follower joints in one synchronized Station snapshot. It can
read joint positions with torque off, before Execute prepares the Follower.

## Launch

From the repository root, with Station already running at localhost:8888:

```bash
bash local/mission_control/setup_hands.sh
software/ai/smolvla_py/.venv/bin/python -m local.mission_control --port 8892 --safe-height-mm 200
```

Open http://127.0.0.1:8892. Default duration is 120 seconds, requested max step
300 ticks, and requested TCP height 200mm. The UI can change these before Execute.
Approach height applies to the positioning route only. Direct VLA uses the
requested duration and at most 100 ticks per joint per action; the positioned
VLA handoff is limited to 20 seconds. The UI displays the selected route and
logs its reason. Calibration or approach-planning failures during preflight
select direct VLA using the same Execute click. After Follower preparation
begins, execution faults stop the run and require another Execute click.
Calibration, interpretation, model loading and planning issue no motor writes.
The movement timer starts only after those prerequisites finish. Workspace
camera identity is selected in the UI and saved in local/calibration/camera_source.json.
WORKSPACE_CAMERA_ID in config.py supplies the initial identity.

## Saved camera calibration and six-pose teaching

Launch the separate calibration page in another terminal:

```bash
software/ai/smolvla_py/.venv/bin/python local/table_calibration.py --port 8894
```

Open http://127.0.0.1:8894/calibration. Capture the tabletop camera, select four
points and enter their measured table X/Y coordinates in millimetres. Save the
homography. These reference points do not restrict the task workspace.

Use existing Leader → Follower teleoperation to teach six reachable table
positions at the same measured gripper height and consistent orientation and
opening. Save Point captures fresh, stationary feedback for all eight Follower
motors. The separate page never commands motion or changes torque/mirroring.
The grasp point is midway between the fingertips; X/Y refer to its tabletop
projection and Z is its measured height above the tabletop.

After all six poses are saved, preview an intermediate X/Y, physically
validate the interpolated positions and clearance, confirm the checkbox and
click Finish setup. Execute uses piecewise linear joint interpolation
inside the taught region when the current pose matches a taught pose. Other
poses use direct VLA. Disable mirroring using the existing supported-arm
handover and keep Follower feedback available before Execute. Approach commands are
bounded to 50 ticks or the smaller operator limit. Joint feedback confirms
arrival; Cartesian height remains an operator-validated estimate.

The files live in local/calibration/tabletop_homography.json and
local/calibration/robot_taught_points.json. Camera identity, calibration
signature, motor order and servo limits must match. Editing points disables
the taught controller until revalidated. A different physical setup needs its
own calibration; the checked-in files describe this Follower/camera only.

## Hand monitoring and voice warnings

MediaPipe runs on the main workspace camera in a separate backend thread.
Landmarks, bounding boxes and independent camera-space warning/exclusion zones
appear in the workspace feed. The homography reference box is not a hand zone.
HAND_EXCLUSION_RECT in config.py defines the red area in normalized image
coordinates; the remaining image is yellow. Hand detections and detector failures
produce visual/spoken warnings without stopping VLA or blocking Execute. Stale
robot observations or camera frames still stop commands independently. Manual
STOP requires a deliberate Execute click to restart and does not disable torque
or stop existing teleoperation.

Browser speech announces warnings with a cooldown. Click Test voice to enable
audio and use the voice selector to preview/save an available browser voice.
This is prototype monitoring, not a certified robot safety system.

## Episode recording

```bash
software/ai/smolvla_py/.venv/bin/python local/episode_dashboard.py
```

Open http://127.0.0.1:8891. Record additional teleoperated demonstrations and
finish/export each take. There is no five-episode recording cap; five successful
exports are the training minimum. Train currently uses all ready exports.
Datasets, trained checkpoints, recordings and local runtime state are not
committed. The configured SmolVLA checkpoint must be supplied locally.

## Bench Mat measurements

bench_mat.json contains the six operator-supplied marker CENTRES in mm:
202=(-75,262), 203=(75,262), 204=(-75,146), 205=(75,146),
200=(-75,30), 201=(75,30). No marker side length or paper size is assumed.

Set the actual dictionary, or leave AUTO to match the six known IDs and
geometry against OpenCV dictionaries. Equivalent decoders are reported in
the saved record; inconsistent/ambiguous matches fail.
Fill both physically measured horizontal and vertical centre distances in
scale_reference and set physically_measured=true. Printed labels are nominal
references and do not prove the physical print scale. Defaults deliberately
leave those measurements unset.

The four outer centres fit the homography; 204/205 independently check it.
All six then refine the fit. Limits are 3px for the independent check and
2px maximum fitted reprojection error. Marker centres use diagonal
intersections. Saves runtime/workspace_homography.json and revalidates cached
geometry against the live logical camera source, resolution, and all markers.
Marker drift, stale images, or changed configuration prevent movement.
Workspace overlays are generated from the actual current image, not sent to
the model; SmolVLA and Codex receive the original unannotated camera data.

## Physical robot alignment

robot_alignment.json deliberately starts verified=false. Its required measured
data are: follower identity and eight motor IDs, eight servo zero ticks and
signed radians-per-tick values, the exact current joint calibration ranges,
tool-link TCP offset (3 values, mm), tool envelope radius, table-to-base
registration, measured obstacle/clearance heights, and XYZ workspace bounds
(three [minimum,maximum] pairs in table mm).

Registration can be a measured rigid table_to_base_mm 4x4 matrix, or a visible
base_reference marker with a measured marker_to_base_mm rigid transform.
Direct registration also requires robot_and_mat_fixed=true and its measured
registration_table_to_image homography; changed camera/mat geometry blocks it.
Marker 312 is only an ID, never an assumed origin. The marker must be coplanar
with the mat and fixed relative to the real base. Its frame is at its centre,
with X along the decoded top-left→top-right edge, Z up from the tabletop, and
right-handed Y. Offsets must be measured in that frame.

Provide at least three verification_samples, each with eight raw ticks and a
measured tcp_table_mm position. Samples must span XY and at least 10mm in Z;
the supplied ElRobot URDF mapping must reproduce them within 3mm.
Set registration_verified, workspace_clearance_verified and verified only
after the actual measurements. This URDF route is the alternative when no
taught-pose file is present. Existing nine-pose sets remain supported. Invalid
teaching, URDF calibration, or an unavailable approach selects direct VLA during
preflight. The mode label and logs show which route was selected. After Follower
preparation begins, geometry or file changes stop the run.

The planner reuses hardware/elrobot/simulation/elrobot_follower.urdf. It checks
FK, solves a fixed-orientation TCP approach, samples bounded joint segments,
and monitors actual feedback. The start must already clear the measured
obstacle/tool envelope. It first raises within that clear region, then moves
horizontally at the requested height. Unknown alignment/Z, unreachable poses,
stalls or changed servo calibration stop before VLA handoff.

## Codex, handoff and STOP

The former text-only call is replaced in Execute by the installed CLI's
--image attachment with a captured JPEG and a vision response schema.
The existing audited CLI/version restrictions and login are retained.
The response supplies an approximate pixel box; its centre supplies the primary
object coordinates, reported with source=codex. A valid Codex result does not
require an OpenCV colour-mask confirmation. The overlay labels this as the last
Codex snapshot result; it does not claim live object tracking. In direct mode,
SmolVLA uses fresh live camera observations rather than a frozen pixel target.
The calibrated positioning route refreshes Codex coordinates after model loading
before computing its tabletop approach. Homography supplies millimetres only
when valid; Codex itself supplies image pixels, never invented robot coordinates.
If image interpretation is unavailable, only an explicitly identified single
blue-block pickup can use the labelled OpenCV fallback. No successful Codex
image interpretation is claimed in that case. Image frame ID/hash and source
appear in Troubleshooting. The object must remain stationary during latency.

The object XY is the tabletop projection of its visible centroid and remains
approximate for a raised block. Table registration converts it to robot XY;
verified FK/IK converts the safe TCP target into servo goals. A homography
alone is never used to invent joint targets.

On the URDF route, feedback must confirm the safe approach within 3mm. On the
taught route, all joints must reach the interpolated goal within 8 ticks;
this does not independently measure Cartesian accuracy. The existing retained
SmolVLA model uses CURRENT live observations. Approach steps are at most 50
ticks; this new, untrained handoff is capped at 100 ticks per joint and 20
seconds, also respecting any smaller requested limits. Finite prediction,
calibration and motor checks stay active. URDF positioning also checks measured
workspace/table clearance; taught handoff limits arm displacement to 500 ticks
from the reached approach and relies on prior physical clearance validation.
FINISHED means the supervised attempt ended, not that a grasp was proven.
Independent grasp verification is explicitly UNVERIFIED; inspect both cameras.

STOP cancels calibration, Codex and subsequent policy/positioning writes and
prevents a late interpretation from restarting motion. It leaves torque
unchanged and is not a physical emergency stop; an already delivered command
may finish. The current Station mirror-stop drops Follower torque, so active
mirroring causes MIRRORING HANDOVER REQUIRED rather than an automatic torque
drop. Safely disable mirroring with the arm supported before retrying.
Leader motors are never command targets.

No tests or services were started during implementation; no movement was initiated.
