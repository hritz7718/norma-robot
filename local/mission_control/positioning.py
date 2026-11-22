"""Taught joint interpolation or measured ElRobot URDF positioning."""
import math
import hashlib
import time
import xml.etree.ElementTree as ET
import json

from .config import (FOLLOWER, MOTOR_IDS, ROBOT_ALIGNMENT, ROOT, MANUAL_CALIBRATION,
                     TAUGHT_POINTS, TAUGHT_START_TOLERANCE_TICKS, TAUGHT_VLA_MAX_OFFSET_TICKS)
from .vision import cancelled, VisionError


class AlignmentError(RuntimeError):
    pass


def homography_signature(calibration):
    geometry = {key: calibration[key] for key in ("method", "units", "camera_unique_id", "image_size",
                                                "pixel_points", "table_points_mm", "image_to_table")}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()


class TaughtMapping:
    """Piecewise linear interpolation of nine operator-validated joint poses.

    Heights are measured during teaching, not observed or guaranteed by FK.
    Starts must match a taught pose; no automatic raising from arbitrary poses.
    """
    taught = True

    def __init__(self, vision=None, snapshot=None, safe_height=None, *, path=TAUGHT_POINTS,
                 calibration=None, require_validation=True):
        import cv2
        import numpy as np
        self.np, self.path = np, path
        self.handoff_anchor = None
        try:
            raw = path.read_bytes()
            self.signature = hashlib.sha256(raw).hexdigest()
            self.config = record = json.loads(raw)
            calibration = json.loads(MANUAL_CALIBRATION.read_text()) if calibration is None else calibration
            self.calibration_signature = homography_signature(calibration)
            if record.get("method") != "taught_xy_grid" or record.get("units") != "mm":
                raise ValueError("unsupported teaching file format")
            if record.get("follower") != FOLLOWER or record.get("motor_ids") != list(MOTOR_IDS):
                raise ValueError("teaching belongs to a different robot or motor order")
            if record.get("calibration_signature") != self.calibration_signature or record.get("camera_unique_id") != calibration["camera_unique_id"]:
                raise ValueError("camera calibration changed; reteach the nine points")
            if vision is not None and (vision.record.get("method") != "manual_four_point" or
                    vision.record["image_size"] != calibration["image_size"] or
                    not np.allclose(vision.record["image_to_table"], calibration["image_to_table"])):
                raise ValueError("live camera calibration does not match teaching")
            points = sorted(record["points"], key=lambda point: point["slot"])
            if len(points) != 9 or [point["slot"] for point in points] != list(range(1, 10)):
                raise ValueError("save all nine teaching points first")
            xyz = np.asarray([[point["table_mm"][axis] for axis in ("x", "y", "z")] for point in points], dtype=float)
            self.ticks = np.asarray([point["joint_ticks"] for point in points], dtype=float)
            if xyz.shape != (9, 3) or self.ticks.shape != (9, 8) or not np.isfinite(xyz).all() or not np.isfinite(self.ticks).all():
                raise ValueError("teaching requires nine finite XYZ coordinates and eight ticks per point")
            self.safe_height = float(record["safe_height_mm"])
            if not math.isfinite(self.safe_height) or not 1 <= self.safe_height <= 500 or np.max(abs(xyz[:, 2]-self.safe_height)) > 1:
                raise ValueError("all nine points must have the same measured safe height within 1mm")
            if safe_height is not None and abs(float(safe_height)-self.safe_height) > 1:
                raise ValueError(f"set Mission Control Approach height to the taught {self.safe_height:g}mm")
            self.xy = xyz[:, :2]
            if np.linalg.matrix_rank(self.xy-self.xy[0]) < 2:
                raise ValueError("teaching points must span X and Y")
            for index, xy in enumerate(self.xy):
                if any(np.linalg.norm(xy-other) < 1 for other in self.xy[index+1:]):
                    raise ValueError("teaching X/Y points overlap")
            limits = np.asarray(points[0]["joint_ranges_ticks"], dtype=float)
            if limits.shape != (8, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] < 0) or np.any(limits[:, 1] > 4095) or np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError("teaching needs eight calibrated, non-wrapping servo ranges")
            self.ranges = [tuple(pair) for pair in limits.astype(int).tolist()]
            if np.any(limits != np.rint(limits)) or np.any(self.ticks != np.rint(self.ticks)):
                raise ValueError("saved motor ticks and limits must be integers")
            if any(point["joint_ranges_ticks"] != limits.tolist() for point in points):
                raise ValueError("servo limits changed between teaching points")
            if snapshot is not None and [(joint["range_min"], joint["range_max"]) for joint in snapshot["joints"]] != self.ranges:
                raise ValueError("live servo calibration differs from teaching")
            for ticks in self.ticks:
                self.check_ticks(ticks)
            # OpenCV is already installed. Triangles support an irregular nine-point grid.
            origin = np.floor(self.xy.min(axis=0))-1
            size = np.ceil(self.xy.max(axis=0)-origin).astype(int)+2
            subdiv = cv2.Subdiv2D((0, 0, int(size[0]), int(size[1])))
            for xy in self.xy:
                subdiv.insert(tuple(float(value) for value in xy-origin))
            self.triangles = []
            for triangle in subdiv.getTriangleList():
                vertices = np.asarray(triangle, dtype=float).reshape(3, 2)+origin
                indices = [int(np.argmin(np.linalg.norm(self.xy-vertex, axis=1))) for vertex in vertices]
                if len(set(indices)) != 3 or any(np.linalg.norm(self.xy[index]-vertex) > .01 for index, vertex in zip(indices, vertices)):
                    continue
                matrix = np.vstack([self.xy[indices].T, np.ones(3)])
                if abs(np.linalg.det(matrix)) > 1e-8:
                    self.triangles.append((indices, np.linalg.inv(matrix)))
            if not self.triangles:
                raise ValueError("teaching points cannot form an interpolation region")
            for xy, ticks in zip(self.xy, self.ticks):
                if np.max(abs(self.interpolate(xy)-ticks)) > 1:
                    raise ValueError("interpolation does not reproduce the taught poses")
            if require_validation and (record.get("physically_validated") is not True or record.get("motion_enabled") is not True):
                raise ValueError("preview and physically validate interpolation, then enable the taught controller on the calibration page")
        except (OSError, KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError, cv2.error) as exc:
            raise AlignmentError(f"TAUGHT ALIGNMENT REQUIRED: {exc}.") from exc

    def assert_current(self):
        try:
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.signature or homography_signature(json.loads(MANUAL_CALIBRATION.read_text())) != self.calibration_signature:
                raise ValueError("teaching or camera calibration changed")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise AlignmentError(f"Taught mapping changed or is unavailable; stop and reload: {exc}.") from exc

    def check_ticks(self, ticks):
        values = self.np.asarray(ticks, dtype=float)
        if values.shape != (8,) or not self.np.isfinite(values).all() or any(not low <= value <= high for value, (low, high) in zip(values, self.ranges)):
            raise AlignmentError("Joint feedback/target exceeds the taught servo limits.")
        return values

    def interpolate(self, xy):
        target = self.np.asarray(xy, dtype=float)
        if target.shape != (2,) or not self.np.isfinite(target).all():
            raise AlignmentError("Target X/Y must be finite measured millimetres.")
        for indices, inverse in self.triangles:
            weights = inverse @ self.np.append(target, 1)
            if self.np.min(weights) >= -1e-6 and self.np.max(weights) <= 1+1e-6:
                weights = self.np.clip(weights, 0, 1); weights /= weights.sum()
                ticks = self.np.rint(weights @ self.ticks[indices]).astype(int)
                self.check_ticks(ticks)
                return ticks
        raise AlignmentError("Target is outside the nine-point taught region. Extrapolation is disabled.")

    def check_segment(self, start, end, *, approach=True):
        first, last = self.check_ticks(start), self.check_ticks(end)
        # Convex non-wrapping servo ranges also contain the interpolated segment.
        # No live Cartesian Z or collision clearance is inferred from joint ticks.
        if not approach:
            if self.handoff_anchor is None:
                raise AlignmentError("The taught approach must be reached before the VLA handoff.")
            if any(self.np.max(abs(values[:7]-self.handoff_anchor[:7])) > TAUGHT_VLA_MAX_OFFSET_TICKS for values in (first, last)):
                raise AlignmentError("VLA exceeded the supervised joint envelope around the taught approach.")

    def plan(self, target_xy, current_raw, cancel):
        self.assert_current()
        current = self.check_ticks(current_raw)
        distances = self.np.max(abs(self.ticks-current), axis=1)
        start = int(self.np.argmin(distances))
        if distances[start] > TAUGHT_START_TOLERANCE_TICKS:
            raise AlignmentError(f"START POSE REQUIRED: use teleoperation to return to a taught pose (nearest point {start+1}), then disable mirroring before Execute.")
        target = self.np.asarray(target_xy, dtype=float)
        final = self.interpolate(target)
        count = max(1, int(math.ceil(self.np.linalg.norm(target-self.xy[start])/5)))
        if count > 200:
            raise AlignmentError("Approach exceeds the prototype path-length bound.")
        waypoints = []
        for fraction in self.np.linspace(0, 1, count+1):
            cancelled(cancel)
            waypoints.append(self.interpolate(self.xy[start]*(1-fraction)+target*fraction).tolist())
        return {"waypoints": waypoints, "target_ticks": final.tolist(),
                "target_table_mm": [float(target[0]), float(target[1]), self.safe_height],
                "target_robot_mm": None, "safe_height_mm": self.safe_height,
                "start_point": start+1, "method": "nine-point joint interpolation",
                "verified": True, "position_estimated": True,
                "height_control": "Operator-validated taught height; no live Cartesian height measurement"}


def load_mapping(vision, snapshot, safe_height):
    if TAUGHT_POINTS.exists():
        return TaughtMapping(vision, snapshot, safe_height)
    return RobotMapping(vision, snapshot, safe_height)


class RobotMapping:
    def __init__(self, vision, snapshot, safe_height, *, config=None):
        import numpy as np
        self.np, self.vision = np, vision
        try:
            raw = ROBOT_ALIGNMENT.read_bytes() if config is None else None
            self.signature = hashlib.sha256(raw).hexdigest() if raw is not None else None
            self.config = config = json.loads(raw) if raw is not None else config
            if config.get("verified") is not True:
                raise ValueError("robot_alignment.json has no verified physical robot alignment")
            if config["follower"] != FOLLOWER or config["motor_ids"] != list(MOTOR_IDS):
                raise ValueError("alignment belongs to a different robot or motor order")
            if config["mat_id"] != vision.record["mat_id"]:
                raise ValueError("alignment belongs to a different Bench Mat")
            if config.get("registration_verified") is not True or config.get("workspace_clearance_verified") is not True:
                raise ValueError("table registration and workspace clearance must be physically verified")
            self.zero = self.vector(config["joint_zero_ticks"], 8)
            self.scale = self.vector(config["joint_radians_per_tick"], 8)
            if np.any(abs(self.scale) < 1e-8):
                raise ValueError("each motor needs a measured signed radians-per-tick mapping")
            self.ranges = [(j["range_min"], j["range_max"]) for j in snapshot["joints"]]
            if config["joint_ranges_ticks"] != [list(pair) for pair in self.ranges]:
                raise ValueError("servo calibration ranges differ from the measured alignment")
            transform = config.get("table_to_base_mm")
            if transform is not None:
                if config.get("robot_and_mat_fixed") is not True:
                    raise ValueError("direct registration requires the robot and mat to remain fixed")
                registration = np.asarray(config["registration_table_to_image"], dtype=float)
                if registration.shape != (3, 3) or not np.isfinite(registration).all():
                    raise ValueError("direct registration must include the homography captured during its physical measurement")
                points = list(vision.layout["centres"].values())
                expected = vision.project(points, registration)
                current = vision.project(points, vision.record["table_to_image"])
                if np.linalg.norm(expected - current, axis=1).max() > 3:
                    raise ValueError("camera/mat geometry changed since physical robot registration")
                self.table_to_base = self.rigid(transform)
            else:
                reference = config["base_reference"]
                if reference.get("fixed_relative_to_base") is not True or reference.get("coplanar_with_mat") is not True:
                    raise ValueError("base marker requires a measured rigid offset and verified tabletop coplanarity")
                marker = vision.base_marker(int(reference["marker_id"]))
                x = marker[1] - marker[0]
                x = x / np.linalg.norm(x)
                table_to_marker = np.eye(4)
                table_to_marker[:2, 0] = x
                table_to_marker[:2, 1] = [-x[1], x[0]]
                table_to_marker[:2, 3] = marker.mean(axis=0)
                self.table_to_base = self.rigid(reference["marker_to_base_mm"]) @ np.linalg.inv(table_to_marker)
            self.base_to_table = np.linalg.inv(self.table_to_base)
            self.tool_offset = self.vector(config["tool_offset_mm"], 3) / 1000
            self.tool_radius = float(config["tool_radius_mm"])
            obstacle, clearance = float(config["maximum_obstacle_height_mm"]), float(config["clearance_mm"])
            if not all(math.isfinite(v) and v >= 0 for v in (self.tool_radius, obstacle, clearance)):
                raise ValueError("measured tool envelope, obstacle height, and clearance are required")
            self.minimum_tcp_height = obstacle + clearance + self.tool_radius
            self.safe_height = float(safe_height)
            if not math.isfinite(self.safe_height) or self.safe_height < self.minimum_tcp_height:
                raise ValueError("requested safe height is below the measured clearance envelope")
            self.bounds = np.asarray(config["workspace_bounds_table_mm"], dtype=float)
            if self.bounds.shape != (3, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
                raise ValueError("verified XYZ workspace bounds are required")
            urdf = (ROOT / config["urdf"]).resolve()
            if not urdf.is_relative_to(ROOT) or not urdf.is_file():
                raise ValueError("ElRobot URDF must be an existing file inside this repository")
            tree = ET.parse(urdf).getroot()
            if tree.attrib.get("name", "").lower() != "elrobot":
                raise ValueError("this mapping must use the inspected ElRobot model")
            by_child = {joint.find("child").attrib["link"]: joint for joint in tree.findall("joint")}
            chain, link = [], config["tool_link"]
            while link != "base_link":
                joint = by_child[link]
                name, kind = joint.attrib["name"], joint.attrib["type"]
                origin = joint.find("origin")
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
                matrix = np.eye(4)
                matrix[:3, 3] = xyz
                matrix[:3, :3] = self.rpy(rpy)
                index, axis, limits = None, None, None
                if kind != "fixed":
                    if kind != "revolute" or name not in [f"rev_motor_{i:02}" for i in range(1, 8)]:
                        raise ValueError(f"unsupported or unmapped tool-chain joint {name}")
                    index = int(name[-2:]) - 1
                    axis = self.vector([float(v) for v in joint.find("axis").attrib["xyz"].split()], 3)
                    axis /= np.linalg.norm(axis)
                    limit = joint.find("limit")
                    limits = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
                chain.append((matrix, index, axis, limits))
                link = joint.find("parent").attrib["link"]
            self.chain = list(reversed(chain))
            if {part[1] for part in self.chain if part[1] is not None} != set(range(7)):
                raise ValueError("tool link must follow all seven ElRobot arm joints")
            self.q_min, self.q_max = np.full(7, -np.inf), np.full(7, np.inf)
            for _, index, _, limits in self.chain:
                if index is not None:
                    ends = [(raw - self.zero[index]) * self.scale[index] for raw in self.ranges[index]]
                    self.q_min[index] = max(limits[0], min(ends))
                    self.q_max[index] = min(limits[1], max(ends))
            if np.any(self.q_min >= self.q_max):
                raise ValueError("measured servo angles do not overlap the URDF limits")
            samples = config["verification_samples"]
            if len(samples) < 3:
                raise ValueError("at least three measured FK verification poses are required")
            measured = np.array([self.vector(sample["tcp_table_mm"], 3) for sample in samples])
            if np.linalg.matrix_rank(measured[:, :2] - measured[0, :2]) < 2 or np.ptp(measured[:, 2]) < 10:
                raise ValueError("verification poses must span table XY and at least 10mm of measured Z")
            for sample, point in zip(samples, measured):
                if np.linalg.norm(self.pose(self.vector(sample["ticks"], 8))[:3, 3] - point) > 3:
                    raise ValueError("URDF/servo/table/tool mapping disagrees with a measured TCP pose by over 3mm")
        except (OSError, KeyError, TypeError, ValueError, VisionError, ET.ParseError, np.linalg.LinAlgError) as exc:
            raise AlignmentError(f"ROBOT ALIGNMENT REQUIRED: {exc}.") from exc

    def assert_current(self):
        if self.signature is None:
            return
        try:
            current = hashlib.sha256(ROBOT_ALIGNMENT.read_bytes()).hexdigest()
        except OSError as exc:
            raise AlignmentError("Robot alignment is unavailable; stop and realign.") from exc
        if current != self.signature:
            raise AlignmentError("Robot alignment changed. Click Execute again after completing alignment.")

    def vector(self, value, length):
        vector = self.np.asarray(value, dtype=float)
        if vector.shape != (length,) or not self.np.isfinite(vector).all():
            raise ValueError(f"expected {length} finite measured values")
        return vector

    def rigid(self, value):
        np = self.np
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("a measured rigid 4x4 transform in millimetres is required")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or abs(np.linalg.det(rotation) - 1) > 1e-5:
            raise ValueError("registration rotation must be orthonormal and right handed")
        return matrix

    def rotation(self, axis, angle):
        np = self.np
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)

    def rpy(self, rpy):
        return self.rotation([0, 0, 1], rpy[2]) @ self.rotation([0, 1, 0], rpy[1]) @ self.rotation([1, 0, 0], rpy[0])

    def pose_q(self, q):
        np = self.np
        transform = np.eye(4)
        for origin, index, axis, _ in self.chain:
            transform = transform @ origin
            if index is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = self.rotation(axis, q[index])
                transform = transform @ rotation
        transform[:3, 3] = (transform @ np.append(self.tool_offset, 1))[:3] * 1000
        return self.base_to_table @ transform

    def pose(self, raw):
        return self.pose_q((self.np.asarray(raw) - self.zero) * self.scale)

    def rotation_error(self, target, current):
        matrix = target @ current.T
        angle = math.acos(float(self.np.clip((self.np.trace(matrix) - 1) / 2, -1, 1)))
        skew = self.np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]) / 2
        if angle < 1e-5:
            return skew
        if math.pi - angle < 1e-4:
            _, vectors = self.np.linalg.eigh((matrix + matrix.T + 2 * self.np.eye(3)) / 4)
            return vectors[:, -1] * angle
        return skew * angle / math.sin(angle)

    def checked_point(self, raw, *, approach=True):
        np = self.np
        raw = self.vector(raw, 8)
        if any(not lo <= value <= hi for value, (lo, hi) in zip(raw, self.ranges)):
            raise AlignmentError("Joint target is outside measured servo limits.")
        q = (raw - self.zero) * self.scale
        if np.any(q[:7] < self.q_min - 1e-6) or np.any(q[:7] > self.q_max + 1e-6):
            raise AlignmentError("Joint feedback/target exceeds the verified URDF limits.")
        point = self.pose(raw)[:3, 3]
        if np.any(point < self.bounds[:, 0]) or np.any(point > self.bounds[:, 1]):
            raise AlignmentError("TCP is outside the verified workspace.")
        minimum = self.minimum_tcp_height if approach else self.tool_radius
        if point[2] < minimum:
            raise AlignmentError(f"TCP height {point[2]:.1f}mm is below the verified {minimum:.1f}mm clearance. Raise to a verified safe start pose before retrying.")
        return point

    def check_segment(self, start, end, *, approach=True):
        for fraction in self.np.linspace(0, 1, 11):
            self.checked_point(self.np.asarray(start) * (1 - fraction) + self.np.asarray(end) * fraction, approach=approach)

    def plan(self, target_xy, current_raw, cancel):
        np = self.np
        current_raw = np.asarray(current_raw, dtype=float)
        start = self.checked_point(current_raw)
        target = np.array([target_xy[0], target_xy[1], self.safe_height])
        if np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
            raise AlignmentError("Requested approach position is outside the verified workspace.")
        rotation = self.pose(current_raw)[:3, :3].copy()
        q = ((current_raw - self.zero) * self.scale)[:7]
        previous = current_raw.copy()
        path, deadline = [], time.monotonic() + 20
        # Raise vertically within the already-clear workspace, then move XY.
        corners = [start.copy(), np.array([start[0], start[1], self.safe_height]), target]
        for first, last in zip(corners, corners[1:]):
            count = max(1, int(math.ceil(np.linalg.norm(last - first) / 5)))
            if count > 100:
                raise AlignmentError("Approach path exceeds the prototype planning bound.")
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                desired = first + (last - first) * fraction
                for _ in range(100):
                    cancelled(cancel)
                    if time.monotonic() > deadline:
                        raise AlignmentError("Safe approach IK planning timed out; no commands were sent.")
                    pose = self.pose_q(q)
                    position_error = desired - pose[:3, 3]
                    orientation_error = self.rotation_error(rotation, pose[:3, :3])
                    if np.linalg.norm(position_error) <= 2 and np.linalg.norm(orientation_error) <= .02:
                        break
                    jacobian = np.zeros((6, 7))
                    for index in range(7):
                        perturbed = q.copy()
                        perturbed[index] += .0001
                        future = self.pose_q(perturbed)
                        jacobian[:3, index] = (future[:3, 3] - pose[:3, 3]) / .0001
                        jacobian[3:, index] = self.rotation_error(future[:3, :3], pose[:3, :3]) * 100 / .0001
                    error = np.concatenate([position_error, orientation_error * 100])
                    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6), error)
                    q = np.clip(q + np.clip(delta, -.08, .08), self.q_min, self.q_max)
                else:
                    raise AlignmentError("Target is unreachable with the verified tool orientation and height.")
                raw = current_raw.copy()
                raw[:7] = np.rint(self.zero[:7] + q / self.scale[:7])
                parts = max(1, int(math.ceil(np.max(abs(raw - previous)) / 50)))
                for t in np.linspace(0, 1, parts + 1)[1:]:
                    step = np.rint(previous + (raw - previous) * t)
                    self.check_segment(path[-1] if path else current_raw, step)
                    path.append(step.astype(int).tolist())
                previous = raw
        if np.linalg.norm(self.pose(path[-1])[:3, 3] - target) > 3:
            raise AlignmentError("Rounded joint goals do not reach the requested approach within 3mm.")
        return {"waypoints": path, "target_table_mm": target.tolist(),
                "target_robot_mm": (self.table_to_base @ np.append(target, 1))[:3].tolist(),
                "safe_height_mm": self.safe_height, "tool_radius_mm": self.tool_radius,
                "method": "measured ElRobot URDF FK/IK", "verified": True}
