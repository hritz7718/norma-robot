"""Bench Mat calibration and a small blue-block detector. No robot commands."""
import copy
import hashlib
import json
import math
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

from .config import BENCH_MAT, CAMERA_CALIBRATION, CAMERAS, STATION, MANUAL_CALIBRATION


class VisionError(RuntimeError):
    pass


def cancelled(cancel):
    if cancel is not None and cancel.is_set():
        from concurrent.futures import CancelledError
        raise CancelledError()


class Vision:
    def __init__(self):
        self.lock = threading.RLock()
        self.cv = self.np = None
        self.layout = self.record = None
        self.last_verified = None
        self.overlay = None
        self.data = {"calibration": "NOT CALIBRATED", "reason": None,
                     "detection": None, "marker_ids": [], "overlay_frame_id": None}

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.data)
            result["last_verified_age_ms"] = None if self.last_verified is None else round((time.monotonic() - self.last_verified) * 1000)
            return result

    def _imports(self):
        if self.cv is None:
            try:
                import cv2
                import numpy as np
                self.cv, self.np = cv2, np
            except ImportError as exc:
                raise VisionError("OpenCV and NumPy are required in the SmolVLA environment.") from exc

    def _layout(self):
        try:
            if not hasattr(self.cv, "aruco"):
                raise VisionError("Automatic marker calibration requires opencv-contrib-python-headless; alternatively save a manual four-point calibration.")
            raw = BENCH_MAT.read_bytes()
            layout = json.loads(raw)
            dictionary = layout.get("dictionary")
            if dictionary != "AUTO" and (not isinstance(dictionary, str) or not dictionary.startswith("DICT_") or not hasattr(self.cv.aruco, dictionary)):
                raise VisionError("Bench Mat ArUco dictionary is missing. Set its actual printed dictionary in bench_mat.json.")
            reference = layout["scale_reference"]
            if reference.get("physically_measured") is not True:
                raise VisionError("PRINT SCALE REQUIRED: physically measure horizontal and vertical marker-centre distances; printed coordinates alone do not verify scale.")
            horizontal = float(reference["measured_horizontal_centre_distance_mm"])
            vertical = float(reference["measured_vertical_centre_distance_mm"])
            if not all(math.isfinite(v) and v > 0 for v in (horizontal, vertical)):
                raise VisionError("Measured print-scale references must be finite positive millimetres.")
            centres = {int(mid): self.np.asarray(xy, dtype=float) for mid, xy in layout["marker_centres_mm"].items()}
            required = {200, 201, 202, 203, 204, 205}
            if set(centres) != required or any(p.shape != (2,) or not self.np.isfinite(p).all() for p in centres.values()):
                raise VisionError("Bench Mat requires the six supplied marker-centre coordinates, in mm.")
            nominal_x = self.np.linalg.norm(centres[203] - centres[202])
            nominal_y = self.np.linalg.norm(centres[202] - centres[200])
            if nominal_x <= 0 or nominal_y <= 0 or layout.get("coordinate_units") != "mm":
                raise VisionError("Invalid Bench Mat reference geometry or units.")
            scale = self.np.array([horizontal / nominal_x, vertical / nominal_y])
            layout["centres"] = {mid: p * scale for mid, p in centres.items()}
            layout["signature"] = hashlib.sha256(raw).hexdigest()
            layout["scale_xy"] = scale.tolist()
            return layout
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            raise VisionError(f"Invalid Bench Mat measurements: {exc}") from exc

    def decode(self, jpeg):
        self._imports()
        image = self.cv.imdecode(self.np.frombuffer(jpeg, dtype=self.np.uint8), self.cv.IMREAD_COLOR)
        if image is None:
            raise VisionError("Workspace camera JPEG could not be decoded.")
        return image

    def markers(self, image, layout):
        cv, np = self.cv, self.np
        dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, layout["dictionary"]))
        parameters = cv.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv.aruco.ArucoDetector(dictionary, parameters).detectMarkers(image)
        found = {}
        for mid, quad in zip([] if ids is None else ids.reshape(-1), corners):
            mid, quad = int(mid), np.asarray(quad, dtype=np.float32).reshape(4, 2)
            if mid in found:
                raise VisionError(f"Duplicate ArUco ID {mid}; calibration is ambiguous.")
            if not cv.isContourConvex(quad) or abs(cv.contourArea(quad)) < 9:
                continue
            found[mid] = quad
        return found

    def resolve_dictionary(self, image, layout, cancel):
        if layout["dictionary"] != "AUTO":
            return
        required, matches = set(layout["centres"]), []
        for name in sorted(name for name in dir(self.cv.aruco) if name.startswith("DICT_")):
            cancelled(cancel)
            dictionary = self.cv.aruco.getPredefinedDictionary(getattr(self.cv.aruco, name))
            if len(dictionary.bytesList) <= max(required):
                continue
            candidate = dict(layout, dictionary=name)
            try:
                found = self.markers(image, candidate)
                if not required.issubset(found):
                    continue
                self._geometry(image, candidate, found)
                matches.append((len(dictionary.bytesList), name, found))
            except VisionError:
                continue
        if not matches:
            raise VisionError("No ArUco dictionary matched all six Bench Mat IDs and their geometry. Show the full mat or set its printed dictionary explicitly.")
        matches.sort(key=lambda item: (item[0], item[1]))
        baseline = matches[0][2]
        for _, _, found in matches[1:]:
            if any(self.np.linalg.norm(found[mid] - baseline[mid]) > 1 for mid in required):
                raise VisionError("ArUco dictionary is ambiguous; specify the dictionary printed on the sheet.")
        layout["dictionary"] = matches[0][1]
        layout["compatible_dictionaries"] = [name for _, name, _ in matches]

    def project(self, points, matrix):
        np = self.np
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
        if not np.isfinite(homogeneous).all() or np.any(abs(homogeneous[:, 2]) < 1e-9):
            raise VisionError("Homography projects through a singularity.")
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    def marker_centre(self, quad):
        # A square's perspective centre is the diagonal intersection, not the
        # arithmetic mean of its four image corners.
        np = self.np
        points = np.column_stack([quad, np.ones(4)])
        centre = np.cross(np.cross(points[0], points[2]), np.cross(points[1], points[3]))
        if abs(centre[2]) < 1e-9:
            raise VisionError("Degenerate ArUco marker diagonals.")
        return centre[:2] / centre[2]

    def _geometry(self, image, layout, found):
        cv, np = self.cv, self.np
        missing = set(layout["centres"]) - set(found)
        if missing:
            raise VisionError("ArUco markers not visible: " + ", ".join(map(str, sorted(missing))))
        order = [int(mid) for mid in layout["outer_corner_ids"]]
        if set(order) != {200, 201, 202, 203}:
            raise VisionError("Outer markers must be 202, 203, 201, 200 in perimeter order.")
        pixels = np.array([self.marker_centre(found[mid]) for mid in order], dtype=np.float32)
        table = np.array([layout["centres"][mid] for mid in order], dtype=np.float32)
        if not cv.isContourConvex(pixels) or abs(cv.contourArea(pixels)) < image.shape[0] * image.shape[1] * .01:
            raise VisionError("Bench Mat corner geometry is degenerate or too small in the image.")
        outer_h = cv.getPerspectiveTransform(table, pixels)
        check_table = [layout["centres"][mid] for mid in (204, 205)]
        check_pixels = np.array([self.marker_centre(found[mid]) for mid in (204, 205)])
        holdout = float(np.linalg.norm(self.project(check_table, outer_h) - check_pixels, axis=1).max())
        if holdout > 3:
            raise VisionError(f"Independent centre-marker reprojection error is {holdout:.2f}px (limit 3px).")
        ids = sorted(layout["centres"])
        table = np.array([layout["centres"][mid] for mid in ids])
        pixels = np.array([self.marker_centre(found[mid]) for mid in ids])
        h, _ = cv.findHomography(table, pixels, 0)
        if h is None or not np.isfinite(h).all() or np.linalg.cond(h) > 1e12:
            raise VisionError("Bench Mat homography is singular or ill-conditioned.")
        error = float(np.linalg.norm(self.project(table, h) - pixels, axis=1).max())
        if error > 2:
            raise VisionError(f"Calibration reprojection error {error:.2f}px exceeds 2px.")
        return h, np.linalg.inv(h), error, holdout

    def _manual(self, image, frame_id, camera):
        """Use the separate calibrator's measured points without requiring ArUco."""
        np, cv = self.np, self.cv
        try:
            raw = MANUAL_CALIBRATION.read_bytes()
            saved = json.loads(raw)
            if saved.get("method") != "manual_four_point" or saved.get("units") != "mm":
                raise ValueError("expected a manual four-point calibration in millimetres")
            uid, queue = saved.get("camera_unique_id"), saved.get("camera_queue")
            if not camera or not ((uid and uid == camera.get("camera_unique_id")) or
                                  (not uid and queue and queue == camera.get("camera_queue"))):
                raise ValueError("saved calibration belongs to a different camera; reconnect that camera or recalibrate")
            size = [image.shape[1], image.shape[0]]
            if saved["image_size"] != size:
                raise ValueError(f"camera resolution is {size}, but the saved calibration uses {saved['image_size']}; recalibrate at this resolution")
            pixels = np.asarray(saved["pixel_points"], dtype=float)
            table = np.asarray(saved["table_points_mm"], dtype=float)
            inverse = np.asarray(saved["image_to_table"], dtype=float)
            if (pixels.shape != (4, 2) or table.shape != (4, 2) or inverse.shape != (3, 3)
                    or not all(np.isfinite(value).all() for value in (pixels, table, inverse))):
                raise ValueError("invalid calibration points or homography")
            if (np.any(pixels < 0) or np.any(pixels[:, 0] >= size[0]) or np.any(pixels[:, 1] >= size[1])
                    or abs(cv.contourArea(cv.convexHull(pixels.astype(np.float32)))) < 4
                    or abs(cv.contourArea(cv.convexHull(table.astype(np.float32)))) < 1
                    or np.linalg.cond(inverse) > 1e12):
                raise ValueError("calibration points are degenerate or outside the image")
            forward = np.linalg.inv(inverse)
            error = float(np.linalg.norm(self.project(table, forward) - pixels, axis=1).max())
            error_mm = float(np.linalg.norm(self.project(pixels, inverse) - table, axis=1).max())
            denominators = np.column_stack([pixels, np.ones(4)]) @ inverse[2]
            if error > .1 or error_mm > .1 or not (np.all(denominators > 0) or np.all(denominators < 0)):
                raise ValueError("homography does not match the saved point pairs")
            signature = hashlib.sha256(raw).hexdigest()
            layout = {"mat_id": saved.get("mat_id", "manual-tabletop"), "dictionary": "MANUAL",
                      "centres": {index: xy for index, xy in enumerate(table)}, "signature": signature}
            record = {"method": "manual_four_point", "mat_id": layout["mat_id"], "layout_signature": signature,
                      "camera_source": camera.get("camera_source"), "camera_unique_id": camera.get("camera_unique_id"),
                      "camera_queue": camera.get("camera_queue"), "image_size": size, "frame_id": frame_id,
                      "dictionary": "MANUAL", "marker_ids": [], "image_to_table": inverse.tolist(),
                      "table_to_image": forward.tolist(), "max_error_px": error, "max_error_mm": error_mm,
                      "independent_error_px": None, "saved_at": saved.get("saved_at"), "reused": True}
            return layout, record, {}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, np.linalg.LinAlgError) as exc:
            raise VisionError(f"Saved manual calibration is invalid: {exc}") from exc

    def calibrate(self, jpeg, frame_id, cancel, camera=None):
        cancelled(cancel)
        with self.lock:
            self.data.update(calibration="CALIBRATING", reason=None, detection=None)
        try:
            image = self.decode(jpeg)
            if MANUAL_CALIBRATION.exists():
                layout, record, found = self._manual(image, frame_id, camera)
                cancelled(cancel)
                with self.lock:
                    self.layout, self.record = layout, record
                    self.last_verified = time.monotonic()
                    self.data.update(calibration="CALIBRATION SUCCESSFUL", reason=None,
                                     marker_ids=[], calibration_details=record)
                self.observe(jpeg, frame_id, camera)
                problem = self.current_problem()
                if problem:
                    raise VisionError(problem)
                return record
            layout = self._layout()
            self.resolve_dictionary(image, layout, cancel)
            found = self.markers(image, layout)
            forward, inverse, error, holdout = self._geometry(image, layout, found)
            camera = camera.get("camera_source") if camera else f"{STATION}/inference/normvla/images/{CAMERAS['workspace']}"
            size = [image.shape[1], image.shape[0]]
            reused = False
            try:
                cached = json.loads(CAMERA_CALIBRATION.read_text())
                if cached["layout_signature"] == layout["signature"] and cached["camera_source"] == camera and cached["image_size"] == size and cached["dictionary"] == layout["dictionary"]:
                    ids = sorted(layout["centres"])
                    actual = self.np.array([self.marker_centre(found[mid]) for mid in ids])
                    cached_h = self.np.asarray(cached["table_to_image"], dtype=float)
                    drift = self.np.linalg.norm(self.project([layout["centres"][mid] for mid in ids], cached_h) - actual, axis=1).max()
                    if drift <= 2:
                        forward, inverse = cached_h, self.np.linalg.inv(cached_h)
                        error = float(drift)
                        reused = True
            except (OSError, ValueError, KeyError, TypeError, VisionError, self.np.linalg.LinAlgError):
                pass
            record = {"mat_id": layout["mat_id"], "layout_signature": layout["signature"],
                      "camera_source": camera, "image_size": size, "frame_id": frame_id,
                      "dictionary": layout["dictionary"], "marker_ids": sorted(layout["centres"]),
                      "compatible_dictionaries": layout.get("compatible_dictionaries", [layout["dictionary"]]),
                      "scale_xy": layout["scale_xy"], "image_to_table": inverse.tolist(),
                      "table_to_image": forward.tolist(), "max_error_px": error,
                      "independent_error_px": holdout, "saved_at": time.time(), "reused": reused}
            cancelled(cancel)
            CAMERA_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
            temporary = CAMERA_CALIBRATION.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
            temporary.replace(CAMERA_CALIBRATION)
            with self.lock:
                self.layout, self.record = layout, record
                self.last_verified = time.monotonic()
                self.data.update(calibration="CALIBRATION SUCCESSFUL", reason=None,
                                 marker_ids=sorted(found), calibration_details=record)
            self.observe(jpeg, frame_id)
            problem = self.current_problem()
            if problem:
                raise VisionError(problem)
            return record
        except CancelledError:
            with self.lock:
                self.data.update(calibration="NOT CALIBRATED", reason="Calibration cancelled.")
            raise
        except Exception as exc:
            with self.lock:
                self.data.update(calibration="CALIBRATION FAILED", reason=str(exc))
            raise

    def _blue(self, image, found, bbox=None):
        cv, np = self.cv, self.np
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array([90, 85, 45]), np.array([135, 255, 255]))
        for quad in found.values():
            cv.fillConvexPoly(mask, quad.astype(np.int32), 0)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            roi = np.zeros_like(mask)
            roi[y0:y1, x0:x1] = 255
            mask = cv.bitwise_and(mask, roi)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv.contourArea(contour)
            x, y, width, height = cv.boundingRect(contour)
            if area < max(20, image.shape[0] * image.shape[1] * .0005) or area > image.shape[0] * image.shape[1] * .2:
                continue
            if min(width, height) < 4 or max(width, height) / min(width, height) > 3:
                continue
            solidity = area / max(1, cv.contourArea(cv.convexHull(contour)))
            fill = area / (width * height)
            score = .6 * solidity + .4 * fill
            if score < .75:
                continue
            moments = cv.moments(contour)
            if moments["m00"] <= 0:
                continue
            candidates.append({"pixel": {"u": moments["m10"] / moments["m00"], "v": moments["m01"] / moments["m00"]},
                               "bbox": [x, y, x + width, y + height], "confidence": round(score, 3), "area": area})
        candidates.sort(key=lambda c: c["area"], reverse=True)
        if not candidates:
            raise VisionError("Blue block not detected confidently by OpenCV.")
        if len(candidates) > 1 and candidates[1]["area"] >= candidates[0]["area"] * .5:
            raise VisionError("Multiple plausible blue objects; refusing an ambiguous pick.")
        return candidates[0]

    def detect(self, jpeg, frame_id, visual=None):
        image = self.decode(jpeg)
        with self.lock:
            layout, record = self.layout, self.record
        if record is None:
            raise VisionError("Calibration must finish before object detection.")
        manual = record.get("method") == "manual_four_point"
        found = {} if manual else self.markers(image, layout)
        if self.current_problem():
            raise VisionError(self.current_problem())
        if [image.shape[1], image.shape[0]] != record["image_size"]:
            raise VisionError("Workspace camera resolution changed; recalibration is required.")
        if not manual:
            if not set(layout["centres"]).issubset(found):
                raise VisionError("All six Bench Mat markers must be visible when locating the object.")
            predicted = self.project([layout["centres"][mid] for mid in sorted(layout["centres"])], record["table_to_image"])
            actual = self.np.array([self.marker_centre(found[mid]) for mid in sorted(layout["centres"])])
            if self.np.linalg.norm(predicted - actual, axis=1).max() > 3:
                raise VisionError("Camera/Bench Mat moved since calibration. Recalibrate before detecting a target.")
        bbox = None
        if visual is not None:
            if visual["detected"] is not True or visual["confidence"] < .6:
                raise VisionError("Codex did not identify the requested object confidently in the attached image.")
            box = visual["bbox"]
            x0, y0, x1, y1 = [box[key] for key in ("x0", "y0", "x1", "y1")]
            height, width = image.shape[:2]
            if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
                raise VisionError("Codex bounding box is outside the actual captured image.")
            padding = max(4, round(max(x1 - x0, y1 - y0) * .15))
            bbox = [max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding)]
        candidate = self._blue(image, found, bbox)
        u, v = candidate["pixel"]["u"], candidate["pixel"]["v"]
        x, y = self.project([[u, v]], record["image_to_table"])[0]
        # The printed references establish coordinates, not the task boundary.
        # Reachability and workspace limits are checked by the robot mapping.
        candidate.update(object="blue block", detected=True, table_mm={"x": float(x), "y": float(y)},
                         source="codex+opencv" if visual is not None else "opencv", frame_id=frame_id,
                         captured_at=time.time(), position_note="Approximate tabletop projection; visible centroid may be above the table.")
        candidate.pop("area", None)
        with self.lock:
            self.data["detection"] = candidate
        self.observe(jpeg, frame_id)
        return candidate

    def current_problem(self):
        with self.lock:
            manual = self.record and self.record.get("method") == "manual_four_point"
        path = MANUAL_CALIBRATION if manual else BENCH_MAT
        try:
            signature = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return "Saved manual calibration is unavailable." if manual else "Bench Mat definition is unavailable."
        with self.lock:
            if self.data["calibration"] != "CALIBRATION SUCCESSFUL":
                return self.data["reason"] or "Camera calibration is unavailable."
            if not self.record or signature != self.record["layout_signature"]:
                return "Calibration changed; click Execute again to load it."
            if self.last_verified is None or time.monotonic() - self.last_verified > 2:
                return "Workspace camera has no fresh calibrated frames." if manual else "Bench Mat markers have not been verified for over 2 seconds."
        return None

    def base_marker(self, marker_id):
        with self.lock:
            quad = (self.data.get("marker_corners") or {}).get(str(marker_id))
            matrix = self.record["image_to_table"] if self.record else None
        if quad is None or matrix is None:
            raise VisionError(f"Robot reference marker {marker_id} is not visible.")
        return self.project(quad, matrix)

    def observe(self, jpeg, frame_id, camera=None):
        """Called by the existing observer, never by an HTTP/STOP handler."""
        with self.lock:
            layout, record = self.layout, self.record
        if record is None:
            return
        try:
            cv, np = self.cv, self.np
            image = self.decode(jpeg)
            if [image.shape[1], image.shape[0]] != record["image_size"]:
                raise VisionError("Workspace camera resolution changed; recalibration is required.")
            manual = record.get("method") == "manual_four_point"
            if manual:
                uid = record.get("camera_unique_id")
                if camera is not None and not ((uid and uid == camera.get("camera_unique_id")) or
                        (not uid and record.get("camera_queue") == camera.get("camera_queue"))):
                    raise VisionError("Workspace camera changed; recalibration is required.")
                found = {}
                if camera is not None:
                    with self.lock:
                        self.last_verified = time.monotonic()
            else:
                found = self.markers(image, layout)
            if not manual and set(layout["centres"]).issubset(found):
                actual = np.array([self.marker_centre(found[mid]) for mid in sorted(layout["centres"])])
                expected = self.project([layout["centres"][mid] for mid in sorted(layout["centres"])], record["table_to_image"])
                error = float(np.linalg.norm(actual - expected, axis=1).max())
                if error > 3:
                    raise VisionError(f"Camera or Bench Mat moved ({error:.1f}px marker drift); recalibration required.")
                with self.lock:
                    self.last_verified = time.monotonic()
            for mid, quad in found.items():
                cv.polylines(image, [quad.astype(np.int32)], True, (0, 220, 0), 1)
                cv.putText(image, str(mid), tuple(quad[0].astype(int)), cv.FONT_HERSHEY_SIMPLEX, .35, (0, 220, 0), 1)
            span = np.ptp(np.array(list(layout["centres"].values())), axis=0)
            length = float(min(span) * .25)
            origin = [float(np.mean([p[0] for p in layout["centres"].values()])),
                      float(min(p[1] for p in layout["centres"].values()))]
            axes = self.project([origin, [origin[0] + length, origin[1]], [origin[0], origin[1] + length]], record["table_to_image"]).astype(int)
            cv.putText(image, f"{origin[0]:.0f},{origin[1]:.0f}", tuple(axes[0]), cv.FONT_HERSHEY_SIMPLEX, .3, (220, 220, 220), 1)
            for index, label, color in ((1, "X", (0, 0, 255)), (2, "Y", (0, 220, 0))):
                cv.arrowedLine(image, tuple(axes[0]), tuple(axes[index]), color, 1)
                cv.putText(image, label, tuple(axes[index]), cv.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            # Detect against unannotated pixels; overlays must not become objects.
            original = self.decode(jpeg)
            try:
                blue = self._blue(original, found)
                u, v = blue["pixel"]["u"], blue["pixel"]["v"]
                xy = self.project([[u, v]], record["image_to_table"])[0]
                x0, y0, x1, y1 = blue["bbox"]
                cv.rectangle(image, (x0, y0), (x1, y1), (255, 160, 0), 1)
                cv.drawMarker(image, (round(u), round(v)), (255, 180, 0), cv.MARKER_CROSS, 9, 1)
                cv.putText(image, f"blue {xy[0]:.0f},{xy[1]:.0f}mm", (max(0, x0), max(10, y0 - 4)), cv.FONT_HERSHEY_SIMPLEX, .35, (255, 200, 0), 1)
            except VisionError:
                pass
            gripper = layout.get("gripper_marker_id")
            if gripper is not None and int(gripper) in found:
                centre = self.marker_centre(found[int(gripper)]).astype(int)
                cv.drawMarker(image, tuple(centre), (255, 0, 255), cv.MARKER_CROSS, 10, 1)
                cv.putText(image, "gripper (pixel)", tuple(centre), cv.FONT_HERSHEY_SIMPLEX, .35, (255, 0, 255), 1)
            ok, encoded = cv.imencode(".jpg", image)
            if ok:
                with self.lock:
                    self.overlay = encoded.tobytes()
                    self.data.update(overlay_frame_id=frame_id, marker_ids=sorted(found),
                                     marker_corners={str(mid): quad.tolist() for mid, quad in found.items()})
        except Exception as exc:
            with self.lock:
                self.data.update(calibration="CALIBRATION FAILED", reason=str(exc))

    def camera(self, frame_id):
        with self.lock:
            return self.overlay if self.data["calibration"] == "CALIBRATION SUCCESSFUL" and self.data["overlay_frame_id"] == frame_id else None
