"""MediaPipe workspace monitoring. Software STOP only; never writes to Station."""
import copy
import math
import threading
import time

from .config import HAND_MODEL, HAND_MAX_AGE_SECONDS, HAND_EXCLUSION_RECT


CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


class HandMonitor:
    def __init__(self, on_trip, vision):
        self.on_trip, self.vision = on_trip, vision
        self.lock = threading.RLock()
        self.wake, self.closed = threading.Event(), threading.Event()
        self.latest = None
        self.result_at = self.source_at = None
        self.overlay = None
        self.latched = False
        self.data = {"level": "STARTING", "ready": False, "reason": "Starting hand detection…",
                     "frame_id": None, "hands": [], "inference_ms": None, "zone_source": None}
        self.thread = threading.Thread(target=self._run, name="workspace-hands", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.closed.set(); self.wake.set()
        self.thread.join(timeout=1)

    def submit(self, jpeg, frame_id, camera=None):
        # Overwrite pending input: never build a queue of old camera frames.
        with self.lock:
            self.latest = (jpeg, frame_id, camera or {}, time.monotonic())
        self.wake.set()

    def _problem(self, ignore_latch=False):
        if not self.data["ready"]:
            return self.data["reason"]
        if self.result_at is None or self.source_at is None or time.monotonic() - min(self.result_at, self.source_at) > HAND_MAX_AGE_SECONDS:
            return "Hand detection or workspace camera is stale. VLA STOP."
        if self.data["level"] in ("RED", "ERROR"):
            return self.data["reason"]
        if self.latched and not ignore_latch:
            return "Hand monitor STOP is latched. Clear the zone, then click Execute to restart."
        return None

    def problem(self):
        with self.lock:
            problem = self._problem()
            if problem and self.data["ready"]:
                self.latched = True
            return problem

    def rearm(self):
        with self.lock:
            problem = self._problem(ignore_latch=True)
            if problem:
                raise RuntimeError(problem)
            self.latched = False

    def snapshot(self):
        with self.lock:
            data = copy.deepcopy(self.data)
            problem = self._problem(ignore_latch=True)
            if problem and data["level"] not in ("STARTING", "RED", "ERROR"):
                data.update(level="ERROR", reason=problem, ready=False)
            data.update(latched=self.latched, restart_allowed=problem is None,
                        overlay_frame_id=self.data["frame_id"] if self.overlay is not None and self.source_at is not None and time.monotonic()-self.source_at <= HAND_MAX_AGE_SECONDS else None,
                        age_ms=None if self.source_at is None else round((time.monotonic()-self.source_at)*1000))
            return data

    def camera(self):
        with self.lock:
            return self.overlay if self.source_at is not None and time.monotonic()-self.source_at <= HAND_MAX_AGE_SECONDS else None

    def _fail(self, reason):
        with self.lock:
            trip = not self.latched
            self.latched = True
            self.data.update(level="ERROR", ready=False, reason=reason, hands=[])
            self.overlay = None
        if trip:
            self.on_trip(reason)

    def _zones(self, cv, np, shape):
        height, width = shape[:2]
        exclusion = np.zeros((height, width), dtype=np.uint8)
        rect = np.asarray(HAND_EXCLUSION_RECT, dtype=float)
        if rect.shape != (4,) or not np.isfinite(rect).all() or not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
            raise RuntimeError("Invalid camera-space hand exclusion rectangle.")
        x0, y0, x1, y1 = [round(v*scale) for v, scale in zip(rect, (width, height, width, height))]
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError("Camera-space hand exclusion rectangle is too small.")
        exclusion[y0:y1, x0:x1] = 255
        warning = np.full((height, width), 255, dtype=np.uint8)
        return exclusion, warning, "Entire workspace camera; independent image-space exclusion zone"

    def _run(self):
        detector = None
        try:
            import cv2 as cv
            import numpy as np
            import mediapipe as mp
            if not HAND_MODEL.is_file():
                raise RuntimeError(f"MediaPipe model missing: {HAND_MODEL}. Run local/mission_control/setup_hands.sh.")
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=4,
                min_hand_detection_confidence=.5, min_hand_presence_confidence=.5, min_tracking_confidence=.5)
            detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
            last_key, timestamp = None, 0
            while not self.closed.is_set():
                self.wake.wait(timeout=.1); self.wake.clear()
                with self.lock:
                    pending = self.latest
                if pending is None or time.monotonic()-pending[3] > HAND_MAX_AGE_SECONDS:
                    self._fail("Workspace camera is stale or unavailable. VLA STOP.")
                    continue
                jpeg, frame_id, camera, received = pending
                key = (camera.get("camera_unique_id"), camera.get("camera_queue"), frame_id)
                if key == last_key:
                    continue
                last_key = key
                try:
                    started = time.monotonic()
                    image = cv.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError("Workspace camera JPEG cannot be decoded.")
                    height, width = image.shape[:2]
                    exclusion, warning, zone_source = self._zones(cv, np, image.shape)
                    timestamp = max(timestamp+1, time.monotonic_ns()//1_000_000)
                    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))), timestamp)
                    hands, level = [], "CLEAR"
                    for landmarks in result.hand_landmarks:
                        if len(landmarks) != 21 or any(not math.isfinite(v) for p in landmarks for v in (p.x, p.y, p.z)):
                            raise RuntimeError("MediaPipe returned invalid hand landmarks.")
                        points = [(max(0, min(width-1, round(p.x*width))), max(0, min(height-1, round(p.y*height)))) for p in landmarks]
                        x0, y0 = max(0, min(p[0] for p in points)-5), max(0, min(p[1] for p in points)-5)
                        x1, y1 = min(width, max(p[0] for p in points)+6), min(height, max(p[1] for p in points)+6)
                        hand_level = "RED" if np.any(exclusion[y0:y1, x0:x1]) else "YELLOW" if np.any(warning[y0:y1, x0:x1]) else "CLEAR"
                        if hand_level == "RED" or hand_level == "YELLOW" and level == "CLEAR":
                            level = hand_level
                        hands.append({"landmarks": points, "bbox": [x0, y0, x1, y1], "level": hand_level})
                    if time.monotonic()-received > HAND_MAX_AGE_SECONDS:
                        raise RuntimeError("Hand detection result arrived too late.")
                    base = self.vision.camera(frame_id)
                    if base is not None:
                        annotated = cv.imdecode(np.frombuffer(base, dtype=np.uint8), cv.IMREAD_COLOR)
                        if annotated is not None and annotated.shape == image.shape:
                            image = annotated
                    for mask, color, label in ((warning, (0, 220, 255), "WARNING"), (exclusion, (0, 50, 255), "EXCLUSION")):
                        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        cv.drawContours(image, contours, -1, color, 1)
                        x, y, _, _ = cv.boundingRect(max(contours, key=cv.contourArea))
                        cv.putText(image, label, (x, max(10, y-3)), cv.FONT_HERSHEY_SIMPLEX, .3, color, 1)
                    for hand in hands:
                        color = (0, 50, 255) if hand["level"] == "RED" else (0, 220, 255) if hand["level"] == "YELLOW" else (100, 230, 100)
                        points, (x0, y0, x1, y1) = hand["landmarks"], hand["bbox"]
                        cv.rectangle(image, (x0, y0), (x1-1, y1-1), color, 2)
                        for a, b in CONNECTIONS:
                            cv.line(image, points[a], points[b], color, 1)
                        for point in points:
                            cv.circle(image, point, 2, color, -1)
                    ok, encoded = cv.imencode(".jpg", image)
                    if not ok:
                        raise RuntimeError("Hand overlay could not be encoded.")
                    reason = {"RED": "Hand in exclusion zone. VLA STOP.", "YELLOW": "Hand approaching robot operating zone.",
                              "CLEAR": "Hand monitor active; exclusion zone clear."}[level]
                    with self.lock:
                        trip = level == "RED" and not self.latched
                        if level == "RED":
                            self.latched = True
                        self.source_at, self.result_at = received, time.monotonic()
                        self.overlay = encoded.tobytes()
                        self.data.update(level=level, ready=True, reason=reason, frame_id=frame_id, hands=hands,
                                         inference_ms=round((time.monotonic()-started)*1000, 1), zone_source=zone_source)
                    if trip:
                        self.on_trip(reason)
                except Exception as exc:
                    self._fail(f"Hand detection failed: {type(exc).__name__}: {str(exc)[:200]}. VLA STOP.")
        except Exception as exc:
            self._fail(f"Hand detector unavailable: {type(exc).__name__}: {str(exc)[:200]}. VLA STOP.")
        finally:
            if detector is not None:
                detector.close()
