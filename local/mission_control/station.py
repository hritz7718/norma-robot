"""Read-only Station adapter; follower bus identity is always explicit."""
import asyncio
import copy
import json
import logging
import math
import sys
import threading
import time
from .config import ROOT, FOLLOWER, FRESH_SECONDS, STATION, CAMERAS, MANUAL_CALIBRATION
from .cameras import CameraReader


class ReadOnlyClient:
    __slots__ = ("__client",)
    def __init__(self, client):
        self.__client = client
    def read_from_tail(self, queue, **kwargs):
        if queue not in ("inference/normvla", "st3215/inference"):
            raise PermissionError("Queue is outside the observation allowlist.")
        return self.__client.read_from_tail(queue, **kwargs)


def decode_observation(frame):
    """Decode a captured observation without importing robot control helpers."""
    joints = []
    for i, joint in enumerate(frame.get_joints() or [], 1):
        position = float(joint.get_position_norm())
        joints.append({"name": f"j{i}", "pos_norm": position if math.isfinite(position) else None,
                       "pos": int(joint.get_position()), "range_min": int(joint.get_range_min()),
                       "range_max": int(joint.get_range_max())})
    return {
        "frame_id": bytes(frame.get_global_frame_id()),
        "joints": joints,
        "images": [bytes(image.get_jpeg()) for image in frame.get_images() or []],
        "image_stamps_ns": [int(image.get_monotonic_stamp_ns()) for image in frame.get_images() or []],
    }


def decode_motors(state):
    """Read only the configured follower; never discover or select a different bus."""
    for bus in state.get_buses() or []:
        info = bus.get_bus()
        serial = info.get_serial_number() if info is not None else ""
        if isinstance(serial, (bytes, bytearray, memoryview)):
            serial = bytes(serial).decode("utf-8", "replace")
        if serial != FOLLOWER:
            continue
        motors = []
        for motor in bus.get_motors() or []:
            mid = int(motor.get_id())
            if mid not in range(1, 9):
                continue
            error = motor.get_error()
            kind = int(error.get_kind()) if error else 0
            servo = [int(x) for x in error.get_servo()] if error else []
            raw = bytes(motor.get_state())
            fault = bool(kind or any(servo))
            motors.append({"id": mid, "status": "fault" if fault else "reported",
                           "error_kind": kind, "servo_errors": servo,
                           "error": (error.get_description() or "")[:200] if error else "",
                           "torque_enabled": bool(raw[0x28]) if len(raw) > 0x28 else None,
                           "sample": [motor.get_app_start_id(), motor.get_monotonic_stamp_ns()],
                           "sample_wall_ns": motor.get_system_stamp_ns()})
        return sorted(motors, key=lambda m: m["id"])
    raise RuntimeError(f"Follower {FOLLOWER} is missing from motor telemetry.")


class StationReader:
    def __init__(self, mock=False):
        self.mock = mock
        self.lock = threading.RLock()
        self.data = {"joints": [], "motors": [], "camera_count": 0, "frame_id": None,
                     "observation_count": 0, "error": "Connecting to Station", "motor_error": "Waiting for follower feedback"}
        self.frame_bytes = None
        self.images = []
        self.image_stamps = []
        self.image_changed = {}
        self.raw_cameras = None if mock else CameraReader(STATION)
        self.live_images = {}
        self.camera_metadata = {}
        self.camera_error = "Waiting for raw camera frames."
        self.camera_polled_at = time.monotonic()
        self.on_observation = None
        self.changed = None
        self.motor_changed = {}
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        if self.raw_cameras:
            self.raw_cameras.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mission-observer")
        self._thread.start()

    def close(self):
        self._stop.set()
        if self.raw_cameras:
            self.raw_cameras.close()
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self):
        try:
            asyncio.run(self._worker())
        except Exception as exc:
            with self.lock:
                self.data["error"] = f"Station reader unavailable: {type(exc).__name__}: {exc}"

    async def _worker(self):
        if self.mock:
            await self._mock_worker()
            return
        sys.path[:0] = [str(ROOT), str(ROOT / "software/station/shared")]
        from station_py.client import Client
        from target.gen_python.protobuf.drivers.inferences import normvla
        from target.gen_python.protobuf.drivers.st3215 import st3215
        # Client owns reconnects. Only expose the queue-reading allowlist below.
        client = Client(STATION, logging.getLogger("mission-station"))
        reader = ReadOnlyClient(client)
        async def tail(queue):
            await client.wait_ready(timeout=5)
            qr = reader.read_from_tail(queue, offset=b"\x00", limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(qr.data.get(), timeout=3)
            if entry is None:
                raise RuntimeError(f"No data from {queue}.")
            return entry
        async def observations():
            while not self._stop.is_set():
                try:
                    entry = await tail("inference/normvla")
                    raw = bytes(entry.Data)
                    obs = decode_observation(normvla.FrameReader(memoryview(raw)))
                    self.update_observation(obs, raw)
                except Exception as exc:
                    with self.lock:
                        self.data["error"] = f"Observation unavailable: {type(exc).__name__}: {exc}"
                await asyncio.sleep(0.4)
        async def motor_feedback():
            while not self._stop.is_set():
                try:
                    entry = await tail("st3215/inference")
                    state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
                    motors = decode_motors(state)
                    self.update_motors(motors)
                except Exception as exc:
                    with self.lock:
                        self.data["motor_error"] = str(exc)
                await asyncio.sleep(0.5)
        try:
            await asyncio.gather(observations(), motor_feedback(), self._camera_feedback())
        finally:
            if client.writer:
                client.writer.close()

    async def _camera_feedback(self):
        last = None
        while not self._stop.is_set():
            with self.raw_cameras.lock:
                state = self.raw_cameras.snapshot()
                frames = {queue: dict(value) for queue, value in self.raw_cameras.frames.items()}
            cameras = state["cameras"]
            selected, problem = {}, state["error"]
            try:
                if MANUAL_CALIBRATION.exists():
                    saved = json.loads(MANUAL_CALIBRATION.read_text())
                    uid, queue = saved.get("camera_unique_id"), saved.get("camera_queue")
                    matches = [cam for cam in cameras if (uid and cam["unique_id"] == uid)
                               or (not uid and queue and cam["queue"] == queue)]
                    if len(matches) != 1:
                        raise ValueError("The calibrated tabletop camera is missing or ambiguous. Reconnect it or recalibrate the correct camera.")
                    selected["workspace"] = matches[0]
                    remaining = [cam for cam in cameras if cam["queue"] != matches[0]["queue"]]
                    if len(remaining) == 1:
                        selected["wrist"] = remaining[0]
                else:
                    selected = {name: cameras[index] for name, index in CAMERAS.items() if index < len(cameras)}
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                problem = f"Workspace camera: {exc}"
            with self.lock:
                self.camera_metadata = selected
                self.live_images = {name: frames[cam["queue"]].get("jpeg") for name, cam in selected.items()}
                self.camera_error = problem
                self.camera_polled_at = time.monotonic()
                workspace = selected.get("workspace", {})
                key = (workspace.get("queue"), workspace.get("frame_id"))
                jpeg = self.live_images.get("workspace")
            if not problem and workspace.get("ready") and jpeg and key != last:
                last = key
                if self.on_observation:
                    self.on_observation(jpeg, workspace["frame_id"], {
                        "camera_unique_id": workspace["unique_id"], "camera_queue": workspace["queue"]})
            await asyncio.sleep(.2)

    def update_observation(self, obs, frame_bytes, clock=None):
        clock = time.monotonic() if clock is None else clock
        fid = obs["frame_id"].hex()
        with self.lock:
            if fid != self.data["frame_id"]:
                self.changed = clock
                self.data["observation_count"] += 1
            self.data.update({"frame_id": fid, "joints": copy.deepcopy(obs["joints"]),
                              "camera_count": len(obs["images"]), "error": ""})
            self.frame_bytes = frame_bytes
            self.images = obs["images"][:2]
            stamps = obs.get("image_stamps_ns", [])
            for index, stamp in enumerate(stamps[:2]):
                if stamp and (index >= len(self.image_stamps) or stamp != self.image_stamps[index]):
                    self.image_changed[index] = clock
            self.image_stamps = stamps[:2]
        if self.mock and self.on_observation and len(obs["images"]) > CAMERAS["workspace"]:
            self.on_observation(obs["images"][CAMERAS["workspace"]], fid)

    def update_motors(self, motors, clock=None):
        clock = time.monotonic() if clock is None else clock
        wall = time.time_ns()
        with self.lock:
            previous = {m["id"]: m for m in self.data["motors"]}
            for m in motors:
                mid = m["id"]
                # A cached historical sample must never become fresh on reconnect.
                source_age = max(0, (wall - m.get("sample_wall_ns", 0)) / 1e9)
                if m["sample"] != previous.get(mid, {}).get("sample"):
                    self.motor_changed[mid] = clock - source_age
            self.data["motors"] = motors
            self.data["motor_error"] = "" if len(motors) == 8 else "Incomplete follower motor set"

    def snapshot(self, clock=None):
        clock = time.monotonic() if clock is None else clock
        with self.lock:
            data = copy.deepcopy(self.data)
            age = None if self.changed is None else max(0, clock - self.changed)
            data["observation_age_ms"] = round(age * 1000) if age is not None else None
            data["live"] = age is not None and age < FRESH_SECONDS and not data["error"]
            data["camera_age_ms"] = {name: round(max(0, clock - self.image_changed[index]) * 1000)
                                     if index in self.image_changed else None for name, index in CAMERAS.items()}
            data["camera_live"] = {name: value is not None and value < FRESH_SECONDS * 1000 and data["live"]
                                   for name, value in data["camera_age_ms"].items()}
            # Model input must still contain fresh cameras and real joint feedback.
            policy_cameras_live = all(data["camera_live"].values())
            if self.raw_cameras:
                data["camera_age_ms"] = {
                    name: cam.get("age_ms", 0) + round(max(0, clock - self.camera_polled_at) * 1000)
                    if (cam := self.camera_metadata.get(name)) else None
                    for name in CAMERAS}
                data["camera_live"] = {
                    name: bool(self.camera_metadata.get(name, {}).get("ready") and not self.camera_error
                               and age is not None and age < FRESH_SECONDS * 1000)
                    for name, age in data["camera_age_ms"].items()}
                data["camera_frame_ids"] = {name: self.camera_metadata.get(name, {}).get("frame_id") for name in CAMERAS}
                data["camera_sources"] = copy.deepcopy(self.camera_metadata)
                data["camera_error"] = self.camera_error
            else:
                data["camera_frame_ids"] = {name: data["frame_id"] if index < len(self.images) else None
                                             for name, index in CAMERAS.items()}
            for motor in data["motors"]:
                motor_age = max(0, clock - self.motor_changed.get(motor["id"], -1e9))
                motor["age_ms"] = round(motor_age * 1000)
                if motor_age >= FRESH_SECONDS or data["motor_error"]:
                    motor["status"] = "stale"
            data["ranges_valid"] = len(data["joints"]) == 8 and all(
                type(j["pos_norm"]) in (int, float) and math.isfinite(j["pos_norm"])
                and 0 <= j["pos_norm"] <= 1
                and j["range_max"] > j["range_min"] for j in data["joints"])
            data["preview_ready"] = (data["live"] and data["ranges_valid"] and data["camera_count"] == 2
                                     and policy_cameras_live and all(data["camera_live"].values())
                                     and len(data["motors"]) == 8 and not data["motor_error"]
                                     and all(m["status"] == "reported" for m in data["motors"]))
            data["source"] = "simulation" if self.mock else "NormaCore Station"
            return data

    def capture(self):
        with self.lock:
            snapshot = self.snapshot()
            if not snapshot["preview_ready"]:
                raise RuntimeError("Preview needs a fresh advancing observation, two cameras, calibrated joints, and eight fault-free follower reports.")
            return self.frame_bytes, snapshot

    def camera(self, name):
        with self.lock:
            if self.raw_cameras:
                return self.live_images.get(name)
            index = CAMERAS[name]
            return self.images[index] if index < len(self.images) else None

    def capture_workspace(self):
        with self.lock:
            snapshot = self.snapshot()
            if not snapshot["camera_live"].get("workspace"):
                raise RuntimeError(snapshot.get("camera_error") or "Workspace camera has no fresh advancing image.")
            if self.raw_cameras:
                camera = self.camera_metadata["workspace"]
                snapshot.update(frame_id=camera["frame_id"], camera_unique_id=camera["unique_id"],
                                camera_queue=camera["queue"], camera_source=camera["source"])
            return self.camera("workspace"), snapshot

    def capture_preview(self):
        """Keep prediction input and its two review images from the same frame."""
        with self.lock:
            raw, snapshot = self.capture()
            return raw, snapshot, {name: self.images[index] for name, index in CAMERAS.items()}

    async def _mock_worker(self):
        import io
        from PIL import Image, ImageDraw
        images = []
        for label in ("WRIST", "WORKSPACE"):
            im = Image.new("RGB", (640, 400), "#101f30")
            draw = ImageDraw.Draw(im)
            for x in range(0, 640, 40): draw.line((x, 0, x, 400), fill="#1b3145")
            for y in range(0, 400, 40): draw.line((0, y, 640, y), fill="#1b3145")
            draw.rectangle((270, 175, 345, 250), fill="#3a8dff", outline="#9fc5ff", width=3)
            draw.text((24, 24), f"SIMULATION / {label} / NO HARDWARE", fill="#d1e1ee")
            buf = io.BytesIO(); im.save(buf, format="JPEG"); images.append(buf.getvalue())
        while not self._stop.is_set():
            stamp = time.monotonic_ns()
            joints = [{"name": f"j{i}", "pos_norm": round(0.3 + i * .04, 4), "pos": 1200 + i * 100,
                       "range_min": 0, "range_max": 4095} for i in range(1, 9)]
            self.update_observation({"frame_id": stamp.to_bytes(8, "little"), "joints": joints, "images": images,
                                     "image_stamps_ns": [stamp, stamp]}, b"mock")
            self.update_motors([{"id": i, "status": "reported", "error_kind": 0, "servo_errors": [],
                                 "error": "", "torque_enabled": False, "sample": [0, stamp],
                                 "sample_wall_ns": time.time_ns()} for i in range(1, 9)])
            await asyncio.sleep(.4)
