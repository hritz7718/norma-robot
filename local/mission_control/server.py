"""Loopback-only Mission Control API with an explicitly started SmolVLA loop."""
import argparse
import copy
import json
import logging
import math
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from .config import (HERE, ROOT, CHECKPOINT, CHECKPOINT_LABEL, VLA_PYTHON,
                     EPISODE_STATE, EPISODE_URL, FOLLOWER, LEADER, STATION, CAMERAS,
                     DEFAULT_RUN_SECONDS, DEFAULT_MAX_DELTA_TICKS,
                     MAX_RUN_SECONDS, MAX_DELTA_TICKS_LIMIT, REQUESTED_SAFE_HEIGHT_MM)
from .interpreter import Interpreter
from .execution import Execution
from .preview import predict
from .station import StationReader
from .tasks import TaskError, build_task, prompt_text


class BusyError(ValueError):
    pass


class MissionControl:
    def __init__(self, mock=False, station=None, interpreter=None, *,
                 run_seconds=DEFAULT_RUN_SECONDS, max_delta_ticks=DEFAULT_MAX_DELTA_TICKS,
                 safe_height=REQUESTED_SAFE_HEIGHT_MM):
        self.mock = mock
        self.lock = threading.RLock()
        self.station = station or StationReader(mock=mock)
        self.interpreter = interpreter or Interpreter(mock=mock)
        self.task = None
        self.preview = None
        self.preview_frames = {}
        self.job = None
        self.logs = deque(maxlen=150)
        self.log_sequence = 0
        self.token = secrets.token_urlsafe(32)
        self.execution = Execution(self.station, self.log, self.lock, duration=run_seconds,
                                   max_delta_ticks=max_delta_ticks, mock=mock, interpreter=self.interpreter,
                                   safe_height=safe_height)
        self.log("info", "Session started. Execute calibrates the camera, detects the blue block, checks robot alignment, approaches at safe height, then hands off to SmolVLA.")
        if mock:
            self.log("warning", "SIMULATION: cameras, telemetry, interpretation and predictions are synthetic.")

    def log(self, level, message):
        with self.lock:
            self.log_sequence += 1
            self.logs.append({"id": self.log_sequence, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "level": level, "message": message})

    def interpret(self, payload):
        if set(payload) != {"prompt"}:
            raise TaskError("Only a prompt field is accepted.")
        prompt = prompt_text(payload["prompt"])
        if self.interpreter.error:
            raise RuntimeError(self.interpreter.error)
        def work():
            task = build_task(self.interpreter.interpret(prompt), prompt)
            with self.lock:
                self.task = task
            self.log("info", f"Validated {task['task_type']} task. Execute independently interprets the current UI instruction before movement.")
        self._start("interpret", work, clear_task=True)

    def start_preview(self, payload):
        if set(payload) != {"task_id"}:
            raise TaskError("Preview accepts only a validated task_id.")
        with self.lock:
            if not self.task or payload["task_id"] != self.task["id"]:
                raise TaskError("Review a current validated task before previewing.")
            task = copy.deepcopy(self.task)
            if task["task_type"] != "pick_up":
                raise TaskError("Inspection has no motor prediction. Use a pickup task for VLA preview.")
            raw, observation, images = self.station.capture_preview()
            def work():
                result = predict(task, self.station, mock=self.mock, captured=(raw, observation))
                with self.lock:
                    self.preview = result
                    self.preview_frames = images
                self.log("info", "Offline VLA preview complete. Eight predictions; zero motor commands. Compare against its captured observation.")
            self._start("preview", work)

    def _start(self, kind, work, clear_task=False):
        with self.lock:
            if self.execution.snapshot()["active"]:
                raise BusyError("Stop execution before starting another worker.")
            if self.job and self.job["status"] == "running":
                raise BusyError("A worker is already running. Wait for it to finish.")
            self.preview = None
            self.preview_frames = {}
            if clear_task:
                self.task = None
            self.job = {"kind": kind, "status": "running", "started_at": time.time(), "error": None}
            self.log("info", f"{kind.capitalize()} worker started; hardware remains read-only.")
        def run():
            try:
                work()
                with self.lock:
                    self.job["status"] = "complete"
                    self.job["finished_at"] = time.time()
            except Exception as exc:
                message = str(exc)[:500]
                with self.lock:
                    self.job.update(status="failed", error=message, finished_at=time.time())
                self.log("error", f"{kind.capitalize()} failed: {message}")
        threading.Thread(target=run, daemon=True, name=f"mission-{kind}").start()

    def execute(self, payload):
        if "prompt" not in payload or set(payload) - {"prompt", "duration_seconds", "max_delta_ticks", "safe_height_mm"}:
            raise TaskError("Execute accepts prompt, duration_seconds, max_delta_ticks, and safe_height_mm.")
        prompt = prompt_text(payload["prompt"])
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise BusyError("Wait for the current worker before executing.")
            self.execution.start(prompt, duration=payload.get("duration_seconds"),
                                 max_delta_ticks=payload.get("max_delta_ticks"), safe_height=payload.get("safe_height_mm"))

    def stop(self, payload):
        if payload:
            raise TaskError("STOP accepts an empty JSON object.")
        self.execution.stop()

    def episode_lab(self):
        try:
            state = json.loads(EPISODE_STATE.read_text())
            records = []
            for record in state.get("recordings", []):
                if record.get("status") != "ready":
                    continue
                filename = record.get("file")
                available = False
                if isinstance(filename, str) and filename:
                    path = (ROOT / filename).resolve()
                    available = path.is_relative_to(ROOT) and path.is_file()
                records.append({"id": record["id"], "task": record.get("task", ""),
                                "bytes": record.get("bytes", 0), "available": available})
            return {"url": EPISODE_URL, "datasets": records,
                    "ready": sum(r["available"] for r in records),
                    "last_checkpoint": state.get("last_checkpoint"), "error": None}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return {"url": EPISODE_URL, "datasets": [], "ready": 0,
                    "last_checkpoint": None, "error": str(exc)[:300]}

    def preview_camera(self, name, task_id, frame_id):
        with self.lock:
            if (self.preview and self.preview["task_id"] == task_id
                    and self.preview["frame_id"] == frame_id):
                return self.preview_frames.get(name)
        return None

    def camera(self, name):
        if name == "workspace":
            monitored = self.execution.hands.camera()
            if monitored is not None:
                return monitored
        with self.station.lock:
            raw = self.station.camera(name)
            frame_id = self.station.snapshot()["camera_frame_ids"].get(name)
            overlay = self.execution.vision.camera(frame_id) if name == "workspace" else None
            return overlay or raw

    def snapshot(self):
        station = self.station.snapshot()
        faults = []
        if not station["live"]: faults.append("Observation unavailable or stale")
        if station["motor_error"]: faults.append(station["motor_error"])
        if not station["ranges_valid"]: faults.append("Eight calibrated joint positions are required")
        if station["camera_count"] != 2: faults.append("Two observation cameras are required")
        for m in station["motors"]:
            if m["status"] in ("fault", "stale"): faults.append(f"J{m['id']}: {m['status']}")
        with self.lock:
            controller = self.execution.snapshot()
            controller["faults"] = faults
            return {"phase": 2, "mode": "simulation" if self.mock else "execution", "csrf_token": self.token,
                    "server_time": time.time(),
                    "config": {"station": STATION, "follower": FOLLOWER, "leader": LEADER,
                               "checkpoint": CHECKPOINT_LABEL, "camera_mapping": CAMERAS,
                               "camera_mapping_verified": False,
                               "checkpoint_available": self.mock or all((CHECKPOINT / name).is_file()
                                   for name in ("config.json", "model.safetensors", "stats.safetensors")),
                               "vla_python_available": self.mock or VLA_PYTHON.is_file(),
                               "execution_limits": {
                                   "duration_seconds": self.execution.default_duration,
                                   "max_delta_ticks": self.execution.default_max_delta_ticks,
                                   "safe_height_mm": self.execution.suggested_safe_height(),
                                   "max_duration_seconds": MAX_RUN_SECONDS,
                                   "max_delta_ticks_limit": MAX_DELTA_TICKS_LIMIT}},
                    "station": station, "task": copy.deepcopy(self.task), "preview": copy.deepcopy(self.preview),
                    "job": copy.deepcopy(self.job), "logs": list(self.logs), "episode_lab": self.episode_lab(),
                    "interpreter": {"provider": "SIMULATED" if self.mock else "Official Codex CLI",
                                    "available": not bool(self.interpreter.error), "error": self.interpreter.error,
                                    "image_supported": self.interpreter.image_supported},
                    "controller": controller}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "ElRobot/2"
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def _host_ok(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, status, body, content_type="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key, value in (extra or {}).items(): self.send_header(key, str(value))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "Loopback Host required."})
        path = urlparse(self.path).path
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/static/app.js": ("app.js", "application/javascript; charset=utf-8"),
                 "/static/styles.css": ("styles.css", "text/css; charset=utf-8")}
        if path in files:
            name, mime = files[path]
            try:
                return self._send(200, (HERE / "static" / name).read_bytes(), mime)
            except OSError:
                return self._send(503, {"error": "Mission Control UI assets are unavailable."})
        if path == "/api/state":
            return self._send(200, self.server.app.snapshot())
        cameras = {f"/api/camera/{name}.jpg": name for name in CAMERAS}
        if path in cameras:
            frame = self.server.app.camera(cameras[path])
            if frame:
                state = self.server.app.station.snapshot()
                return self._send(200, frame, "image/jpeg", {"X-Observation-Age-Ms": state["observation_age_ms"], "X-Observation-Live": str(state["live"]).lower()})
            return self._send(503, {"error": "Camera is unavailable; no placeholder is presented as live."})
        preview_cameras = {f"/api/preview/camera/{name}.jpg": name for name in CAMERAS}
        if path in preview_cameras:
            query = parse_qs(urlparse(self.path).query)
            frame = self.server.app.preview_camera(preview_cameras[path],
                query.get("task_id", [""])[0], query.get("frame_id", [""])[0])
            if frame:
                return self._send(200, frame, "image/jpeg", {"X-Observation-Live": "false"})
            return self._send(404, {"error": "This captured preview frame is no longer available."})
        return self._send(404, {"error": "Not found"})

    def do_POST(self):
        if not self._host_ok(): return self._send(403, {"error": "Loopback Host required."})
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self._send(403, {"error": "Same-origin requests only."})
        if not secrets.compare_digest(self.headers.get("X-Mission-Token", ""), self.server.app.token):
            return self._send(403, {"error": "Missing or invalid session token."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192: return self._send(413, {"error": "Request body must be 1–8192 bytes."})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(415, {"error": "JSON required."})
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise TaskError("Incomplete JSON body.")
            payload = json.loads(raw)
            if not isinstance(payload, dict): raise TaskError("Expected a JSON object.")
            path = urlparse(self.path).path
            if path == "/api/execute":
                self.server.app.execute(payload)
                return self._send(202, {"accepted": True, "message": "Calibration and automatic picking workflow started."})
            if path == "/api/stop":
                self.server.app.stop(payload)
                return self._send(200, {"accepted": True, "message": "Execution cancelled. No new motor commands; torque unchanged."})
            if path in ("/api/approve", "/api/pause", "/api/move", "/api/teleop"):
                return self._send(403, {"error": "Use Execute or STOP. Other motor-control routes are unavailable."})
            if path == "/api/interpret": self.server.app.interpret(payload)
            elif path == "/api/preview": self.server.app.start_preview(payload)
            else: return self._send(404, {"error": "Not found"})
            self._send(202, {"accepted": True, "message": "Read-only worker started; no motor commands."})
        except BusyError as exc: self._send(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc: self._send(400, {"error": str(exc)[:500]})
        except RuntimeError as exc: self._send(409, {"error": str(exc)[:500]})
        except TimeoutError: self._send(408, {"error": "Request body timed out."})

    def log_message(self, fmt, *args):
        logging.getLogger("mission-http").debug(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="ElRobot Mission Control with SmolVLA execution")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--mock", action="store_true", help="Simulate all inputs; never connect to hardware or Codex")
    parser.add_argument("--run-seconds", type=float, default=DEFAULT_RUN_SECONDS,
                        help=f"Default run duration after model loading (default: {DEFAULT_RUN_SECONDS:g}s, maximum: {MAX_RUN_SECONDS:g}s)")
    parser.add_argument("--max-delta-ticks", type=int, default=DEFAULT_MAX_DELTA_TICKS,
                        help=f"Default movement limit per joint per action (default: {DEFAULT_MAX_DELTA_TICKS} ticks)")
    parser.add_argument("--safe-height-mm", type=float, default=REQUESTED_SAFE_HEIGHT_MM,
                        help="Requested TCP approach height above the table; measured alignment is required (default: 200mm)")
    args = parser.parse_args()
    if not math.isfinite(args.run_seconds) or not 0 < args.run_seconds <= MAX_RUN_SECONDS:
        parser.error(f"--run-seconds must be greater than zero and at most {MAX_RUN_SECONDS:g} seconds.")
    if not 1 <= args.max_delta_ticks <= MAX_DELTA_TICKS_LIMIT:
        parser.error(f"--max-delta-ticks must be between 1 and {MAX_DELTA_TICKS_LIMIT}.")
    if not math.isfinite(args.safe_height_mm) or not 1 <= args.safe_height_mm <= 500:
        parser.error("--safe-height-mm must be 1–500mm and fit the measured workspace.")
    logging.basicConfig(level=logging.WARNING)
    app = MissionControl(mock=args.mock, run_seconds=args.run_seconds, max_delta_ticks=args.max_delta_ticks,
                         safe_height=args.safe_height_mm)
    server = Server(("127.0.0.1", args.port), app)
    app.station.start()
    print(f"ElRobot Mission Control: http://127.0.0.1:{server.server_port} ({'SIMULATION' if args.mock else 'STOPPED — click Execute to move'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.execution.close()
        app.station.close()
        server.server_close()


if __name__ == "__main__": main()
