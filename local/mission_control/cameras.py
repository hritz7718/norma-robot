"""Raw, read-only Station cameras shared by Mission Control and calibration."""
import asyncio
import logging
import sys
import threading
import time

from .config import ROOT, CAMERAS, FRESH_SECONDS


class CameraReader:
    """Read raw USB camera frames, independently of motors and VLA inference."""
    def __init__(self, server):
        self.server = server
        self.lock = threading.RLock()
        self.frames = {}
        self.error = f"Connecting to Station at {server}…"
        self.stopped = threading.Event()
        self.changed = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="station-cameras")
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1)

    def _run(self):
        try:
            asyncio.run(self._worker())
        except Exception as exc:
            with self.lock:
                self.error = f"Camera reader failed: {type(exc).__name__}: {exc}"
            self.changed.set()

    async def _worker(self):
        sys.path.insert(0, str(ROOT / "software/station/shared"))
        from station_py.client import Client
        from target.gen_python.protobuf.station import inference, drivers
        from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame

        async def read(client, queue, pointer=None):
            if pointer is None:
                reader = client.read_from_tail(queue, offset=b"\x00", limit=1, step=1, buf_size=1)
            else:
                reader = client.read_from_offset(queue, offset=pointer, limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(reader.data.get(), timeout=3)
            if entry is None:
                raise RuntimeError(f"No data from {queue}: {reader.err}")
            return entry

        while not self.stopped.is_set():
            previous_tasks = set(asyncio.all_tasks())
            client = Client(self.server, logging.getLogger("station-cameras"))
            try:
                await client.wait_ready(timeout=5)
                with self.lock:
                    self.frames = {}
                    self.error = "Waiting for raw camera frames…"
                while not self.stopped.is_set():
                    state_entry = await read(client, "inference-states")
                    state = inference.InferenceRxReader(memoryview(bytes(state_entry.Data)))
                    entries = sorted((entry for entry in state.get_entries()
                                      if entry.get_type() == drivers.QueueDataType.QDT_USB_VIDEO_FRAMES),
                                     key=lambda entry: entry.get_queue())
                    with self.lock:
                        active = {entry.get_queue() for entry in entries}
                        self.frames = {queue: value for queue, value in self.frames.items() if queue in active}
                        self.error = "" if entries else "Station has no USB camera streams. Check its camera connection."
                    for entry in entries:
                        queue = entry.get_queue()
                        envelope_entry = await read(client, queue, bytes(entry.get_ptr()))
                        envelope = usbvideo.RxEnvelopeReader(memoryview(bytes(envelope_entry.Data)))
                        info = envelope.get_camera()
                        name = info.get_product() or info.get_unique_id() or queue
                        if envelope.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES:
                            with self.lock:
                                self.frames[queue] = {"queue": queue, "name": name, "serial": info.get_serial_number(),
                                                      "error": envelope.get_error() or "Camera is not streaming frames.", "changes": 0}
                            continue
                        pack = envelope.get_frames()
                        data, stamps = pack.get_frames_data(), pack.get_stamps()
                        if not data or not stamps or pack.get_format().get_kind() != frame.FrameFormatKind.FF_JPEG:
                            raise RuntimeError(f"{queue} did not supply a timestamped JPEG frame.")
                        stamp = stamps[-1]
                        sample = (stamp.get_app_start_id(), stamp.get_monotonic_stamp_ns(), stamp.get_index())
                        jpeg = bytes(data[-1])
                        if not jpeg:
                            continue
                        with self.lock:
                            old = self.frames.get(queue, {})
                            updated = sample != old.get("sample")
                            self.frames[queue] = {"queue": queue, "name": name, "serial": info.get_serial_number(),
                                                  "unique_id": info.get_unique_id(), "jpeg": jpeg,
                                                  "frame_id": bytes(envelope_entry.ID.ID).hex(), "sample": sample,
                                                  "changes": old.get("changes", 0) + int(updated),
                                                  "updated": time.monotonic() if updated else old.get("updated"),
                                                  "error": None}
                        self.changed.set()
                    await asyncio.sleep(.2)
            except Exception as exc:
                with self.lock:
                    self.error = f"Station camera read failed at {self.server}: {type(exc).__name__}: {exc}"
                self.changed.set()
            finally:
                client.close_connection()
                tasks = set(asyncio.all_tasks()) - previous_tasks
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(.5)

    def snapshot(self):
        with self.lock:
            cameras = []
            for queue, value in sorted(self.frames.items()):
                age = time.monotonic() - value.get("updated", -1e9)
                cameras.append({"queue": queue, "name": value["name"], "serial": value.get("serial", ""),
                                "unique_id": value.get("unique_id"), "frame_id": value.get("frame_id"),
                                "source": f"{self.server}/{queue}", "error": value.get("error"),
                                "age_ms": round(max(0, age) * 1000),
                                "ready": not self.error and value.get("changes", 0) >= 2 and age < FRESH_SECONDS})
            return {"cameras": cameras, "error": self.error}

    def capture_camera(self, queue=None):
        deadline = time.monotonic() + 4
        while True:
            self.changed.clear()
            with self.lock:
                state = self.snapshot()
                cameras = state["cameras"]
                chosen = next((camera for camera in cameras if camera["queue"] == queue), None) if queue else (
                    cameras[min(CAMERAS["workspace"], len(cameras)-1)] if cameras else None)
                if chosen and chosen["ready"]:
                    value = self.frames[chosen["queue"]]
                    return value["jpeg"], {"frame_id": value["frame_id"], "camera_source": chosen["source"],
                                            "camera_queue": chosen["queue"], "camera_unique_id": value["unique_id"]}
                reason = state["error"] or (chosen.get("error") if chosen else None) or (
                    "Waiting for two advancing raw camera frames." if chosen else "Selected camera is not connected.")
            if time.monotonic() >= deadline:
                raise RuntimeError(reason)
            self.changed.wait(timeout=.2)


