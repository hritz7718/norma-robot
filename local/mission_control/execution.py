"""One cancellable execution loop and one retained SmolVLA model in this backend."""
import asyncio
import copy
import json
import logging
import math
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import (CHECKPOINT, FOLLOWER, LEADER, MOTOR_IDS, ROOT, STATION,
                     DEFAULT_RUN_SECONDS, DEFAULT_MAX_DELTA_TICKS,
                     MAX_RUN_SECONDS, MAX_DELTA_TICKS_LIMIT, PREDICTION_BOUND_TOLERANCE,
                     REQUESTED_SAFE_HEIGHT_MM, APPROACH_MAX_STEP_TICKS, HANDOFF_MAX_STEP_TICKS, HANDOFF_SECONDS)
from .preview import policy_instruction
from .tasks import TaskError, build_task
from .interpreter import blue_fallback_intent
from .vision import Vision, VisionError, cancelled
from .positioning import load_mapping, AlignmentError
from .hands import HandMonitor
from .config import HAND_MAX_AGE_SECONDS, TAUGHT_POINTS


class Execution:
    def __init__(self, station, log, lock, *, interpreter, duration=DEFAULT_RUN_SECONDS,
                 max_delta_ticks=DEFAULT_MAX_DELTA_TICKS, mock=False, safe_height=REQUESTED_SAFE_HEIGHT_MM):
        self.station, self.log, self.lock = station, log, lock
        self.interpreter = interpreter
        self.vision = Vision()
        self.default_safe_height = self.safe_height = safe_height
        self.mapping = None
        self.movement_ready = False
        self.default_duration, self.default_max_delta_ticks = duration, max_delta_ticks
        self.duration, self.max_delta_ticks, self.mock = duration, max_delta_ticks, mock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mission-policy")
        self.model = None
        self.rp = None
        self.cancel = threading.Event()
        self.loop = self.task = self.client = None
        self.deadline = self.run_started = self.run_finished = None
        self.data = {"state": "STOPPED", "stage": "Click Execute to run SmolVLA",
                     "active": False, "commands_sent": 0, "model_state": "not_loaded",
                     "error": None, "prompt": None, "started_at": None,
                     "duration_seconds": duration, "max_delta_ticks": max_delta_ticks,
                     "teleoperation": "Not checked", "pause_available": False,
                     "stop_available": True, "owner": None, "progress": 0,
                     "stop_reason": None, "device": None, "load_ms": None,
                     "inference_ms": None, "cycle_ms": None, "last_write_ms": None,
                     "write_status": "No write", "last_action": None,
                     "last_prediction": None, "clamped_predictions": 0,
                     "prediction_tolerance": PREDICTION_BOUND_TOLERANCE}
        self.data.update(original_prompt=None, interpretation=None, interpretation_ms=None)
        self.data.update(safe_height_mm=safe_height, detection=None, approach=None, image_input=None,
                         verification=None, interpretation_source=None, vla_step_limit=None)
        self.hands = HandMonitor(self._hand_trip, self.vision)
        self.station.on_observation = self._observe_workspace
        self.hands.start()

    def _observe_workspace(self, jpeg, frame_id, camera=None):
        self.vision.observe(jpeg, frame_id, camera)
        self.hands.submit(jpeg, frame_id, camera)

    def _hand_trip(self, reason):
        with self.lock:
            active = self.data["active"]
        if active:
            self.stop(reason, error=True)
        else:
            self.log("warning", reason)

    def _hand_problem(self):
        camera = self.station.snapshot()
        age = camera.get("camera_age_ms", {}).get("workspace")
        if not camera.get("camera_live", {}).get("workspace") or age is None or age > HAND_MAX_AGE_SECONDS*1000:
            return "Workspace camera is stale or unavailable. VLA STOP."
        return self.hands.problem()

    def snapshot(self):
        with self.lock:
            data = copy.deepcopy(self.data)
            data["vision"] = self.vision.snapshot()
            data["hand_monitor"] = self.hands.snapshot()
            end = self.run_finished if self.run_finished is not None else time.monotonic()
            elapsed = max(0, end - self.run_started) if self.run_started is not None else 0
            data.update(elapsed_seconds=round(elapsed, 1),
                        timer_started=self.run_started is not None,
                        remaining_seconds=round(max(0, self.duration - elapsed), 1),
                        actions_per_second=round(data["commands_sent"] / elapsed, 2) if elapsed else 0)
            return data

    def suggested_safe_height(self):
        try:
            taught = json.loads(TAUGHT_POINTS.read_text())
            height = float(taught["safe_height_mm"])
            if taught.get("motion_enabled") is True and taught.get("physically_validated") is True and math.isfinite(height) and 1 <= height <= 500:
                return height
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return self.default_safe_height

    def start(self, prompt, *, duration=None, max_delta_ticks=None, safe_height=None):
        duration = self.default_duration if duration is None else duration
        max_delta_ticks = self.default_max_delta_ticks if max_delta_ticks is None else max_delta_ticks
        safe_height = self.suggested_safe_height() if safe_height is None else safe_height
        if type(safe_height) not in (int, float) or not math.isfinite(safe_height) or not 1 <= safe_height <= 500:
            raise ValueError("Requested safe height must be 1–500mm and must fit the measured robot alignment.")
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= MAX_RUN_SECONDS:
            raise ValueError(f"Duration must be greater than zero and at most {MAX_RUN_SECONDS:g} seconds.")
        if type(max_delta_ticks) is not int or not 1 <= max_delta_ticks <= MAX_DELTA_TICKS_LIMIT:
            raise ValueError(f"Max step must be an integer from 1 to {MAX_DELTA_TICKS_LIMIT} ticks.")
        with self.lock:
            if self.mock:
                raise RuntimeError("Execution is disabled in simulation mode.")
            if self.data["active"]:
                raise RuntimeError("Execution is already active; click STOP first.")
            feedback = self.station.snapshot()
            if not feedback.get("camera_live", {}).get("workspace"):
                raise RuntimeError(feedback.get("camera_error") or "A fresh workspace camera image is required for calibration.")
            age = feedback.get("camera_age_ms", {}).get("workspace")
            if age is None or age > HAND_MAX_AGE_SECONDS*1000:
                raise RuntimeError("Workspace camera is too old for hand monitoring.")
            # Only an explicit Execute click clears a recovered monitor STOP.
            self.hands.rearm()
            self.duration, self.max_delta_ticks = float(duration), max_delta_ticks
            self.safe_height, self.mapping, self.movement_ready = float(safe_height), None, False
            self.cancel = threading.Event()
            self.data.update(state="RUNNING", active=True, commands_sent=0, error=None,
                             stage="Calibrating", original_prompt=prompt,
                             prompt=None, interpretation=None, interpretation_ms=None,
                             started_at=time.time(), owner="Calibration",
                             teleoperation="Checking mirroring", stop_reason=None,
                             duration_seconds=self.duration, max_delta_ticks=max_delta_ticks,
                             inference_ms=None, cycle_ms=None, last_write_ms=None,
                             write_status="No write", last_action=None,
                             last_prediction=None, clamped_predictions=0, safe_height_mm=self.safe_height,
                             detection=None, approach=None, image_input=None, verification=None,
                             interpretation_source=None, vla_step_limit=None)
            # Loading cannot move the robot and must not consume the run's
            # movement budget or require a second Execute click.
            self.deadline = None
            self.run_started = self.run_finished = None
            threading.Thread(target=self._thread, args=(prompt, self.cancel), daemon=True,
                             name="mission-execution").start()
        self.log("info", f"Execute: {prompt!r}; calibrate → detect → verified {self.safe_height:g}mm approach → supervised SmolVLA. Run limit {self.duration:g}s.")

    def stop(self, reason="Operator STOP", *, error=False):
        # No inference lock, model wait, network operation, or torque write here.
        with self.lock:
            self.cancel.set()
            active = self.data["active"]
            self.data.update(state="STOPPED", stage=reason, stop_reason=reason, owner=None)
            if active and self.run_finished is None:
                self.run_finished = time.monotonic()
            if error:
                self.data["error"] = reason
            loop, task = self.loop, self.task
        if loop and task:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The loop has already exited.
        if active:
            self.log("error" if error else "info", f"{reason}. No further actions will be sent; torque is unchanged.")

    def close(self):
        self.stop("Backend shutting down")
        self.hands.close()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _load(self):
        if self.model is None:
            started = time.perf_counter()
            with self.lock:
                self.data["model_state"] = "loading"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            sys.path[:0] = [str(ROOT), str(ROOT / "software/station/shared"),
                            str(ROOT / "software/ai/smolvla_py")]
            try:
                from scripts import run_policy
                self.rp = run_policy
                self.model = run_policy.load_policy(CHECKPOINT)
            except Exception:
                with self.lock:
                    self.data["model_state"] = "unavailable"
                raise
            with self.lock:
                self.data["model_state"] = "ready"
                self.data["device"] = str(self.model[2])
                self.data["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.log("info", f"SmolVLA checkpoint loaded on {self.model[2]}; retained for subsequent actions and runs.")

    def _predict(self, raw, prompt):
        frame = self.rp.normvla.FrameReader(memoryview(raw))
        policy, stats, device = self.model
        goals, ranges = self.rp.predict_goals(frame, policy, stats, prompt, device)
        if goals.shape != (8,):
            raise RuntimeError(f"Rejected prediction: expected 8 joints, got shape {goals.shape}.")
        tolerance = PREDICTION_BOUND_TOLERANCE
        invalid = [f"J{i + 1}={float(g):.5g}" for i, g in enumerate(goals)
                   if not math.isfinite(float(g)) or not -tolerance <= g <= 1 + tolerance]
        if invalid:
            raise RuntimeError(f"Rejected invalid prediction (allowed raw range [{-tolerance:g}, {1 + tolerance:g}]): "
                               + ", ".join(invalid))
        original = goals.tolist()
        bounded = [max(0.0, min(1.0, g)) for g in original]
        clamped = [mid for mid, raw, goal in zip(MOTOR_IDS, original, bounded) if raw != goal]
        return bounded, ranges, original, clamped

    @staticmethod
    def _feedback_problem(state):
        """Name the actual blocker instead of a combined fault/stale message."""
        reasons = [state["error"], state["motor_error"]]
        if not state["live"]:
            age = state["observation_age_ms"]
            reasons.append(f"Observation stale ({age} ms old)" if age is not None else "No camera observation received")
        if state["camera_count"] != 2:
            reasons.append(f"Camera inputs: {state['camera_count']}/2")
        if not state["ranges_valid"]:
            reasons.append("Invalid or missing calibrated joint positions")
        reported = {m["id"] for m in state["motors"]}
        missing = [f"J{mid}" for mid in MOTOR_IDS if mid not in reported]
        if missing:
            reasons.append("Missing feedback: " + ", ".join(missing))
        for motor in state["motors"]:
            if motor["error_kind"] or any(motor["servo_errors"]) or motor["status"] == "fault":
                reasons.append(f"J{motor['id']} fault: {motor['error'] or 'motor error'} "
                               f"(kind={motor['error_kind']}, servo={motor['servo_errors']})")
            if motor["status"] == "stale":
                reasons.append(f"J{motor['id']} feedback stale ({motor['age_ms']} ms old)")
        return "; ".join(dict.fromkeys(reason for reason in reasons if reason)) or "Follower feedback is unavailable"

    def _thread(self, prompt, cancel):
        asyncio.run(self._run(prompt, cancel))

    async def _run(self, prompt, cancel):
        with self.lock:
            self.loop = asyncio.get_running_loop()
            self.task = asyncio.create_task(self._execute(prompt, cancel))
            task = self.task
        watchdog = asyncio.create_task(self._watch(task, cancel))
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not cancel.is_set():
                self.stop(f"Execution stopped: {type(exc).__name__}: {str(exc)[:350]}", error=True)
        finally:
            cancel.set()
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
            if self.client:
                # Each run gets a new connection. Never retry or replay a motor write.
                self.client.close_connection()
            with self.lock:
                self.loop = self.task = self.client = None
                if self.run_finished is None:
                    self.run_finished = time.monotonic()
                self.data.update(active=False, owner=None)
                if self.data["state"] != "FINISHED":
                    self.data["state"] = "STOPPED"

    async def _watch(self, task, cancel):
        while not task.done():
            if cancel.is_set():
                task.cancel()
                return
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.stop(f"Run duration limit reached ({self.duration:g}s)")
                return
            problem = self._hand_problem()
            if problem:
                self.stop(problem, error=True)
                return
            if self.mapping is not None:
                try:
                    self.mapping.assert_current()
                except AlignmentError as exc:
                    self.stop(str(exc), error=True)
                    return
            # Interpretation, loading, and waiting for input cannot issue motor commands.
            # Start the hardware watchdog when preparation is about to begin.
            if self.run_started is None:
                await asyncio.sleep(.05)
                continue
            state = self.station.snapshot()
            if not state["preview_ready"]:
                self.stop(self._feedback_problem(state), error=True)
                return
            problem = self.vision.current_problem()
            if problem:
                self.stop(problem, error=True)
                return
            if self.client and (not self.client.connected or not self.client.setup_done):
                self.stop("Station connection lost", error=True)
                return
            await asyncio.sleep(.05)

    def _check(self, cancel):
        if self.deadline is not None and time.monotonic() >= self.deadline and not cancel.is_set():
            self.stop(f"Run duration limit reached ({self.duration:g}s)")
        if cancel.is_set():
            raise asyncio.CancelledError()
        if self.mapping is not None:
            self.mapping.assert_current()
        problem = self._hand_problem()
        if problem:
            raise RuntimeError(problem)
        if FOLLOWER == LEADER:
            raise RuntimeError("Follower must be distinct from Leader.")

    async def _send(self, command, cancel):
        # This check follows inference and immediately precedes enqueueing a write.
        async def send():
            self._check(cancel)
            if not self.movement_ready:
                raise RuntimeError("Calibration, detection, and verified safe approach planning must finish before motor commands.")
            problem = self.vision.current_problem()
            if problem:
                raise VisionError(problem)
            feedback = self.station.snapshot()
            if not feedback["preview_ready"]:
                raise RuntimeError(self._feedback_problem(feedback))
            if self.mapping and [(j["range_min"], j["range_max"]) for j in feedback["joints"]] != self.mapping.ranges:
                raise AlignmentError("Servo calibration changed after robot alignment was checked.")
            await self.rp.send_commands(self.client, [command])
        started = time.perf_counter()
        with self.lock:
            self.data["write_status"] = "Waiting for Station"
            self.data["last_write_ms"] = None
        try:
            await asyncio.wait_for(send(), timeout=2)
        except asyncio.CancelledError:
            with self.lock:
                self.data["write_status"] = "Cancelled"
            raise
        except asyncio.TimeoutError:
            with self.lock:
                self.data["write_status"] = "Timed out"
            raise RuntimeError("Station command acknowledgement timed out after 2 seconds.") from None
        except Exception:
            with self.lock:
                self.data["write_status"] = "Failed"
            raise
        else:
            with self.lock:
                self.data["write_status"] = "Acknowledged"
        finally:
            with self.lock:
                self.data["last_write_ms"] = round((time.perf_counter() - started) * 1000, 1)

    async def _mirroring(self):
        from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
        qr = self.client.read_from_tail("inference/mirroring", offset=b"\x00", limit=1, step=1, buf_size=1)
        try:
            entry = await asyncio.wait_for(qr.data.get(), timeout=2)
        except asyncio.TimeoutError:
            raise RuntimeError("Station mirroring status timed out after 2 seconds.") from None
        if entry is None:
            raise RuntimeError(f"Mirroring status unavailable: {qr.err}")
        envelope = mirroring.RxEnvelopeReader(memoryview(bytes(entry.Data)))
        if envelope.get_monotonic_stamp_ns() <= 0:
            raise RuntimeError("Invalid mirroring status.")
        return envelope

    @staticmethod
    def _follower_mirrored(envelope):
        active = False
        for mapping in envelope.get_state().get_mirroring():
            if mapping.get_source().get_id().get_unique_id() == FOLLOWER and mapping.get_targets():
                raise RuntimeError("Follower is configured as a mirroring source; refusing handover.")
            active |= any(target.get_id().get_unique_id() == FOLLOWER for target in mapping.get_targets())
        return active

    async def _prepare(self, cancel):
        from target.gen_python.protobuf.drivers.st3215 import st3215
        from target.gen_python.protobuf.station import commands, drivers

        before = await self._mirroring()
        if self._follower_mirrored(before):
            raise RuntimeError("MIRRORING HANDOVER REQUIRED: Station's mirror-stop disables Follower torque. Safely disable mirroring with the arm supported before retrying; Mission Control will not drop torque automatically.")
        with self.lock:
            self.data["teleoperation"] = "Follower mirroring disabled"
        self.log("info", "Follower mirroring is disabled; Leader is never a motor-command target.")

        _, snapshot = self.station.capture()
        if any(m["torque_enabled"] is not True for m in snapshot["motors"]):
            # Replace old goals with the current pose before enabling torque.
            self._raw_goals([j["pos_norm"] for j in snapshot["joints"]], snapshot)
            hold = [j["pos"] for j in snapshot["joints"]]
            await self._send(self.rp.build_sync_write_command(FOLLOWER, list(MOTOR_IDS), hold), cancel)
            torque = st3215.Command(target_bus_serial=FOLLOWER,
                sync_write=st3215.ST3215SyncWriteCommand(address=0x28, motors=[
                    st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=b"\x01") for mid in MOTOR_IDS]))
            await self._send(commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND,
                                                    body=torque.encode()), cancel)
            end = time.monotonic() + 3
            while not all(m["torque_enabled"] is True for m in self.station.snapshot()["motors"]):
                if time.monotonic() >= end:
                    raise RuntimeError("Follower torque enable was not confirmed.")
                await asyncio.sleep(.05)

    def _raw_goals(self, goals, snapshot, limit=None):
        limit = self.max_delta_ticks if limit is None else limit
        raw = []
        for goal, joint in zip(goals, snapshot["joints"]):
            lo, hi, pos = joint["range_min"], joint["range_max"], joint["pos"]
            if not 0 <= lo < hi <= 4095 or not lo <= pos <= hi:
                raise RuntimeError("Invalid calibrated range or current joint position.")
            target = round(lo + goal * (hi - lo))
            raw.append(max(lo, min(hi, pos + max(-limit, min(limit, target - pos)))))
        return raw

    async def _fresh_workspace(self, cancel, after=None):
        end = time.monotonic() + 3
        while True:
            self._check(cancel)
            try:
                jpeg, snapshot = self.station.capture_workspace()
                if snapshot["frame_id"] != after:
                    return jpeg, snapshot
            except RuntimeError:
                pass
            if time.monotonic() >= end:
                raise VisionError("Workspace camera did not deliver a fresh image within 3 seconds.")
            await asyncio.sleep(.05)

    async def _calibrate(self, cancel):
        end = time.monotonic() + 10
        last = None
        while True:
            jpeg, snapshot = await self._fresh_workspace(cancel, last)
            last = snapshot["frame_id"]
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self.pool, self.vision.calibrate, jpeg, last, cancel, snapshot)
                self._check(cancel)
                problem = self.vision.current_problem()
                if problem:
                    raise VisionError(problem)
                if result.get("method") == "manual_four_point":
                    self.log("info", f"CALIBRATION SUCCESSFUL: loaded saved four-point homography for camera {result['camera_unique_id']}; point-fit error {result['max_error_px']:.2f}px. Keep the camera and tabletop reference fixed.")
                else:
                    self.log("info", f"CALIBRATION SUCCESSFUL: {result['dictionary']}; max reprojection {result['max_error_px']:.2f}px, independent check {result['independent_error_px']:.2f}px. {'Revalidated saved homography.' if result['reused'] else 'Saved new homography.'}")
                return snapshot
            except VisionError as exc:
                if time.monotonic() >= end or any(word in str(exc) for word in ("dictionary", "PRINT SCALE", "measurements", "Saved manual calibration")):
                    self.log("error", f"CALIBRATION FAILED: {exc}")
                    raise
                with self.vision.lock:
                    self.vision.data["calibration"] = "CALIBRATING"
                await asyncio.sleep(.1)

    def _interpret_scene(self, prompt, jpeg, frame_id, cancel):
        cancelled(cancel)
        metadata = {"frame_id": frame_id, "sha256": hashlib.sha256(jpeg).hexdigest(),
                    "attached": False, "interpreted": False}
        try:
            metadata["attached"] = bool(self.interpreter.image_supported and not self.interpreter.error)
            result = self.interpreter.interpret_image(prompt, jpeg, cancel)
            metadata["interpreted"] = True
            return result["intent"], result["visual"], metadata
        except Exception as exc:
            cancelled(cancel)
            intent = blue_fallback_intent(prompt)
            metadata["reason"] = str(exc)[:300]
            self.log("warning", f"OpenCV blue-object fallback: Codex image interpretation unavailable ({str(exc)[:200]}). This detection is not attributed to Codex.")
            return intent, None, metadata

    async def _approach(self, plan, cancel):
        mapping = self.mapping
        taught = getattr(mapping, "taught", False)
        with self.lock:
            self.data.update(stage="Moving above target" if taught else "Moving to safe height",
                             owner="Taught interpolation" if taught else "Verified positioning")
        last_id = None
        for waypoint_index, waypoint in enumerate(plan["waypoints"]):
            moved_at, previous = time.monotonic(), None
            while True:
                self._check(cancel)
                _, feedback = self.station.capture()
                if feedback["frame_id"] == last_id:
                    await asyncio.sleep(.03)
                    continue
                last_id = feedback["frame_id"]
                current = [j["pos"] for j in feedback["joints"]]
                if [(j["range_min"], j["range_max"]) for j in feedback["joints"]] != mapping.ranges:
                    raise AlignmentError("Servo calibration changed during approach.")
                if any(motor["torque_enabled"] is not True for motor in feedback["motors"]):
                    raise RuntimeError("Follower torque is unavailable during approach.")
                actual_tcp = None if taught else mapping.checked_point(current)
                if taught:
                    mapping.check_ticks(current)
                distance = max(abs(goal - pos) for goal, pos in zip(waypoint, current))
                final_reached = distance <= 8 if taught else mapping.np.linalg.norm(actual_tcp - mapping.np.array(plan["target_table_mm"])) <= 3
                if distance <= 8 and (waypoint_index < len(plan["waypoints"]) - 1 or final_reached):
                    break
                if previous is None or max(abs(a - b) for a, b in zip(current, previous)) >= 2:
                    moved_at, previous = time.monotonic(), current
                elif time.monotonic() - moved_at > 3:
                    raise AlignmentError("Approach stalled: actual motor feedback is not reaching the target.")
                limit = min(APPROACH_MAX_STEP_TICKS, self.max_delta_ticks)
                fraction = min(1.0, limit/max(1, distance))
                sent = [round(pos+(goal-pos)*fraction) for pos, goal in zip(current, waypoint)]
                mapping.check_segment(current, sent)
                if self._follower_mirrored(await self._mirroring()):
                    raise RuntimeError("Follower mirroring was re-enabled during approach.")
                await self._send(self.rp.build_sync_write_command(FOLLOWER, list(MOTOR_IDS), sent), cancel)
                with self.lock:
                    self.data["commands_sent"] += 1
                    count = self.data["commands_sent"]
                    self.data["last_action"] = {
                        "number": count, "source": "taught_interpolation" if taught else "verified_ik", "sent_at": time.time(),
                        "sent_ticks": sent, "requested_ticks": waypoint,
                        "delta_ticks": [goal - pos for goal, pos in zip(sent, current)],
                        "limited_joints": [mid for mid, a, b in zip(MOTOR_IDS, sent, waypoint) if a != b],
                        "clamped_joints": [],
                    }
                self.log("info", f"Approach {count}: bounded Follower goals {sent}")
        _, feedback = self.station.capture()
        if taught:
            actual = mapping.check_ticks([joint["pos"] for joint in feedback["joints"]])
            if mapping.np.max(abs(actual-mapping.np.asarray(plan["target_ticks"]))) > 8:
                raise AlignmentError("Taught approach joint targets were not reached.")
            mapping.handoff_anchor = actual.copy()
            self.log("info", f"Taught approach reached by motor feedback; estimated table position {plan['target_table_mm']}mm. Height relies on the operator-validated teaching. Handing live observations to supervised SmolVLA.")
            return
        actual = mapping.checked_point([j["pos"] for j in feedback["joints"]])
        error = mapping.np.linalg.norm(actual - mapping.np.array(plan["target_table_mm"]))
        if error > 3:
            raise AlignmentError(f"Safe approach was not reached: TCP error {error:.1f}mm.")
        self.log("info", f"Safe approach reached: table TCP {actual.round(1).tolist()}mm. Handing current live observations to SmolVLA; this handoff state was not separately trained.")

    async def _execute(self, prompt, cancel):
        self._check(cancel)
        calibrated = await self._calibrate(cancel)
        jpeg, image_snapshot = await self._fresh_workspace(cancel, calibrated["frame_id"])
        with self.lock:
            self.data.update(stage="Detecting object", owner="Vision")
        started = time.perf_counter()
        self.log("info", f"Captured workspace frame {image_snapshot['frame_id']} for Codex image interpretation and OpenCV refinement.")
        try:
            intent, visual, image_input = await asyncio.get_running_loop().run_in_executor(
                self.pool, self._interpret_scene, prompt, jpeg, image_snapshot["frame_id"], cancel)
        finally:
            with self.lock:
                self.data["interpretation_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._check(cancel)
        with self.lock:
            self.data.update(image_input=image_input,
                             interpretation_source="Codex vision + OpenCV" if image_input["interpreted"] else "OpenCV")
        task = build_task(intent, prompt)
        if task["task_type"] != "pick_up" or task["target"]["color"] != "blue":
            with self.lock:
                self.data["interpretation"] = task
            raise TaskError("This automatic-picking demonstration supports a single blue-block pickup only.")
        # Execute is the deliberate authorization for this accepted task.
        task.update(requires_operator_approval=False, execution_enabled=True)
        instruction = policy_instruction(task)
        detection = await asyncio.get_running_loop().run_in_executor(
            self.pool, self.vision.detect, jpeg, image_snapshot["frame_id"], visual)
        self._check(cancel)
        with self.lock:
            self.data.update(interpretation=task, prompt=instruction, owner="SmolVLA",
                             detection=detection, image_input=image_input,
                             interpretation_source="Codex vision + OpenCV" if image_input["interpreted"] else "OpenCV",
                             stage="Checking robot alignment")
        self.log("info", f"Detected {detection['object']} via {detection['source']}: pixel {detection['pixel']}, tabletop {detection['table_mm']}mm; score {detection['confidence']:.2f}.")
        # Mapping validation precedes model load, torque changes, or any writes.
        initial = self.station.snapshot()
        self.mapping = await asyncio.get_running_loop().run_in_executor(
            self.pool, load_mapping, self.vision, initial, self.safe_height)
        with self.lock:
            self.data["stage"] = "Loading model" if self.model is None else "Planning safe approach"
        await asyncio.get_running_loop().run_in_executor(self.pool, self._load)
        self._check(cancel)
        with self.lock:
            if not cancel.is_set():
                self.data["stage"] = "Waiting for live feedback"
        # Model initialization can briefly starve observation polling. No motor
        # writes are permitted until the reader has caught up and is healthy.
        fresh_deadline = time.monotonic() + 3
        while True:
            self._check(cancel)
            feedback = self.station.snapshot()
            if feedback["preview_ready"]:
                break
            motor_fault = any(m["error_kind"] or any(m["servo_errors"]) or m["status"] == "fault"
                              for m in feedback["motors"])
            if motor_fault or time.monotonic() >= fresh_deadline:
                raise RuntimeError(self._feedback_problem(feedback))
            await asyncio.sleep(.05)
        # Refuse stale targets after Codex/model latency; do not chase a moved block.
        latest_jpeg, latest_image = await self._fresh_workspace(cancel)
        current_detection = await asyncio.get_running_loop().run_in_executor(
            self.pool, self.vision.detect, latest_jpeg, latest_image["frame_id"])
        moved = math.hypot(current_detection["table_mm"]["x"] - detection["table_mm"]["x"],
                           current_detection["table_mm"]["y"] - detection["table_mm"]["y"])
        if moved > 5:
            raise VisionError(f"Object moved {moved:.1f}mm while interpreting/loading. Retry with the object stationary.")
        current_detection["source"] = detection["source"]
        _, feedback = self.station.capture()
        current_raw = [j["pos"] for j in feedback["joints"]]
        with self.lock:
            self.data["stage"] = "Planning safe approach"
        xy = [current_detection["table_mm"][key] for key in ("x", "y")]
        plan = await asyncio.get_running_loop().run_in_executor(self.pool, self.mapping.plan, xy, current_raw, cancel)
        self._check(cancel)
        with self.lock:
            self.data.update(approach={key: value for key, value in plan.items() if key != "waypoints"}, detection=current_detection)
            self.data["approach"]["waypoint_count"] = len(plan["waypoints"])
            self.movement_ready = True
        with self.lock:
            self.run_started = time.monotonic()
            self.deadline = self.run_started + self.duration
            if not cancel.is_set():
                self.data["stage"] = "Preparing follower"
        from station_py.client import Client
        client = Client(STATION, logging.getLogger("mission-execution"))
        # Wait before exposing it to the connection watchdog.
        try:
            await client.wait_ready(timeout=2)
        except BaseException:
            client.close_connection()
            raise
        self.client = client
        await self._prepare(cancel)
        _, prepared = self.station.capture()
        if max(abs(j["pos"] - expected) for j, expected in zip(prepared["joints"], current_raw)) > 10:
            raise AlignmentError("Robot moved after safe-path planning; retry from a stable, verified safe start pose.")
        await self._approach(plan, cancel)
        with self.lock:
            self.data.update(stage="Running VLA", owner="SmolVLA", vla_step_limit=min(self.max_delta_ticks, HANDOFF_MAX_STEP_TICKS))
        vla_deadline = min(self.deadline, time.monotonic() + HANDOFF_SECONDS)
        last_id = None
        while True:
            if time.monotonic() >= vla_deadline:
                break
            self._check(cancel)
            cycle_started = time.perf_counter()
            raw, snapshot = self.station.capture()
            if snapshot["frame_id"] == last_id:
                await asyncio.sleep(.02)
                continue
            last_id = snapshot["frame_id"]
            with self.lock:
                self.data["stage"] = "Running VLA"
            inference_started = time.perf_counter()
            try:
                goals, ranges, original, clamped = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(self.pool, self._predict, raw, instruction),
                    timeout=max(.001, vla_deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            finally:
                with self.lock:
                    self.data["inference_ms"] = round((time.perf_counter() - inference_started) * 1000, 1)
            self._check(cancel)
            if time.monotonic() >= vla_deadline:
                break
            with self.lock:
                self.data["last_prediction"] = {"raw": original, "bounded": goals, "clamped_joints": clamped}
                if clamped:
                    self.data["clamped_predictions"] += 1
                first_clamp = bool(clamped) and self.data["clamped_predictions"] == 1
            if first_clamp:
                details = ", ".join(f"J{mid}={original[mid - 1]:.5g} → {goals[mid - 1]:g}" for mid in clamped)
                self.log("warning", f"Small prediction overshoot clamped to calibrated bounds: {details}. Continuing with the step limit.")
            _, latest = self.station.capture()
            if ranges != [(j["range_min"], j["range_max"]) for j in latest["joints"]]:
                raise RuntimeError("Joint calibration changed during inference.")
            if any(m["torque_enabled"] is not True for m in latest["motors"]):
                raise RuntimeError("Follower torque is unavailable.")
            if self._follower_mirrored(await self._mirroring()):
                raise RuntimeError("Follower mirroring was re-enabled during execution.")
            raw_goals = self._raw_goals(goals, latest, min(self.max_delta_ticks, HANDOFF_MAX_STEP_TICKS))
            self.mapping.check_segment([j["pos"] for j in latest["joints"]], raw_goals, approach=False)
            requested = [round(j["range_min"] + g * (j["range_max"] - j["range_min"]))
                         for g, j in zip(goals, latest["joints"])]
            await self._send(self.rp.build_sync_write_command(FOLLOWER, list(MOTOR_IDS), raw_goals), cancel)
            with self.lock:
                self.data["commands_sent"] += 1
                count = self.data["commands_sent"]
                self.data["cycle_ms"] = round((time.perf_counter() - cycle_started) * 1000, 1)
                self.data["last_action"] = {
                    "number": count, "sent_at": time.time(), "input_frame_id": last_id,
                    "input_age_ms": snapshot["observation_age_ms"],
                    "predicted_norm": goals, "requested_ticks": requested, "sent_ticks": raw_goals,
                    "raw_predicted_norm": original, "clamped_joints": clamped,
                    "delta_ticks": [goal - j["pos"] for goal, j in zip(raw_goals, latest["joints"])],
                    "limited_joints": [mid for mid, want, sent in zip(MOTOR_IDS, requested, raw_goals) if want != sent],
                }
                if not cancel.is_set():
                    self.data["stage"] = "Running VLA"
            suffix = " · boundary clamped: " + ", ".join(f"J{mid}" for mid in clamped) if clamped else ""
            self.log("info", f"Action {count}: Follower goals {raw_goals}{suffix}")
        self._check(cancel)
        _, final_feedback = self.station.capture()
        tcp = None if getattr(self.mapping, "taught", False) else self.mapping.pose([j["pos"] for j in final_feedback["joints"]])[:3, 3].tolist()
        with self.lock:
            self.run_finished = time.monotonic()
            self.data.update(state="FINISHED", stage="Finished — verify grasp", owner=None,
                             verification={"status": "UNVERIFIED", "tcp_table_mm": tcp,
                                           "reason": "Supervised attempt ended. A lifted gripper does not prove the object was grasped; inspect both live cameras."})
        self.log("info", "FINISHED: supervised SmolVLA attempt ended. Grasp verification remains operator visual review; torque unchanged.")
