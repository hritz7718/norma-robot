"""Assemble SmolVLA inputs from one synchronized, read-only Station snapshot."""

import sys
import time
from io import BytesIO

from PIL import Image

from target.gen_python.protobuf.drivers.inferences import normvla
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.usbvideo import frame, usbvideo
from target.gen_python.protobuf.station import inference

from .config import FOLLOWER, FRESH_SECONDS, MOTOR_IDS, POLICY_CAMERAS


_POLICY_SIZE = (224, 224)
_MAX_SYNC_NS = 100_000_000  # Station's normvla maximum sensor skew.
_STATUS = 0x41
_TORQUE_ENABLE = 0x28
_PRESENT_POSITION = 0x38
_GOAL_POSITION = 0x2A
_PRESENT_SPEED = 0x3A
_PRESENT_CURRENT = 0x45


def station_monotonic_ns():
    # Match the Station Rust clock on the same host.
    clock = time.CLOCK_BOOTTIME if sys.platform == "linux" else time.CLOCK_MONOTONIC_RAW
    return time.clock_gettime_ns(clock)


def _check_stamp(sample, state, label):
    stamp = sample.get_monotonic_stamp_ns()
    delay = state.get_monotonic_stamp_ns() - stamp
    if sample.get_app_start_id() != state.get_app_start_id():
        raise RuntimeError(f"{label} is from a different Station session.")
    if stamp <= 0 or not 0 <= delay <= _MAX_SYNC_NS:
        raise RuntimeError(f"{label} is {delay / 1e6:.1f} ms from the Station snapshot (limit 100 ms).")
    return stamp


def _image(entry, expected_id, state, role):
    envelope = usbvideo.RxEnvelopeReader(memoryview(entry.Data))
    camera = envelope.get_camera()
    label = f"{role.capitalize()} camera"
    if camera is None or camera.get_unique_id() != expected_id:
        raise RuntimeError(f"{label} identity changed; select the camera again.")
    pack = envelope.get_frames()
    if envelope.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES or pack is None:
        raise RuntimeError(f"{label} is not streaming frames.")
    format_info = pack.get_format()
    stamps, images = pack.get_stamps(), pack.get_frames_data()
    if not images or len(images) != len(stamps) or (
            format_info is None or format_info.get_kind() != frame.FrameFormatKind.FF_JPEG):
        raise RuntimeError(f"{label} did not supply a timestamped JPEG.")
    stamp = _check_stamp(stamps[-1], state, label)
    jpeg = bytes(images[-1])
    try:
        with Image.open(BytesIO(jpeg)) as image:
            size = (format_info.get_width(), format_info.get_height())
            if image.format != "JPEG" or image.size != size:
                raise ValueError("JPEG dimensions or format disagree with Station metadata.")
            image.load()
            if image.size != _POLICY_SIZE:
                output = BytesIO()
                image.convert("RGB").resize(_POLICY_SIZE, Image.Resampling.BILINEAR).save(
                    output, format="JPEG", quality=90)
                jpeg = output.getvalue()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{label} supplied an invalid JPEG: {exc}") from exc
    return normvla.Image(jpeg=jpeg, monotonic_stamp_ns=stamp)


def _register(state, address):
    return int.from_bytes(state[address:address + 2], "little") if len(state) >= address + 2 else 0


def _position(state, address):
    encoded = _register(state, address)
    magnitude = encoded & 4095
    return (4096 - magnitude) & 4095 if encoded & 0x8000 else magnitude


def _joints(entry, state):
    telemetry = st3215.InferenceStateReader(memoryview(entry.Data))
    buses = [bus for bus in telemetry.get_buses()
             if bus.get_bus() is not None and bus.get_bus().get_serial_number() == FOLLOWER]
    if len(buses) != 1:
        raise RuntimeError(f"Expected one Follower bus {FOLLOWER}, found {len(buses)}.")
    bus = buses[0]
    _check_stamp(bus, state, "Follower bus")
    motors = sorted(bus.get_motors(), key=lambda motor: motor.get_id())
    if tuple(motor.get_id() for motor in motors) != MOTOR_IDS:
        raise RuntimeError(f"Follower {FOLLOWER} must report motor IDs 1–8.")
    joints = []
    for motor in motors:
        label = f"Follower J{motor.get_id()}"
        stamp = _check_stamp(motor, state, label)
        registers = motor.get_state()
        error = motor.get_error()
        if len(registers) <= _STATUS:
            raise RuntimeError(f"{label} feedback is incomplete.")
        if registers[_STATUS] or (error and (error.get_kind() or any(error.get_servo()))):
            raise RuntimeError(f"{label} reports a motor fault.")
        lo, hi = motor.get_range_min(), motor.get_range_max()
        present = _position(registers, _PRESENT_POSITION)
        if not 0 <= lo < hi <= 4095 or not lo <= present <= hi:
            raise RuntimeError(f"{label} position {present} is incompatible with calibrated limits [{lo}, {hi}].")
        # Torque-off feedback is still a measurement. Report the present
        # position as the goal, just as Station does for an unpowered joint.
        goal = _position(registers, _GOAL_POSITION) if registers[_TORQUE_ENABLE] else present
        joints.append(normvla.Joint(range_min=lo, range_max=hi, position=present,
                                    position_norm=(present - lo) / (hi - lo), goal=goal,
                                    goal_norm=(min(hi, max(lo, goal)) - lo) / (hi - lo),
                                    current_ma=_register(registers, _PRESENT_CURRENT),
                                    velocity=_register(registers, _PRESENT_SPEED),
                                    monotonic_stamp_ns=stamp))
    return joints


def build_observation(state_entry, camera_entries, motor_entry, workspace_id, wrist_id):
    """Encode the model frame from exact queue pointers in the caller's metadata snapshot."""
    if not workspace_id or not wrist_id or workspace_id == wrist_id:
        raise RuntimeError("SmolVLA needs distinct workspace and wrist cameras.")
    state = inference.InferenceRxReader(memoryview(state_entry.Data))
    stamp = state.get_monotonic_stamp_ns()
    if stamp <= 0 or not state.get_app_start_id():
        raise RuntimeError("Station snapshot has no valid session timestamp.")

    def check_age():
        age = station_monotonic_ns() - stamp
        if not 0 <= age <= FRESH_SECONDS * 1e9 - _MAX_SYNC_NS:
            raise RuntimeError(f"Station snapshot is not fresh ({age / 1e6:.1f} ms old).")

    check_age()
    joints = _joints(motor_entry, state)
    camera_ids = {"workspace": workspace_id, "wrist": wrist_id}
    images = [_image(camera_entries[role], camera_ids[role], state, role) for role in POLICY_CAMERAS]
    check_age()
    return normvla.Frame(global_frame_id=bytes(state_entry.ID.ID), monotonic_stamp_ns=stamp,
                         joints=joints, images=images).encode()
