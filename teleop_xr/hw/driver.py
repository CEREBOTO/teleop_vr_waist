"""CAN dispatch for TeaArm: right can0, left can1, waist can2.

Does not call SingleArmDriver.start() — that yaml home pose is not the
TeaArm rest used by teleop_xr IK.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from teleop_xr.hw.gravity import GravityModel
from teleop_xr.hw.mapping import (
    GRIPPER_CLOSED,
    GRIPPER_LIMITS,
    LEFT_ARM_JOINTS,
    PITCH_OFFSET_DEFAULT,
    RIGHT_ARM_JOINTS,
    YAW_OFFSET_DEFAULT,
    motor_tau_from_urdf,
    pack_arm,
    rate_limit,
    waist_motor_command,
    waist_urdf_command,
    waist_urdf_from_motor,
)


class HardwareError(RuntimeError):
    """CAN, motors, or rest-pose verification failed."""


def damiao_set_zero(can_interface: str, send_id: int) -> None:
    """Write EEPROM zero for one Damiao motor (disable → set_zero → disable)."""
    import socket
    import struct

    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        sock.bind((can_interface,))

        def _send(cmd: int) -> None:
            data = bytes([0xFF] * 7 + [cmd])
            sock.send(struct.pack("=IB3x8s", send_id, 8, data))

        _send(0xFD)
        time.sleep(0.1)
        _send(0xFE)
        time.sleep(0.1)
        _send(0xFD)
        time.sleep(0.1)
    finally:
        sock.close()


def require_can_interfaces(names: list[str]) -> None:
    problems: list[str] = []
    for name in names:
        root = Path(f"/sys/class/net/{name}")
        if not root.is_dir():
            problems.append(f"{name}: interface does not exist")
            continue
        operstate = (root / "operstate").read_text().strip()
        mtu = int((root / "mtu").read_text())
        if operstate == "down":
            problems.append(f"{name}: DOWN (operstate=down, mtu={mtu})")
        elif mtu < 72:
            problems.append(f"{name}: mtu={mtu} (need CAN FD, mtu>=72)")
    if problems:
        raise HardwareError("CAN preflight failed: " + "; ".join(problems))


def _enable_arm(arm) -> None:
    import openarm_can as oa

    arm.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
    arm.openarm.enable_all()
    arm.set_latest_state(timeout_us=500)
    arm.openarm.refresh_all()
    arm.set_latest_state(timeout_us=500)
    arm.started = True
    arm.last_command = arm.fetch_position(refresh=True)


def _patch_arm_config(config, *, right_can: str, left_can: str) -> None:
    config._config["can_interface"]["right_arm"] = right_can
    config._config["can_interface"]["left_arm"] = left_can
    low, high = GRIPPER_LIMITS
    config._config["joint_limits"]["right_arm"][7] = [low, high]
    config._config["joint_limits"]["left_arm"][7] = [low, high]


class WaistDriver:
    """pitch = DM10010L 0x01/0x11, yaw = DM8009 0x02/0x12, both on can2 FD."""

    def __init__(
        self,
        can_interface: str = "can2",
        *,
        pitch_kp: float = 150.0,
        pitch_kd: float = 4.5,
        yaw_kp: float = 70.0,
        yaw_kd: float = 2.5,
        yaw_offset: float = YAW_OFFSET_DEFAULT,
        pitch_offset: float = PITCH_OFFSET_DEFAULT,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
    ) -> None:
        try:
            import openarm_can as oa
        except ImportError as exc:
            raise SystemExit(
                "openarm_can is required (comes with openarm_driver).\n"
                "  uv pip install -e ../openarm_driver"
            ) from exc

        self._oa = oa
        self.can_interface = can_interface
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self.kps = np.array([pitch_kp, yaw_kp], dtype=float)
        self.kds = np.array([pitch_kd, yaw_kd], dtype=float)
        self.openarm = oa.OpenArm(can_interface, True)
        self.openarm.init_arm_motors(
            [oa.MotorType.DM10010L, oa.MotorType.DM8009],
            [0x01, 0x02],
            [0x11, 0x12],
            [oa.ControlMode.MIT, oa.ControlMode.MIT],
        )
        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        self._enabled = False
        self._last_motor = np.zeros(2, dtype=float)

    def enable(self) -> None:
        self.openarm.enable_all()
        self.openarm.recv_all(1000)
        self.openarm.refresh_all()
        self.openarm.recv_all(1000)
        self._enabled = True
        self._last_motor = self.positions()

    def disable(self) -> None:
        if self._enabled:
            self.openarm.disable_all()
            self.openarm.recv_all(1000)
        self._enabled = False

    def positions(self) -> np.ndarray:
        self.openarm.refresh_all()
        self.openarm.recv_all(300)
        motors = list(self.openarm.get_arm().get_motors())
        return np.array([m.get_position() for m in motors], dtype=float)

    def send_motor(
        self,
        pitch: float,
        yaw: float,
        tau_pitch: float = 0.0,
        tau_yaw: float = 0.0,
    ) -> None:
        self.openarm.get_arm().mit_control_all(
            [
                self._oa.MITParam(
                    self.kps[0], self.kds[0], float(pitch), 0.0, float(tau_pitch)
                ),
                self._oa.MITParam(
                    self.kps[1], self.kds[1], float(yaw), 0.0, float(tau_yaw)
                ),
            ]
        )
        self.openarm.recv_all(300)
        self._last_motor = np.array([pitch, yaw], dtype=float)

    def set_pitch_zero(self) -> float:
        """Set EEPROM zero on pitch only (send 0x01). Robot must sit on the -10 deg stop."""
        before = self.positions()
        print(
            f"[hw] pitch encoder before set_zero: {before[0]:+.4f} rad "
            f"({math.degrees(before[0]):+.2f} deg); yaw left untouched "
            f"({before[1]:+.4f} rad)"
        )
        damiao_set_zero(self.can_interface, 0x01)
        time.sleep(0.2)
        after = self.positions()
        print(
            f"[hw] pitch encoder after set_zero: {after[0]:+.4f} rad "
            f"({math.degrees(after[0]):+.2f} deg)"
        )
        if abs(after[0]) > 0.05:
            raise HardwareError(
                f"pitch did not land near 0 after set_zero (got {after[0]:+.4f}). "
                "Confirm the torso is on the -10 deg hard stop and ID 0x01 is pitch."
            )
        return float(after[0])

    def send_urdf(self, yaw_urdf: float, pitch_urdf: float) -> None:
        yaw_m, pitch_m = waist_motor_command(
            yaw_urdf,
            pitch_urdf,
            yaw_offset=self.yaw_offset,
            pitch_offset=self.pitch_offset,
            invert_yaw=self.invert_yaw,
            invert_pitch=self.invert_pitch,
        )
        self.send_motor(pitch_m, yaw_m)


class TeaArmHardware:
    def __init__(
        self,
        *,
        right_can: str = "can0",
        left_can: str = "can1",
        waist_can: str = "can2",
        yaw_offset: float = YAW_OFFSET_DEFAULT,
        pitch_offset: float = PITCH_OFFSET_DEFAULT,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
        pitch_kp: float = 150.0,
        pitch_kd: float = 4.5,
        yaw_kp: float = 70.0,
        yaw_kd: float = 2.5,
        home_max_speed: float = 0.25,
        home_min_duration: float = 8.0,
        gravity: GravityModel | None = None,
    ) -> None:
        try:
            from openarm_driver import Config, SingleArmDriver, set_default_config
        except ImportError as exc:
            raise SystemExit(
                "openarm_driver is required. From the teleop_xr directory run:\n"
                "  uv pip install -e ../openarm_driver"
            ) from exc

        require_can_interfaces([right_can, left_can, waist_can])
        config = Config()
        _patch_arm_config(config, right_can=right_can, left_can=left_can)
        set_default_config(config)

        self.right = SingleArmDriver("right_arm", config=config)
        self.left = SingleArmDriver("left_arm", config=config)
        self.waist = WaistDriver(
            waist_can,
            pitch_kp=pitch_kp,
            pitch_kd=pitch_kd,
            yaw_kp=yaw_kp,
            yaw_kd=yaw_kd,
            yaw_offset=yaw_offset,
            pitch_offset=pitch_offset,
            invert_yaw=invert_yaw,
            invert_pitch=invert_pitch,
        )
        self.home_max_speed = home_max_speed
        self.home_min_duration = home_min_duration
        self.gravity = gravity
        self._enabled = False
        self._shutting_down = False
        self._last_named: dict[str, float] | None = None
        self._last_left_grip = GRIPPER_CLOSED
        self._last_right_grip = GRIPPER_CLOSED

    def enable(self) -> None:
        try:
            _enable_arm(self.right)
            _enable_arm(self.left)
            self.waist.enable()
            self._enabled = True
            self.hold_current()
        except Exception:
            self.disable()
            raise

    def disable(self) -> None:
        try:
            if getattr(self.right, "started", False):
                self.right.openarm.disable_all()
                self.right.started = False
        finally:
            try:
                if getattr(self.left, "started", False):
                    self.left.openarm.disable_all()
                    self.left.started = False
            finally:
                self.waist.disable()
                self._enabled = False

    def _named_from_measured(
        self, wq: np.ndarray, rq: np.ndarray, lq: np.ndarray
    ) -> dict[str, float]:
        yaw_u, pitch_u = waist_urdf_from_motor(
            float(wq[1]),
            float(wq[0]),
            yaw_offset=self.waist.yaw_offset,
            pitch_offset=self.waist.pitch_offset,
            invert_yaw=self.waist.invert_yaw,
            invert_pitch=self.waist.invert_pitch,
        )
        named = {"waist_yaw": yaw_u, "waist_pitch": pitch_u}
        for i, name in enumerate(RIGHT_ARM_JOINTS):
            named[name] = float(rq[i])
        for i, name in enumerate(LEFT_ARM_JOINTS):
            named[name] = float(lq[i])
        return named

    def measured_urdf(self) -> dict[str, float]:
        return self._named_from_measured(
            self.waist.positions(),
            self.right.fetch_position(refresh=True),
            self.left.fetch_position(refresh=True),
        )

    def _waist_taus(self, named: Mapping[str, float]) -> tuple[float, float]:
        if self.gravity is None:
            return 0.0, 0.0
        taus = self.gravity.torques(named)
        return (
            motor_tau_from_urdf(
                taus.get("waist_pitch", 0.0), invert=self.waist.invert_pitch
            ),
            motor_tau_from_urdf(
                taus.get("waist_yaw", 0.0), invert=self.waist.invert_yaw
            ),
        )

    def _send_waist_motor(
        self,
        pitch: float,
        yaw: float,
        named: Mapping[str, float] | None = None,
        *,
        measure_gravity: bool = False,
    ) -> None:
        if measure_gravity:
            src = self.measured_urdf()
        else:
            src = named if named is not None else self._last_named
        tau_p, tau_y = self._waist_taus(src) if src is not None else (0.0, 0.0)
        self.waist.send_motor(pitch, yaw, tau_pitch=tau_p, tau_yaw=tau_y)

    def hold_current(self) -> None:
        rq = self.right.fetch_position(refresh=True)
        lq = self.left.fetch_position(refresh=True)
        wq = self.waist.positions()
        self.right.send_position(rq)
        self.left.send_position(lq)
        tau_p, tau_y = self._waist_taus(self._named_from_measured(wq, rq, lq))
        self.waist.send_motor(float(wq[0]), float(wq[1]), tau_pitch=tau_p, tau_yaw=tau_y)
        self._last_left_grip = float(lq[-1])
        self._last_right_grip = float(rq[-1])

    def send_named(
        self,
        named: Mapping[str, float],
        trigger_left: float,
        trigger_right: float,
        *,
        max_delta: float | None = None,
    ) -> None:
        if not self._enabled or self._shutting_down:
            return
        left_grip = gripper_hold_or_map(trigger_left, self._last_left_grip)
        right_grip = gripper_hold_or_map(trigger_right, self._last_right_grip)
        left_q = pack_arm(named, "left", left_grip)
        right_q = pack_arm(named, "right", right_grip)
        yaw_u, pitch_u = waist_urdf_command(named)
        if max_delta is not None:
            left_q = rate_limit(
                self.left.last_command, left_q, max_delta, free_tail=1
            )
            right_q = rate_limit(
                self.right.last_command, right_q, max_delta, free_tail=1
            )
            yaw_m, pitch_m = waist_motor_command(
                yaw_u,
                pitch_u,
                yaw_offset=self.waist.yaw_offset,
                pitch_offset=self.waist.pitch_offset,
                invert_yaw=self.waist.invert_yaw,
                invert_pitch=self.waist.invert_pitch,
            )
            limited = rate_limit(
                self.waist._last_motor, np.array([pitch_m, yaw_m]), max_delta
            )
            self.right.send_position(right_q)
            self.left.send_position(left_q)
            self._send_waist_motor(float(limited[0]), float(limited[1]), named)
        else:
            self.right.send_position(right_q)
            self.left.send_position(left_q)
            yaw_m, pitch_m = waist_motor_command(
                yaw_u,
                pitch_u,
                yaw_offset=self.waist.yaw_offset,
                pitch_offset=self.waist.pitch_offset,
                invert_yaw=self.waist.invert_yaw,
                invert_pitch=self.waist.invert_pitch,
            )
            self._send_waist_motor(pitch_m, yaw_m, named)
        self._last_named = dict(named)
        self._last_left_grip = float(left_q[-1])
        self._last_right_grip = float(right_q[-1])

    def hold(self) -> None:
        if self._shutting_down:
            return
        if self._last_named is None:
            self.hold_current()
            return
        self.right.send_position(self.right.last_command)
        self.left.send_position(self.left.last_command)
        self._send_waist_motor(
            float(self.waist._last_motor[0]), float(self.waist._last_motor[1])
        )

    def _interpolate_motors(
        self,
        left1: np.ndarray,
        right1: np.ndarray,
        waist1: np.ndarray,
        label: str,
        hz: float = 50.0,
    ) -> None:
        left0 = np.array(self.left.last_command, dtype=float)
        right0 = np.array(self.right.last_command, dtype=float)
        waist0 = np.array(self.waist._last_motor, dtype=float)
        dist = max(
            float(np.max(np.abs(left1 - left0))),
            float(np.max(np.abs(right1 - right0))),
            float(np.max(np.abs(waist1 - waist0))),
        )
        duration = max(self.home_min_duration, dist / max(self.home_max_speed, 1e-3))
        steps = max(1, int(hz * duration))
        print(
            f"[hw] interpolating to {label} over {duration:.1f}s "
            f"(max |dq|={dist:.3f} rad)"
        )
        dt = 1.0 / hz
        for i in range(1, steps + 1):
            a = i / steps
            self.left.send_position((1.0 - a) * left0 + a * left1)
            self.right.send_position((1.0 - a) * right0 + a * right1)
            w = (1.0 - a) * waist0 + a * waist1
            self._send_waist_motor(float(w[0]), float(w[1]), measure_gravity=True)
            time.sleep(dt)

    def go_to_rest(self, rest_named: Mapping[str, float], hz: float = 50.0) -> None:
        yaw_u, pitch_u = waist_urdf_command(rest_named)
        yaw_m, pitch_m = waist_motor_command(
            yaw_u,
            pitch_u,
            yaw_offset=self.waist.yaw_offset,
            pitch_offset=self.waist.pitch_offset,
            invert_yaw=self.waist.invert_yaw,
            invert_pitch=self.waist.invert_pitch,
        )
        self._interpolate_motors(
            pack_arm(rest_named, "left", GRIPPER_CLOSED),
            pack_arm(rest_named, "right", GRIPPER_CLOSED),
            np.array([pitch_m, yaw_m], dtype=float),
            "rest",
            hz=hz,
        )
        self._last_named = dict(rest_named)
        self._last_left_grip = GRIPPER_CLOSED
        self._last_right_grip = GRIPPER_CLOSED
        print("[hw] rest pose reached; squeeze both grips to follow")

    def go_to_motor_zero(self, hz: float = 50.0) -> None:
        self._interpolate_motors(
            np.zeros(8, dtype=float),
            np.zeros(8, dtype=float),
            np.zeros(2, dtype=float),
            "motor zero",
            hz=hz,
        )
        self._last_left_grip = GRIPPER_CLOSED
        self._last_right_grip = GRIPPER_CLOSED
        print("[hw] motor zero reached")

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self._enabled:
                print("[hw] shutdown: slow move to motor zero, then disable")
                self.go_to_motor_zero()
        except KeyboardInterrupt:
            print("[hw] second interrupt; disabling at current pose")
        self.disable()


def gripper_hold_or_map(trigger: float, last: float) -> float:
    from teleop_xr.hw.mapping import gripper_from_trigger

    if math.isnan(trigger):
        return last
    return gripper_from_trigger(trigger)
