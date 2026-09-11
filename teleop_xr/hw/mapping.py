"""Map TeaArm IK joints and XR triggers onto motor commands.

Measured on the real cell (2026-09-07):

- pitch EEPROM zero is the -10 deg hard stop; offset is +10 deg so URDF 0 is vertical
- yaw increases when turning left (same sign as URDF +Z)
- both grippers: 0 = closed, -1.209 = fully open
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

LEFT_ARM_JOINTS = tuple(f"left_j{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"right_j{i}" for i in range(1, 8))
WAIST_JOINTS = ("waist_yaw", "waist_pitch")

PITCH_URDF_MIN = math.radians(-10.0)
PITCH_URDF_MAX = math.radians(45.0)
YAW_URDF_MIN = -math.pi
YAW_URDF_MAX = math.pi

# After set_zero at the -10 deg stop: motor = urdf_pitch - (-10 deg)
PITCH_OFFSET_DEFAULT = -PITCH_URDF_MIN
YAW_OFFSET_DEFAULT = 0.0

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = -1.209
GRIPPER_LIMITS = (-1.25, 0.15)


def named_positions(joint_names: list[str], q: np.ndarray) -> dict[str, float]:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != len(joint_names):
        raise ValueError(
            f"IK size {q.shape[0]} does not match joint names {len(joint_names)}"
        )
    return {name: float(val) for name, val in zip(joint_names, q)}


def require_named(
    named: Mapping[str, float], keys: tuple[str, ...]
) -> dict[str, float]:
    missing = [key for key in keys if key not in named]
    if missing:
        raise KeyError(f"IK output missing joints: {missing}")
    return {key: float(named[key]) for key in keys}


def gripper_from_trigger(trigger: float, open_pos: float = GRIPPER_OPEN) -> float:
    """trigger 0 → open, trigger 1 → closed. Same sign on both arms."""
    t = float(np.clip(trigger, 0.0, 1.0))
    return float(open_pos * (1.0 - t))


def pack_arm(named: Mapping[str, float], side: str, gripper: float) -> np.ndarray:
    joints = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
    q = require_named(named, joints)
    return np.array([q[name] for name in joints] + [float(gripper)], dtype=float)


def waist_urdf_command(named: Mapping[str, float]) -> tuple[float, float]:
    q = require_named(named, WAIST_JOINTS)
    yaw = float(np.clip(q["waist_yaw"], YAW_URDF_MIN, YAW_URDF_MAX))
    pitch = float(np.clip(q["waist_pitch"], PITCH_URDF_MIN, PITCH_URDF_MAX))
    return yaw, pitch


def waist_motor_command(
    yaw_urdf: float,
    pitch_urdf: float,
    *,
    yaw_offset: float = YAW_OFFSET_DEFAULT,
    pitch_offset: float = PITCH_OFFSET_DEFAULT,
    invert_yaw: bool = False,
    invert_pitch: bool = False,
) -> tuple[float, float]:
    yaw = (-yaw_urdf if invert_yaw else yaw_urdf) + yaw_offset
    pitch = (-pitch_urdf if invert_pitch else pitch_urdf) + pitch_offset
    return float(yaw), float(pitch)


def waist_urdf_from_motor(
    yaw_motor: float,
    pitch_motor: float,
    *,
    yaw_offset: float = YAW_OFFSET_DEFAULT,
    pitch_offset: float = PITCH_OFFSET_DEFAULT,
    invert_yaw: bool = False,
    invert_pitch: bool = False,
) -> tuple[float, float]:
    yaw = yaw_motor - yaw_offset
    pitch = pitch_motor - pitch_offset
    if invert_yaw:
        yaw = -yaw
    if invert_pitch:
        pitch = -pitch
    return float(yaw), float(pitch)


def motor_tau_from_urdf(
    tau_urdf: float, *, invert: bool = False, limit: float = 80.0
) -> float:
    tau = -tau_urdf if invert else tau_urdf
    return float(np.clip(tau, -limit, limit))


def rate_limit(
    current: np.ndarray,
    target: np.ndarray,
    max_delta: float,
    *,
    free_tail: int = 0,
) -> np.ndarray:
    """Slew toward target; the last ``free_tail`` elements jump immediately.

    Arm teleop uses ``free_tail=1`` so the gripper is not tied to joint speed.
    """
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    out = current + np.clip(target - current, -max_delta, max_delta)
    if free_tail:
        out[-free_tail:] = target[-free_tail:]
    return out
