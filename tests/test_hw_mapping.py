import math

import numpy as np

from teleop_xr.hw.mapping import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PITCH_OFFSET_DEFAULT,
    PITCH_URDF_MIN,
    gripper_from_trigger,
    named_positions,
    pack_arm,
    rate_limit,
    waist_motor_command,
    waist_urdf_command,
)


def test_gripper_same_sign_both_sides():
    assert gripper_from_trigger(1.0) == GRIPPER_CLOSED
    assert abs(gripper_from_trigger(0.0) - GRIPPER_OPEN) < 1e-9
    assert gripper_from_trigger(0.5) < 0.0


def test_rate_limit_skips_gripper():
    current = np.zeros(8)
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.209])
    out = rate_limit(current, target, 0.1, free_tail=1)
    assert abs(out[0] - 0.1) < 1e-12
    assert abs(out[-1] - (-1.209)) < 1e-12


def test_pack_arm_appends_gripper():
    names = [f"left_j{i}" for i in range(1, 8)] + [f"right_j{i}" for i in range(1, 8)]
    q = np.arange(len(names), dtype=float)
    named = named_positions(names, q)
    left = pack_arm(named, "left", -1.2)
    assert left.shape == (8,)
    assert left[-1] == -1.2
    assert list(left[:7]) == [named[f"left_j{i}"] for i in range(1, 8)]


def test_pitch_offset_is_zero_at_minus_10_stop():
    assert abs(PITCH_OFFSET_DEFAULT + PITCH_URDF_MIN) < 1e-12
    named = {
        "waist_yaw": 0.0,
        "waist_pitch": math.radians(-10.0),
    }
    yaw_u, pitch_u = waist_urdf_command(named)
    yaw_m, pitch_m = waist_motor_command(yaw_u, pitch_u)
    assert abs(yaw_m) < 1e-9
    assert abs(pitch_m) < 1e-9

    _, pitch_m = waist_motor_command(0.0, 0.0)
    assert abs(pitch_m - math.radians(10.0)) < 1e-9

    named["waist_pitch"] = math.radians(45.0)
    _, pitch_u = waist_urdf_command(named)
    _, pitch_m = waist_motor_command(0.0, pitch_u)
    assert abs(pitch_m - math.radians(55.0)) < 1e-9


def test_left_turn_increases_yaw_motor():
    yaw_m, _ = waist_motor_command(0.4, 0.0)
    assert yaw_m == 0.4


def test_waist_motor_urdf_roundtrip():
    from teleop_xr.hw.mapping import waist_urdf_from_motor

    yaw_m, pitch_m = waist_motor_command(-0.2, 0.1)
    yaw_u, pitch_u = waist_urdf_from_motor(yaw_m, pitch_m)
    assert abs(yaw_u + 0.2) < 1e-9
    assert abs(pitch_u - 0.1) < 1e-9
