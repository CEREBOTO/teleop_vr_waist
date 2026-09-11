"""Hardware teleop for TeaArm: same IK as the sim, then CAN dispatch."""

from teleop_xr.hw.driver import TeaArmHardware, WaistDriver
from teleop_xr.hw.gravity import GravityModel
from teleop_xr.hw.mapping import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
    gripper_from_trigger,
    named_positions,
    pack_arm,
    waist_motor_command,
    waist_urdf_command,
)

__all__ = [
    "TeaArmHardware",
    "WaistDriver",
    "LEFT_ARM_JOINTS",
    "RIGHT_ARM_JOINTS",
    "WAIST_JOINTS",
    "gripper_from_trigger",
    "named_positions",
    "pack_arm",
    "waist_motor_command",
    "waist_urdf_command",
]
