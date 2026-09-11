"""Run TeaArm IK (same as the sim demo) and send joints to real motors.

CAN defaults: right=can0, left=can1, waist=can2 (CAN FD).
Waist: pitch DM10010L 0x01/0x11, yaw DM8009 0x02/0x12.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import tyro
from loguru import logger as loguru_logger

from teleop_xr import Teleop
from teleop_xr.common_cli import CommonCLI
from teleop_xr.config import TeleopSettings
from teleop_xr.hw.driver import HardwareError, TeaArmHardware, WaistDriver, require_can_interfaces
from teleop_xr.hw.gravity import try_load_gravity
from teleop_xr.hw.mapping import PITCH_OFFSET_DEFAULT, YAW_OFFSET_DEFAULT, named_positions
from teleop_xr.events import EventProcessor, XRButton, XRController
from teleop_xr.ik_utils import ensure_ik_dependencies, list_robots_or_exit
from teleop_xr.messages import XRDeviceRole, XRHandedness, XRState


@dataclass
class HardwareCLI(CommonCLI):
    robot_class: str = "teaarm"
    robot_args: str = "{}"
    list_robots: bool = False
    right_can: str = "can0"
    left_can: str = "can1"
    waist_can: str = "can2"
    pitch_offset: float = PITCH_OFFSET_DEFAULT
    yaw_offset: float = YAW_OFFSET_DEFAULT
    invert_yaw: bool = False
    invert_pitch: bool = False
    pitch_kp: float = 150.0
    pitch_kd: float = 4.5
    yaw_kp: float = 70.0
    yaw_kd: float = 2.5
    gravity: bool = True
    gravity_scale: float = 1.0
    home_max_speed: float = 0.25
    home_min_duration: float = 8.0
    teleop_max_speed: float = 0.8
    can_hz: float = 250.0
    ik_max_iterations: int = 10
    self_collision: bool = False
    calibrate_pitch_zero: bool = False


def triggers_from_state(state: XRState) -> tuple[float, float]:
    left = float("nan")
    right = float("nan")
    for device in state.devices:
        if device.role != XRDeviceRole.CONTROLLER or device.gamepad is None:
            continue
        if not device.gamepad.buttons:
            continue
        value = float(device.gamepad.buttons[0].value)
        if device.handedness == XRHandedness.LEFT:
            left = value
        elif device.handedness == XRHandedness.RIGHT:
            right = value
    return left, right


class HardwareIKWorker(threading.Thread):
    """Solve IK from XR packets. Does not touch CAN."""

    def __init__(
        self,
        controller,
        robot,
        state_container: dict[str, Any],
        command_box: dict[str, Any],
        command_lock: threading.Lock,
        logger: logging.Logger,
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.robot = robot
        self.state_container = state_container
        self.command_box = command_box
        self.command_lock = command_lock
        self.logger = logger
        self.latest_xr_state: Optional[XRState] = None
        self.new_state_event = threading.Event()
        self.running = True

    def update_state(self, state: XRState) -> None:
        self.latest_xr_state = state
        self.new_state_event.set()

    def run(self) -> None:
        while self.running:
            if not self.new_state_event.wait(timeout=0.1):
                continue
            self.new_state_event.clear()
            state = self.latest_xr_state
            if state is None:
                continue
            try:
                with self.command_lock:
                    q_current = np.array(self.state_container["q"], copy=True)
                    was_active = self.controller.active
                new_config = np.array(self.controller.step(state, q_current))
                named = named_positions(self.robot.actuated_joint_names, new_config)
                trig_l, trig_r = triggers_from_state(state)
                with self.command_lock:
                    self.state_container["q"] = new_config
                    self.state_container["active"] = self.controller.active
                    self.command_box["named"] = named
                    self.command_box["trigger_left"] = trig_l
                    self.command_box["trigger_right"] = trig_r
                if not was_active and self.controller.active:
                    self.logger.info("in_control start")
                if was_active and not self.controller.active:
                    self.logger.info("in_control stop; holding last pose")
            except Exception as exc:
                self.logger.error(f"IK step failed: {exc}")


class HardwareCANWorker(threading.Thread):
    """Fixed-rate CAN replay of the latest IK command."""

    def __init__(
        self,
        hardware: TeaArmHardware,
        command_box: dict[str, Any],
        command_lock: threading.Lock,
        logger: logging.Logger,
        teleop_max_speed: float,
        can_hz: float,
    ):
        super().__init__(daemon=True)
        self.hardware = hardware
        self.command_box = command_box
        self.command_lock = command_lock
        self.logger = logger
        self.teleop_max_speed = teleop_max_speed
        self.period = 1.0 / max(can_hz, 1.0)
        self.running = True
        self._last_send_t = time.perf_counter()

    def run(self) -> None:
        while self.running:
            t0 = time.perf_counter()
            if self.hardware._shutting_down:
                break
            try:
                with self.command_lock:
                    named = self.command_box.get("named")
                    if named is not None:
                        named = dict(named)
                    trig_l = self.command_box.get("trigger_left", float("nan"))
                    trig_r = self.command_box.get("trigger_right", float("nan"))
                now = time.perf_counter()
                dt = max(1e-3, now - self._last_send_t)
                self._last_send_t = now
                if named is not None:
                    self.hardware.send_named(
                        named,
                        trig_l,
                        trig_r,
                        max_delta=self.teleop_max_speed * dt,
                    )
                else:
                    self.hardware.hold()
            except Exception as exc:
                self.logger.error(f"CAN step failed: {exc}")
            sleep_s = self.period - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)


def calibrate_pitch_zero(can_interface: str) -> None:
    require_can_interfaces([can_interface])
    waist = WaistDriver(can_interface)
    print(
        "[hw] sit on the pitch -10 deg hard stop, then this writes EEPROM zero "
        "for pitch 0x01 only (yaw 0x02 is not changed)."
    )
    waist.set_pitch_zero()
    print(
        f"[hw] done. teleop offset is +10 deg ({PITCH_OFFSET_DEFAULT:.4f} rad): "
        "URDF -10 deg → motor 0, URDF 0 (vertical) → motor +10 deg."
    )


def main() -> None:
    cli = tyro.cli(HardwareCLI)
    if cli.calibrate_pitch_zero:
        try:
            calibrate_pitch_zero(cli.waist_can)
        except HardwareError as exc:
            raise SystemExit(str(exc)) from exc
        return

    ensure_ik_dependencies()
    if cli.list_robots:
        list_robots_or_exit()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("hw")
    # loguru defaults to DEBUG and would print 16-D q every IK tick in the
    # control thread; that stalls CAN when the terminal cannot keep up.
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="INFO")

    from teleop_xr.ik.controller import IKController
    from teleop_xr.ik.loader import load_robot_class
    from teleop_xr.ik.solver import PyrokiSolver

    robot_cls = load_robot_class(cli.robot_class)
    robot_kwargs = json.loads(cli.robot_args)
    if cli.robot_class == "teaarm":
        robot_kwargs.setdefault("self_collision", cli.self_collision)
    robot = robot_cls(**robot_kwargs)
    controller = IKController(
        robot, PyrokiSolver(robot, max_iterations=cli.ik_max_iterations)
    )
    rest = named_positions(robot.actuated_joint_names, np.array(robot.get_default_config()))
    state_container: dict[str, Any] = {
        "q": np.array(robot.get_default_config()),
        "active": False,
    }
    command_lock = threading.Lock()
    command_box: dict[str, Any] = {
        "named": None,
        "trigger_left": float("nan"),
        "trigger_right": float("nan"),
    }

    gravity = None
    if cli.gravity:
        gravity = try_load_gravity(getattr(robot, "urdf_path", None), cli.gravity_scale)
        if gravity is None:
            logger.warning("gravity compensation off (URDF/pinocchio unavailable)")
        else:
            logger.info(
                "gravity compensation on (scale=%.2f); pitch kp=%.1f",
                cli.gravity_scale,
                cli.pitch_kp,
            )

    hardware = TeaArmHardware(
        right_can=cli.right_can,
        left_can=cli.left_can,
        waist_can=cli.waist_can,
        yaw_offset=cli.yaw_offset,
        pitch_offset=cli.pitch_offset,
        invert_yaw=cli.invert_yaw,
        invert_pitch=cli.invert_pitch,
        pitch_kp=cli.pitch_kp,
        pitch_kd=cli.pitch_kd,
        yaw_kp=cli.yaw_kp,
        yaw_kd=cli.yaw_kd,
        home_max_speed=cli.home_max_speed,
        home_min_duration=cli.home_min_duration,
        gravity=gravity,
    )

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        robot_vis=None,
        input_mode=cli.input_mode,
        speed=robot.default_speed_ratio,
        enable_video=False,
    )
    teleop = Teleop(settings=settings)
    teleop.set_pose(np.eye(4))
    teleop.bind_control_mode_provider(lambda: controller.get_mode().value)

    ik_worker = HardwareIKWorker(
        controller,
        robot,
        state_container,
        command_box,
        command_lock,
        logger,
    )
    can_worker = HardwareCANWorker(
        hardware,
        command_box,
        command_lock,
        logger,
        cli.teleop_max_speed,
        cli.can_hz,
    )

    quit_once = threading.Event()

    def request_quit(_event=None) -> None:
        if quit_once.is_set():
            return
        quit_once.set()
        logger.info("quit requested (right B or Ctrl+C); returning to motor zero")
        os.kill(os.getpid(), signal.SIGINT)

    processor = EventProcessor(cli.event_settings())
    processor.on_button_down(
        button=XRButton.BUTTON_SECONDARY,
        controller=XRController.RIGHT,
        callback=lambda event: request_quit(event),
    )

    def on_xr_update(_pose: np.ndarray, message: dict[str, Any]) -> None:
        try:
            xr_data = message.get("data", message)
            processor.process(_pose, xr_data)
            ik_worker.update_state(XRState.model_validate(xr_data))
        except Exception:
            pass

    teleop.subscribe(on_xr_update)

    logger.info(
        "waist pitch=DM10010L 0x01/0x11 kp=%.1f kd=%.1f offset=%.3f; "
        "yaw=DM8009 0x02/0x12 kp=%.1f kd=%.1f",
        cli.pitch_kp,
        cli.pitch_kd,
        cli.pitch_offset,
        cli.yaw_kp,
        cli.yaw_kd,
    )
    logger.info(
        "IK %d iters, self_collision=%s, CAN %.0f Hz, video off, twin off",
        cli.ik_max_iterations,
        cli.self_collision,
        cli.can_hz,
    )
    logger.info("enabling can0/can1/can2 and holding current pose")
    logger.info("right controller B or Ctrl+C: slow return to motor zero, then disable")
    hardware.enable()
    hardware.go_to_rest(rest)
    command_box["named"] = dict(rest)
    ik_worker.start()
    can_worker.start()
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        ik_worker.running = False
        can_worker.running = False
        ik_worker.join(timeout=2.0)
        can_worker.join(timeout=2.0)
        hardware.shutdown()


if __name__ == "__main__":
    main()
