"""Pinocchio gravity G(q), same role as openarm_teleop Dynamics::GetGravity.

Uses the measured joint configuration (waist + both arms). Pitch torque depends
on the arms' pose because that moves the center of mass.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from teleop_xr.hw.mapping import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS

_GRAVITY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


class GravityModel:
    def __init__(self, urdf_path: str, scale: float = 1.0) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.scale = float(scale)
        self._q_index: dict[str, int] = {}
        self._v_index: dict[str, int] = {}
        for name in _GRAVITY_JOINTS:
            if not self.model.existJointName(name):
                continue
            jid = self.model.getJointId(name)
            self._q_index[name] = int(self.model.idx_qs[jid])
            self._v_index[name] = int(self.model.idx_vs[jid])
        missing = [name for name in WAIST_JOINTS if name not in self._q_index]
        if missing:
            raise ValueError(f"URDF missing gravity joints: {missing}")

    def torques(self, named: Mapping[str, float]) -> dict[str, float]:
        q = self._pin.neutral(self.model)
        for name, idx in self._q_index.items():
            if name in named:
                q[idx] = float(named[name])
        self._pin.computeGeneralizedGravity(self.model, self.data, q)
        return {
            name: float(self.data.g[idx]) * self.scale
            for name, idx in self._v_index.items()
        }


def try_load_gravity(urdf_path: str | None, scale: float) -> GravityModel | None:
    if not urdf_path:
        return None
    try:
        return GravityModel(urdf_path, scale=scale)
    except Exception as exc:
        print(f"[hw] gravity model disabled: {exc}")
        return None
