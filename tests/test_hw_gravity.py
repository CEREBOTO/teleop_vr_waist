import math
from pathlib import Path

import pytest

try:
    import pinocchio  # noqa: F401
except ImportError:
    pytest.skip("pinocchio not installed", allow_module_level=True)

from teleop_xr.hw.gravity import GravityModel

_MINIMAL_PITCH_URDF = """
<robot name="pitch_only">
  <link name="base_link"/>
  <link name="waist_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
  <link name="torso_link">
    <inertial>
      <origin xyz="0.3 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="waist_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="waist_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="waist_pitch" type="revolute">
    <parent link="waist_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.1" rpy="0 1.5708 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-0.2" upper="0.8" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def test_gravity_changes_with_pitch(tmp_path: Path):
    urdf = tmp_path / "pitch.urdf"
    urdf.write_text(_MINIMAL_PITCH_URDF)
    model = GravityModel(str(urdf))
    tau0 = model.torques({"waist_yaw": 0.0, "waist_pitch": 0.0})
    tau1 = model.torques({"waist_yaw": 0.0, "waist_pitch": math.radians(30.0)})
    assert "waist_pitch" in tau0
    assert tau0["waist_pitch"] != tau1["waist_pitch"]
