# TeaArm 仿真接入真机：实现步骤

目标：仿真 `python -m teleop_xr.demo --mode ik --robot-class teaarm` 的 WebXR + pyroki IK 不变，把解出的关节下发到真机 CAN。入口为 `python -m teleop_xr.hw --robot-class teaarm`。

代码都在 `teleop_xr/teleop_xr/hw/`。

---

### 1. 确认仿真缺什么

仿真链路到 `IKController.step()` 为止：`q_new` 只画到头显孪生，电机收不到。

要补的是：把 16 维 IK（按名字：`waist_yaw` / `waist_pitch` / `left_j1..j7` / `right_j1..j7`）加上夹爪，变成三条 CAN 上的合法指令，并处理上电、跟随、退出。

夹爪不在 IK 里，用扳机 `gamepad.buttons[0]`。双手 grip `buttons[1]` 仍作 deadman。

---

### 2. 确认实机接线和电机

现场量过，按这个写驱动：

| 关节 | 电机 | send / recv | CAN | 供电 |
|---|---|---|---|---|
| pitch（点头，托双臂） | DM10010L | `0x01` / `0x11` | can2 FD 1M/5M | 独立 24V |
| yaw（转向） | DM8009 | `0x02` / `0x12` | can2 同上 | 同上 |
| 右臂 j1–j7 + 夹爪 | yaml 默认 | `0x01–0x08` / `0x11–0x18` | can0 | 独立 24V |
| 左臂 | 同右 | 同 | can1 | 独立 24V |

臂 `joint_offsets` 全 0。三路一起使能。Yaw 左转编码器增大，不 invert。

---

### 3. 写关节映射 `mapping.py`

作用：IK 字典 → 电机可吃的数。按名字取关节，缺名就报错，避免左右/腰切错。

- `pack_arm`：一侧 j1–j7 + 夹爪 → 8 维
- `waist_urdf_command`：取出 yaw/pitch，夹到 URDF 限位（pitch −10°～+45°，yaw ±180°）
- `waist_motor_command`：URDF → 电机角（加 offset / 可选 invert）
- 夹爪：闭合 0，全开 −1.209，左右同号。扳机 0→开、1→合：`grip = −1.209 × (1 − trigger)`
- 限位覆盖为 `[-1.25, 0.15]`，避免驱动按旧 yaml 把合法张开裁掉

---

### 4. 写 CAN 驱动 `driver.py`

作用：三条总线收发，不走 `SingleArmDriver.start()`（那条 yaml home 是 `j4=π/2`，和仿真 rest 不是同一套）。

- `WaistDriver`：can2 上 pitch=`DM10010L`、yaw=`DM8009`，MIT 位置 + 前馈力矩
- `TeaArmHardware`：右 `can0`、左 `can1`、腰 `can2`；启动前检查接口存在、UP、CAN FD（MTU≥72）
- 使能后先按当前编码器抱住，再慢速插值，避免上电抽一下

---

### 5. 标 pitch 零点

Pitch 硬限位是 URDF −10°。靠在限位上给 `0x01` 写 EEPROM 零，软件 offset = **+10°**：

`电机角 = URDF角 + 10°`

| URDF | 电机 |
|---|---|
| −10° 限位 | 0 |
| 0° 竖直 rest | +10° |
| +45° 前俯最低 | +55° |

只标一次。命令：`python -m teleop_xr.hw --calibrate-pitch-zero`（只写 pitch，yaw 和双臂不动）。标完编码器应接近 0。Yaw offset=0。

---

### 6. 加腰重力补偿 `gravity.py`

Pitch 托双臂，光靠 PD 会塌。用 Pinocchio 对 TeaArm URDF 算 `G(q)`（当前腰 + 双臂姿态），力矩只加在腰 MIT 的 `tau`。`--gravity-scale` 默认 1.0。

---

### 7. 接 XR 主循环 `__main__.py`

作用：仿真同一套 Teleop + IK，多一步下发。

1. 加载 `TeaArmRobot` + `PyrokiSolver` + `IKController`
2. 三路 enable，抱住当前角
3. `go_to_rest()`：50 Hz 插值到仿真 rest（≥8 s，≤0.25 rad/s）。rest 来自 `get_default_config()`：腰 0（竖直），双臂约  
   左 `66°/−19°/−16°/85°/−18°/−17°/−64°`，右除 j4 外符号相反，j4 同为 85°
4. 双手 grip 同时按下才跟随；松开则 hold
5. 跟随：IK → `send_named()`。臂+腰限速 0.8 rad/s；头显孪生仍 `publish_joint_state`
6. 右柄 B 或 Ctrl+C：停跟随 → 慢插值到**电机零**（臂全 0、腰电机 0 = pitch 在 −10° 限位、夹爪闭合）→ disable。插值中再 Ctrl+C 就地失能

---

### 8. 装本地驱动

`teleop_xr/.venv` 里：`uv pip install -e ../openarm_driver`，带上 `openarm-can==1.3.4`（`DM10010L` / `DM8009`）。没装会在 IK JIT 完之后报缺包退出。JAX CUDA 失败会回退 CPU，IK 仍可用。

---

### 9. 夹爪速度

夹爪曾和手臂绑在同一条 0.8 rad/s 限速上，全开到全合约 1.5 s。改成 `rate_limit(..., free_tail=1)`：j1–j7 仍限速，夹爪目标直达，由 POS_FORCE（50 rad/s、1 Nm）执行。

---

### 10. 调 pitch PD

默认曾抄手臂 J1：kp=70、kd=2.75。惯量是躯干+双臂，前俯会软、晃一下再停（欠阻尼）。改到 **kp=150、kd=4.5** 后写入默认。还晃可试 `200 / 5.0`（kd 不超过 5）。

---

### 运行

```bash
cd /home/cereboto/worktest/teleop_xr
source .venv/bin/activate
# can0/can1/can2 均为 CAN FD 1M/5M 且 UP
python -m teleop_xr.hw --robot-class teaarm
```
