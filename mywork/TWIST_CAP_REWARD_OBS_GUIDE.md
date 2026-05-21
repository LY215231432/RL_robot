# Twist Cap Grasp Reward and Observation Guide

本文档整理 xArm7 机械臂抓取瓶盖任务中与 reward 和 observation 相关的核心文件，重点说明“定位瓶盖 -> 从上方下压 -> 夹稳瓶盖”这一阶段。旋拧瓶盖相关 reward 也在同一个 reward 文件中，但本文会单独标出，避免和抓取阶段混在一起。

## 1. 核心文件

| 类型 | 文件路径 | 作用 |
| --- | --- | --- |
| Reward 函数定义 | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/rewards.py` | 定义瓶盖定位、下压、夹取、接触、旋拧等奖励函数 |
| Observation 函数定义 | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/observations.py` | 定义策略可观测到的瓶盖位置、夹爪中心、夹爪到瓶盖误差等状态 |
| Reward/Obs 配置入口 | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/twist_cap_env_cfg.py` | 把 reward 函数和 obs 函数注册到 IsaacLab 任务配置中，并设置权重 |
| MDP 统一导出 | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/__init__.py` | 将 `observations.py`、`rewards.py`、`terminations.py` 中的函数导出为 `mdp.xxx` |

如果只是向老师说明“抓取瓶盖任务怎么设计 reward 和 obs”，建议至少发送：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/rewards.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/observations.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/twist_cap_env_cfg.py
```

## 2. Observation 文件整理

Observation 文件：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/observations.py
```

该文件的核心作用是把仿真世界中的瓶盖、夹爪、机械臂状态转换为策略网络可以直接输入的向量。

### 2.1 常量

| 常量 | 含义 |
| --- | --- |
| `CAP_SIDE_GRASP_HEIGHT = 0.018` | 瓶盖抓取目标点相对于瓶盖中心的高度偏移 |
| `CAP_APPROACH_CLEARANCE = 0.055` | 预抓取点在瓶盖抓取点上方的安全高度 |
| `LEFT_FINGER_PAD_CENTER_LOCAL` | 左夹爪指尖接触区域在左爪坐标系中的局部点 |
| `RIGHT_FINGER_PAD_CENTER_LOCAL` | 右夹爪指尖接触区域在右爪坐标系中的局部点 |

### 2.2 辅助函数

| 函数 | 作用 |
| --- | --- |
| `_single_body_position()` | 从 IsaacLab 返回的 body tensor 中取出单个刚体的位置 |
| `_single_body_quaternion()` | 从 body tensor 中取出单个刚体的四元数 |
| `_body_point_w()` | 把 link 上的局部点转换到世界坐标系 |
| `_gripper_pad_midpoint_w()` | 计算左右两个夹爪指尖中心点的世界坐标中点 |
| `_cap_grasp_target_w()` | 计算瓶盖实际抓取目标点 |
| `_cap_approach_target_w()` | 计算瓶盖上方的预抓取目标点 |

其中 `_gripper_pad_midpoint_w()` 很关键。之前不是直接用 TCP 中心，而是用左右爪尖中心点作为夹取定位依据，这样 reward 和 obs 都更贴近真实夹爪接触瓶盖的位置。

### 2.3 策略使用的 Obs 项

这些 obs 在 `twist_cap_env_cfg.py` 的 `ObservationsCfg.PolicyCfg` 中被注册：

| Obs 名称 | 函数 | 含义 |
| --- | --- | --- |
| `joint_pos` | `mdp.joint_pos_rel` | 机械臂关节相对位置 |
| `joint_vel` | `mdp.joint_vel_rel` | 机械臂关节相对速度 |
| `cap_position` | `cap_position_in_robot_root_frame()` | 瓶盖中心在机器人根坐标系下的位置 |
| `gripper_center_pos` | `gripper_center_position_in_robot_root_frame()` | 左右爪尖中心点在机器人根坐标系下的位置 |
| `gripper_center_to_cap` | `gripper_center_to_cap()` | 爪尖中心到瓶盖中心的误差向量 |
| `gripper_center_to_approach` | `gripper_center_to_approach_target()` | 爪尖中心到瓶盖上方预抓取点的误差向量 |
| `cap_angle` | `cap_joint_position()` | 当前瓶盖关节角度 |
| `cap_angle_error` | `cap_angle_error()` | 当前瓶盖角度距离目标旋转角的误差 |
| `actions` | `mdp.last_action` | 上一步策略输出动作 |

抓取阶段最关键的 obs 是：

```text
gripper_center_to_approach
gripper_center_to_cap
gripper_center_pos
cap_position
actions
```

其中：

```text
gripper_center_to_approach = 瓶盖上方预抓取点 - 爪尖中心点
gripper_center_to_cap = 瓶盖中心/抓取点 - 爪尖中心点
```

策略通过这两个误差向量知道自己应该先移动到瓶盖上方，再向下靠近瓶盖。

## 3. Reward 文件整理

Reward 文件：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/rewards.py
```

该文件将抓取瓶盖拆成多个阶段：

```text
保持末端垂直
动作朝瓶盖方向移动
爪尖中心对准瓶盖上方
从上方下降到瓶盖
夹爪闭合并产生左右接触
夹稳后再旋转瓶盖
```

### 3.1 抓取阶段关键 reward 函数

| 函数 | 阶段 | 作用 |
| --- | --- | --- |
| `tcp_vertical_alignment()` | 姿态约束 | 奖励 TCP 末端轴线垂直于地面，避免斜着插向瓶盖 |
| `vertical_press_posture()` | 关节姿态辅助 | 奖励机械臂保持适合从上方下压的关节姿态 |
| `approach_action_towards_cap()` | 动作引导 | 奖励策略输出的 3D IK 动作朝向瓶盖上方目标点 |
| `gripper_center_to_approach_alignment()` | 粗定位 | 奖励爪尖中心靠近瓶盖上方预抓取点，主要强调 XY 对准 |
| `approach_cap_from_above()` | 上方定位 | 奖励爪尖中心位于瓶盖上方，而不是从侧面靠近 |
| `gripper_center_to_cap_alignment()` | 精定位 | 奖励左右爪尖中心对准瓶盖抓取目标点 |
| `side_approach_penalty()` | 反向约束 | 惩罚从瓶盖侧面错误靠近，减少撞瓶身/侧边接触 |
| `descend_to_cap_grasp()` | 下压抓取 | 当爪尖已在瓶盖上方时，奖励沿 Z 方向下降到抓取高度 |
| `grasp_cap()` | 夹稳瓶盖 | 奖励夹爪闭合并且左右夹爪都与瓶盖产生接触 |

### 3.2 旋拧阶段 reward 函数

这些函数不是“抓取瓶盖”的核心，但在同一个 reward 文件中，用于夹稳后的拧瓶盖阶段：

| 函数 | 作用 |
| --- | --- |
| `twist_tcp_angular_velocity()` | 奖励 TCP 围绕瓶盖轴向产生正向旋转速度 |
| `cap_rotation_progress()` | 奖励瓶盖关节角度向目标角度增加 |
| `cap_positive_rotation_velocity()` | 奖励瓶盖关节产生正向旋转速度 |

在 `twist_cap_env_cfg.py` 中，这些旋拧 reward 初始权重为 `0.0`，通过 curriculum 在训练后期启用，避免模型还没抓住瓶盖就急着旋转。

## 4. Reward 配置权重

Reward 权重在：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/twist_cap_env_cfg.py
```

当前抓取阶段主要 reward 配置如下：

| Reward term | 初始权重 | 后期 curriculum 权重 | 作用 |
| --- | ---: | ---: | --- |
| `vertical_tcp` | `8.0` | `4.0` | 保持末端垂直 |
| `vertical_press_posture` | `1.0` | `0.5` | 保持适合下压的关节姿态 |
| `approach_action_direction` | `7.0` | `2.0` | 鼓励动作朝瓶盖上方移动 |
| `approach_center_coarse` | `12.0` | `3.0` | 粗定位到瓶盖上方 |
| `finger_center_on_cap` | `9.0` | `4.0` | 爪尖中心对准瓶盖 |
| `side_approach` | `-3.0` | `-1.5` | 惩罚从侧面靠近 |
| `find_cap_above` | `6.0` | `2.5` | 找到瓶盖上方位置 |
| `descend_to_grasp` | `7.0` | `3.0` | 从上方下压到瓶盖 |
| `grasp_cap` | `10.0` | `6.0` | 闭合夹爪并产生双侧接触 |

这里的设计逻辑是：

```text
前期：强引导机械臂先找到瓶盖上方位置
中期：强化爪尖中心对准瓶盖和下压抓取
后期：在能夹稳后逐渐加入旋拧奖励
```

## 5. 抓取瓶盖的 reward 流程

可以把当前 reward 理解为一个分阶段任务：

### 阶段 1：找瓶盖上方

相关 reward：

```text
approach_action_direction
approach_center_coarse
find_cap_above
vertical_tcp
```

目标是让爪尖中心移动到瓶盖上方的预抓取点，而不是直接碰瓶盖。

### 阶段 2：对准瓶盖中心

相关 reward：

```text
finger_center_on_cap
side_approach
```

目标是让左右爪尖中心和瓶盖中心对齐，避免从侧面撞瓶盖或瓶身。

### 阶段 3：从上方下压

相关 reward：

```text
descend_to_grasp
vertical_press_posture
```

目标是让机械臂在对准瓶盖后再向下靠近，形成从上到下的抓取过程。

### 阶段 4：夹稳瓶盖

相关 reward：

```text
grasp_cap
```

`grasp_cap()` 同时检查：

```text
夹爪是否闭合
左夹爪是否接触瓶盖
右夹爪是否接触瓶盖
是否已经完成下降接近阶段
```

只有满足这些条件时，才认为完成了稳定抓取。

## 6. 设计要点总结

当前抓取瓶盖任务不是简单奖励“夹爪闭合”，而是通过多个 reward 共同塑造动作链：

```text
看到瓶盖位置
知道爪尖中心位置
计算爪尖中心到瓶盖上方目标的误差
动作朝误差方向移动
保持末端垂直
从上方下降
闭合夹爪
通过左右接触传感器确认夹稳
```

这种设计的好处是：

```text
不会鼓励机械臂在远处乱闭合
不会鼓励从瓶盖侧面撞过去
可以把抓取和旋拧拆成两个学习阶段
爪尖中心作为定位点，比单纯使用 TCP 更贴近真实夹取动作
```

## 7. 给老师说明时可以这样概括

本项目中，抓取瓶盖的 observation 主要包含瓶盖位置、爪尖中心位置、爪尖中心到瓶盖和预抓取点的误差向量、瓶盖关节角度以及上一时刻动作。Reward 采用分阶段设计：先奖励机械臂将爪尖中心移动到瓶盖上方，再奖励从上方垂直下压，最后通过夹爪闭合程度和左右接触传感器判断是否夹稳瓶盖。在夹稳之前，旋拧相关 reward 初始权重为 0，避免模型尚未抓住瓶盖就学习旋转动作。

