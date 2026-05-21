# IsaacLab xArm7 Training File Guide

说明：
- 本文档只关注 IsaacLab 侧和训练直接相关的文件。
- MuJoCo 目录目前仍保留在 `mywork/mymujoco/`，但这里不展开整理，只把它视为 `xarm7.urdf` 的来源依赖。

## 1. 推荐关注的目录

```text
IsaacLab-main/
├─ mywork/
│  ├─ isaaclab_training/
│  ├─ PROJECT_FILE_GUIDE.md
│  └─ mymujoco/                # 仅作为 URDF/资源依赖保留
├─ source/
├─ scripts/
└─ logs/
```

## 2. IsaacLab 训练主链路

当前这套 xArm7 抓取任务的训练链路可以按下面理解：

1. `xarm7.urdf` 提供机器人和夹爪结构
2. `source/isaaclab_assets/.../xarm7.py` 把 URDF 接成 IsaacLab 机器人资产
3. `source/isaaclab_tasks/.../lift/config/xarm7/` 定义抓方块任务
4. `source/isaaclab_tasks/.../agents/rsl_rl_ppo_cfg.py` 定义 PPO 训练参数
5. `scripts/reinforcement_learning/rsl_rl/train.py` 启动训练
6. `logs/rsl_rl/xarm7_lift/...` 保存 checkpoint、配置快照和回放视频

## 3. 关键代码文件

### 机器人资产

- `source/isaaclab_assets/isaaclab_assets/robots/xarm7.py`
  作用：IsaacLab 的 xArm7 机器人资产配置入口。
  这里负责定位 `xarm7.urdf`、设置驱动器、指定初始关节角，并处理 URDF 到 USD 的缓存目录。

### 抓方块任务

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/__init__.py`
  作用：注册 xArm7 抓方块任务的 Gym ID。

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/joint_pos_env_cfg.py`
  作用：xArm7 抓方块任务环境配置。
  这里定义了：
  - 机器人类型
  - 机械臂动作空间
  - 夹爪开合动作
  - 方块初始随机范围
  - 目标位置范围
  - 播放和录屏时的相机视角

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/agents/rsl_rl_ppo_cfg.py`
  作用：xArm7 抓方块任务的 PPO 超参数配置。

### 通用任务逻辑

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py`
  作用：lift 任务的通用环境结构，定义 observation、reward、event、termination 的基本框架。

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py`
  作用：抓方块任务的 reward 函数实现。

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/observations.py`
  作用：抓方块任务的关键观测实现，例如物块在机器人根坐标系下的位置。

### 技能扩展相关

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/`
  作用：新补的扭瓶盖技能骨架。

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/xarm7_skill_lib/registry.py`
  作用：xArm7 技能注册表，当前收录了 `lift_cube` 和 `twist_cap`。

## 4. 训练结果文件

### 训练日志目录

- `logs/rsl_rl/xarm7_lift/`
  作用：xArm7 抓方块任务的训练输出根目录。

### 具体一次训练的结果

以这次训练为例：

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/model_2999.pt`
  作用：训练得到的最终 checkpoint。

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/params/env.yaml`
  作用：该次训练实际使用的环境配置快照。

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/params/agent.yaml`
  作用：该次训练实际使用的 PPO 配置快照。

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/exported/policy.pt`
  作用：导出的策略模型。

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/exported/policy.onnx`
  作用：ONNX 格式导出模型。

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/videos/play/rl-video-step-0.mp4`
  作用：`play.py --video` 录制的回放视频。

## 5. 与训练直接相关的说明文档

- `mywork/isaaclab_training/README.md`
  作用：IsaacLab 训练资料入口说明。

- `mywork/isaaclab_training/xarm7_isaaclab_group_meeting.md`
  作用：面向组会讲解的 IsaacLab 训练说明。

- `mywork/isaaclab_training/isaaclab_h5py_check.txt`
  作用：之前修复 `play.py` 与 `h5py` 导入冲突时保留的排查记录。

## 6. VS Code 与启动入口

- `.vscode/launch.json`
  作用：VS Code 调试启动配置。
  目前已经补了：
  - `Python: Train xArm7 Lift`
  - `Python: Play xArm7 Lift`
  - `Python: Record xArm7 Lift Video`

- `.vscode/tasks.json`
  作用：VS Code 任务入口。
  目前已经补了：
  - `xarm7_train_headless`
  - `xarm7_play`
  - `xarm7_record_video`

## 7. 当前最值得看的文件

如果你现在只想快速把训练链路讲清楚，优先看这几个：

1. `source/isaaclab_assets/isaaclab_assets/robots/xarm7.py`
2. `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/joint_pos_env_cfg.py`
3. `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/agents/rsl_rl_ppo_cfg.py`
4. `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/params/env.yaml`
5. `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/params/agent.yaml`
6. `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/model_2999.pt`

## 8. 当前整理约定

- `mywork/isaaclab_training/` 只放 IsaacLab 训练相关说明资料
- `source/` 继续保留正式源码，不挪动
- `logs/` 继续保留 IsaacLab 标准训练输出，不改训练框架默认习惯
- `mywork/mymujoco/` 先保留为依赖目录，但不作为这次整理重点
