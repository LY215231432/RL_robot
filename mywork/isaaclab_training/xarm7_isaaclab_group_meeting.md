# xArm7 IsaacLab 训练组会说明

## 1. 目标

这套工作要完成的是：

- 将 xArm7 机器人接入 IsaacLab
- 定义一个抓取方块并抬起的强化学习任务
- 使用 PPO 完成训练
- 使用 checkpoint 做回放和录屏验证

这次讲解只聚焦 IsaacLab 训练链路，不展开 MuJoCo 侧工程细节。

## 2. 训练链路

完整链路可以概括成：

`URDF -> IsaacLab 机器人资产 -> lift 任务配置 -> PPO 训练 -> checkpoint -> play / video`

对应到代码上：

1. `source/isaaclab_assets/isaaclab_assets/robots/xarm7.py`
   负责把 xArm7 的 URDF 接成 IsaacLab 可用的 articulation。

2. `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/joint_pos_env_cfg.py`
   负责定义 xArm7 抓方块任务。

3. `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/agents/rsl_rl_ppo_cfg.py`
   负责定义 PPO 训练超参数。

4. `scripts/reinforcement_learning/rsl_rl/train.py`
   作为通用训练入口，根据 task id 加载环境配置和 agent 配置。

5. `logs/rsl_rl/xarm7_lift/...`
   保存训练得到的 checkpoint、配置快照和回放视频。

## 3. 机器人接入

机器人资产入口在：

- `source/isaaclab_assets/isaaclab_assets/robots/xarm7.py`

这个文件做了三件关键事：

1. 定位本地 `xarm7.urdf`
2. 配置关节驱动器和初始关节角
3. 指定 URDF 转 USD 的缓存目录

也就是说，IsaacLab 训练不是直接“读一个 mesh”，而是先把 URDF 解析成仿真里的机器人资产，再由环境使用。

## 4. 任务定义

任务配置入口在：

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/joint_pos_env_cfg.py`

这里定义了：

- 机械臂动作空间
- 夹爪开合动作
- 方块初始随机范围
- 方块目标位置范围
- 末端 frame
- 回放相机视角

当前任务的本质是：

- 回合开始时，方块在桌面上一个小范围内随机出现
- 策略控制机械臂和夹爪去接近、夹住并抬起方块
- 如果方块被抬起并接近目标位置，就得到更高奖励

## 5. Reward 设计

当前 reward 主要包括几项：

1. `reaching_object`
   鼓励末端接近方块

2. `lifting_object`
   鼓励把方块抬离桌面

3. `object_goal_tracking`
   鼓励将已抬起的方块移动到目标区域

4. `object_goal_tracking_fine_grained`
   在接近目标时提供更细的定位奖励

5. `action_rate`
   惩罚动作变化太快

6. `joint_vel`
   惩罚关节速度过大

这套 reward 的特点是：

- 没有显式“接触奖励”
- 没有显式“夹紧奖励”
- 抓取行为是靠物理接触和最终回报反向学出来的

## 6. PPO 配置

PPO 配置在：

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/agents/rsl_rl_ppo_cfg.py`

当前设置的核心参数包括：

- `num_steps_per_env = 24`
- `max_iterations = 3000`
- `learning_rate = 1e-4`
- `gamma = 0.98`
- `lam = 0.95`

网络结构是：

- actor: `[256, 128, 64]`
- critic: `[256, 128, 64]`
- activation: `elu`

## 7. 训练结果

本次训练结果目录在：

- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/`

关键结果包括：

- `model_2999.pt`
  最终 checkpoint

- `params/env.yaml`
  训练时实际使用的环境配置快照

- `params/agent.yaml`
  训练时实际使用的算法配置快照

- `videos/play/rl-video-step-0.mp4`
  回放录屏结果

## 8. 运行与验证

训练命令：

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Lift-Cube-XArm7-v0 --num_envs 1024 --headless
```

回放命令：

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Lift-Cube-XArm7-Play-v0 --num_envs 1 --checkpoint E:\IsaacLab-main\logs\rsl_rl\xarm7_lift\2026-04-08_12-42-49\model_2999.pt
```

录屏命令：

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Lift-Cube-XArm7-Play-v0 --headless --video --video_length 300 --num_envs 1 --checkpoint E:\IsaacLab-main\logs\rsl_rl\xarm7_lift\2026-04-08_12-42-49\model_2999.pt
```

## 9. 后续扩展

在抓方块任务基础上，后面已经开始往技能库方向扩展：

- `twist_cap` 作为扭瓶盖技能骨架
- `xarm7_skill_lib/registry.py` 作为技能注册入口

后续可以继续补：

- `screw` 技能任务
- IK 控制下的旋拧类 reward
- 更明确的接触/夹紧奖励
- 多技能统一切换入口
