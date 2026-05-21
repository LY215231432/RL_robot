# RL Robot xArm7 Isaac Lab Project

这个仓库是基于 NVIDIA Isaac Lab 的 xArm7 具身强化学习项目，当前重点是机械臂抓取和旋拧瓶盖任务。项目保留了 Isaac Lab 的运行框架，同时加入了 xArm7 技能注册、训练脚本、twist-cap 任务环境、reward/observation 设计和实验说明。

## 项目内容

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/`
  xArm7 旋拧瓶盖任务，包括场景、动作、观测、奖励、终止条件和 PPO 配置。
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/xarm7_skill_lib/registry.py`
  xArm7 技能注册表，当前包含 `lift_cube`、`twist_cap`、`switch_panel` 和 `slider_panel`。
- `scripts/reinforcement_learning/rsl_rl/train_xarm7_skill.py`
  按技能名启动训练的封装脚本。
- `scripts/reinforcement_learning/rsl_rl/play_xarm7_skill.py`
  按技能名加载 checkpoint 并测试策略的封装脚本。
- `mywork/isaaclab_training/`
  项目说明、组会材料、训练记录和批量训练脚本。

训练日志、checkpoint、视频、缓存、虚拟环境和压缩归档不会提交到仓库。重新训练后，它们会生成在 `logs/`、`outputs/` 或本地缓存目录中。

## 环境准备

推荐使用带 NVIDIA GPU 的机器，并安装与本仓库 Isaac Lab 版本匹配的 Isaac Sim。本仓库当前 Isaac Lab 版本为 `2.3.2`。

Windows PowerShell:

```powershell
cd E:\IsaacLab-main
.\isaaclab.bat -c env_isaaclab
conda activate env_isaaclab
.\isaaclab.bat -i rsl_rl
```

Linux:

```bash
cd /path/to/IsaacLab-main
./isaaclab.sh -c env_isaaclab
conda activate env_isaaclab
./isaaclab.sh -i rsl_rl
```

如果已经有可用的 Isaac Lab / Isaac Sim 环境，可以直接激活对应环境后执行安装扩展：

```powershell
conda activate env_isaaclab
.\isaaclab.bat -i rsl_rl
```

## 查看可训练技能

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --list
```

常用技能名：

- `lift_cube`: xArm7 抓取方块并移动到目标区域。
- `twist_cap`: xArm7 对准瓶盖、下压夹取并沿瓶盖轴向旋转。
- `switch_panel`: 用夹爪侧面拨动旋转开关。
- `slider_panel`: 用夹爪侧面推动滑块到目标区域。

## 训练

先做一次最小烟雾测试，确认环境、任务注册和 PPO 配置都能正常跑通：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill twist_cap --num_envs 32 --max_iterations 1 --headless
```

正式训练 twist-cap 技能：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill twist_cap --num_envs 256 --headless
```

指定训练轮数：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill twist_cap --num_envs 256 --max_iterations 4000 --headless
```

也可以直接使用 Isaac Lab 原始任务 ID：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Twist-Cap-XArm7-v0 --num_envs 256 --headless
```

训练结果默认保存到：

```text
logs/rsl_rl/xarm7_twist_cap/<run_time>/
```

其中常用文件包括：

- `model_*.pt`: 训练 checkpoint。
- `params/env.yaml`: 本次训练使用的环境配置快照。
- `params/agent.yaml`: 本次训练使用的 PPO 配置快照。
- `exported/policy.pt` 和 `exported/policy.onnx`: 运行 `play.py` 后导出的策略模型。

## 测试和回放

`twist_cap` 没有随仓库提交默认 checkpoint。训练完成后，用实际生成的 `model_*.pt` 路径测试：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play_xarm7_skill.py --skill twist_cap --num_envs 1 --checkpoint logs\rsl_rl\xarm7_twist_cap\<run_time>\model_<iter>.pt
```

录制测试视频：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play_xarm7_skill.py --skill twist_cap --num_envs 1 --headless --video --video_length 300 --checkpoint logs\rsl_rl\xarm7_twist_cap\<run_time>\model_<iter>.pt
```

也可以直接使用 Play 任务 ID：

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Twist-Cap-XArm7-Play-v0 --num_envs 1 --checkpoint logs\rsl_rl\xarm7_twist_cap\<run_time>\model_<iter>.pt
```

视频会保存在对应训练目录下：

```text
logs/rsl_rl/xarm7_twist_cap/<run_time>/videos/play/
```

## 批量训练

项目中还提供了 twist-cap 批量实验脚本：

```bash
NUM_ENVS=256 MAX_ITERATIONS=4000 ./mywork/isaaclab_training/batch_train_twist_cap.sh
```

仅打印将要执行的命令：

```bash
DRY_RUN=1 ./mywork/isaaclab_training/batch_train_twist_cap.sh
```

## 关键实现

twist-cap 的 observation 主要包含瓶盖位置、夹爪中心位置、夹爪中心到瓶盖和预抓取点的误差、瓶盖关节角度和上一时刻动作。reward 采用分阶段设计：先引导夹爪移动到瓶盖上方，再从上方下压对准，最后通过夹爪闭合和左右接触判断稳定抓取，抓稳之后再逐步启用旋拧相关奖励。

重点文件：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/observations.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/mdp/rewards.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/twist_cap_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/config/xarm7/agents/rsl_rl_ppo_cfg.py
```

更详细的 reward 和 observation 说明见：

```text
mywork/TWIST_CAP_REWARD_OBS_GUIDE.md
mywork/isaaclab_training/skill_library_training.md
```
