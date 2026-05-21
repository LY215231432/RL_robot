# xArm7 Skill Library Training

## 1. Purpose

This note explains how the xArm7 skill library is trained in IsaacLab after the project was moved to `E:\IsaacLab-main`.

The current skill registry contains:

- `lift_cube`: pick up a cube and move it to a target region
- `twist_cap`: reach a bottle cap, close the gripper, and rotate the cap

## 2. Core Files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/xarm7_skill_lib/registry.py`
  Role: central registry that maps skill names to train/play task IDs and optional default checkpoints.

- `scripts/reinforcement_learning/rsl_rl/train_xarm7_skill.py`
  Role: thin launcher that trains a skill by registry name instead of manually typing the task ID.

- `scripts/reinforcement_learning/rsl_rl/play_xarm7_skill.py`
  Role: thin launcher that plays a skill by registry name and injects the default checkpoint when available.

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/twist_cap/`
  Role: first new skill task package for bottle-cap twisting.

## 3. Training Flow

The skill-library training flow is:

`skill name -> registry -> task ID -> IsaacLab env cfg -> PPO cfg -> logs/checkpoints`

For example:

- `twist_cap`
- `Isaac-Twist-Cap-XArm7-v0`
- `isaaclab_tasks.manager_based.manipulation.twist_cap.config.xarm7.ik_rel_env_cfg:XArm7TwistCapEnvCfg`
- `isaaclab_tasks.manager_based.manipulation.twist_cap.config.xarm7.agents.rsl_rl_ppo_cfg:XArm7TwistCapPPORunnerCfg`

## 4. Common Commands

List all registered skills:

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --list
```

Train the existing cube-lift skill:

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill lift_cube --num_envs 1024 --headless
```

Train the twist-cap skill:

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill twist_cap --num_envs 256 --headless
```

Play the lift skill using the registry default checkpoint:

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play_xarm7_skill.py --skill lift_cube --num_envs 1
```

Record a video for the lift skill:

```powershell
cd /d E:\IsaacLab-main
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play_xarm7_skill.py --skill lift_cube --headless --video --video_length 300 --num_envs 1
```

## 5. Smoke-Test Result

On `2026-04-19`, a one-iteration smoke test for `twist_cap` completed successfully with:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_xarm7_skill.py --skill twist_cap --num_envs 32 --max_iterations 1 --headless
```

Result summary:

- log root: `logs/rsl_rl/xarm7_twist_cap/`
- experiment directory: `logs/rsl_rl/xarm7_twist_cap/2026-04-19_17-02-50/`
- training finished one PPO iteration successfully
- mean reward in the smoke run: about `0.46`

## 6. Notes

- `lift_cube` already has a default checkpoint in the registry, so playback can be launched with only `--skill lift_cube`.
- `twist_cap` does not yet have a default checkpoint in the registry because it has only passed the smoke test so far.
- The xArm7 URDF cache currently emits unresolved USD reference warnings for `link_eef`, `link_tcp`, and `world` visuals. These warnings did not block training in the smoke test, but the cache should be refreshed if cleaner logs are needed later.

