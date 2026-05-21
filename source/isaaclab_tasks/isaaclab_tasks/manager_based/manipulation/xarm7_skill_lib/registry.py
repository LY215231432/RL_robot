# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    """Simple task registry entry for xArm7 manipulation skills."""

    name: str
    train_task: str
    play_task: str
    description: str
    default_checkpoint: str | None = None


SKILL_REGISTRY: dict[str, SkillSpec] = {
    "lift_cube": SkillSpec(
        name="lift_cube",
        train_task="Isaac-Lift-Cube-XArm7-v0",
        play_task="Isaac-Lift-Cube-XArm7-Play-v0",
        description="Pick up a cube and move it to a target region.",
        default_checkpoint="logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/model_2999.pt",
    ),
    "twist_cap": SkillSpec(
        name="twist_cap",
        train_task="Isaac-Twist-Cap-XArm7-v0",
        play_task="Isaac-Twist-Cap-XArm7-Play-v0",
        description="Reach the bottle cap, close the gripper, and rotate it along the cap axis.",
    ),
    "switch_panel": SkillSpec(
        name="switch_panel",
        train_task="Isaac-Switch-Panel-XArm7-v0",
        play_task="Isaac-Switch-Panel-XArm7-Play-v0",
        description="Use one side of the two-finger gripper to flick a revolute switch lever.",
    ),
    "slider_panel": SkillSpec(
        name="slider_panel",
        train_task="Isaac-Slider-Panel-XArm7-v0",
        play_task="Isaac-Slider-Panel-XArm7-Play-v0",
        description="Use the two-finger gripper side surface to push a prismatic slider to its goal zone.",
    ),
}
