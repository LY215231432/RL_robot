# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def panel_joint_reached(
    env: ManagerBasedRLEnv,
    goal_position: float,
    start_position: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
):
    """Terminate once the switch or slider joint reaches the goal side."""

    panel: Articulation = env.scene[asset_cfg.name]
    joint_pos = panel.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    if goal_position >= start_position:
        return joint_pos >= goal_position
    return joint_pos <= goal_position

