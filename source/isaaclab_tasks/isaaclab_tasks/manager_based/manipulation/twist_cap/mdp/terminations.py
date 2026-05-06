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


def cap_rotation_reached(
    env: ManagerBasedRLEnv,
    goal_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
):
    """Terminate the episode once the cap has been rotated enough."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    cap_angle = twist_object.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    return cap_angle >= goal_angle

