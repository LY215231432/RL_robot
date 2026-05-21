# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


LEFT_FINGER_PAD_CENTER_LOCAL = (0.0, 0.011462, 0.083039)
RIGHT_FINGER_PAD_CENTER_LOCAL = (0.0, -0.011462, 0.083039)


def _single_body_position(body_pos_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return one body's world position for configs that may resolve to a list or slice."""

    selected_pos_w = body_pos_w[:, body_ids, :]
    if selected_pos_w.ndim == 3:
        return selected_pos_w[:, 0, :]
    return selected_pos_w


def _single_body_quaternion(body_quat_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return one body's world quaternion for configs that may resolve to a list or slice."""

    selected_quat_w = body_quat_w[:, body_ids, :]
    if selected_quat_w.ndim == 3:
        return selected_quat_w[:, 0, :]
    return selected_quat_w


def _single_joint_value(joint_tensor: torch.Tensor, joint_ids: list[int] | slice | int) -> torch.Tensor:
    """Return one joint value as a flat tensor."""

    selected = joint_tensor[:, joint_ids]
    if selected.ndim == 2:
        return selected[:, 0]
    return selected


def _body_point_w(
    asset: Articulation,
    body_cfg: SceneEntityCfg,
    local_point: tuple[float, float, float],
) -> torch.Tensor:
    """Transform a body-local point into the world frame."""

    body_pos_w = _single_body_position(asset.data.body_pos_w, body_cfg.body_ids)
    body_quat_w = _single_body_quaternion(asset.data.body_quat_w, body_cfg.body_ids)
    local_point_tensor = body_pos_w.new_tensor(local_point).expand_as(body_pos_w)
    return body_pos_w + quat_apply(body_quat_w, local_point_tensor)


def _gripper_pad_midpoint_w(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Approximate the midpoint between the two finger pads in world frame."""

    robot: Articulation = env.scene[left_finger_cfg.name]
    left_pad_center_w = _body_point_w(robot, left_finger_cfg, LEFT_FINGER_PAD_CENTER_LOCAL)
    right_pad_center_w = _body_point_w(robot, right_finger_cfg, RIGHT_FINGER_PAD_CENTER_LOCAL)
    return 0.5 * (left_pad_center_w + right_pad_center_w)


def _panel_target_point_w(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return a task target point on the switch lever or slider knob in world frame."""

    panel: Articulation = env.scene[body_cfg.name]
    target_w = _body_point_w(panel, body_cfg, target_local_offset)
    return target_w + target_w.new_tensor(target_world_offset).unsqueeze(0)


def panel_target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Return the task target point in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _panel_target_point_w(
        env,
        body_cfg=body_cfg,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
    )
    target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_w)
    return target_b


def gripper_center_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the midpoint of the two finger pads in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return gripper_center_b


def gripper_center_to_panel_target(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the target-point error vector relative to the gripper-pad midpoint in robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _panel_target_point_w(
        env,
        body_cfg=body_cfg,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
    )
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return target_b - gripper_center_b


def panel_joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Return the current switch or slider joint position."""

    panel: Articulation = env.scene[asset_cfg.name]
    return panel.data.joint_pos[:, asset_cfg.joint_ids]


def panel_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Return the current switch or slider joint velocity."""

    panel: Articulation = env.scene[asset_cfg.name]
    return panel.data.joint_vel[:, asset_cfg.joint_ids]


def panel_joint_error(
    env: ManagerBasedRLEnv,
    goal_position: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Return the remaining joint motion needed to reach the goal."""

    panel: Articulation = env.scene[asset_cfg.name]
    return goal_position - panel.data.joint_pos[:, asset_cfg.joint_ids]

