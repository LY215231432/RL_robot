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


CAP_SIDE_GRASP_HEIGHT = 0.033
CAP_APPROACH_CLEARANCE = 0.055
LEFT_FINGER_PAD_CENTER_LOCAL = (0.0, 0.011462, 0.083039)
RIGHT_FINGER_PAD_CENTER_LOCAL = (0.0, -0.011462, 0.083039)


def _single_body_position(body_pos_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world position for configs that may resolve to a list or slice."""

    selected_pos_w = body_pos_w[:, body_ids, :]
    if selected_pos_w.ndim == 3:
        return selected_pos_w[:, 0, :]
    return selected_pos_w


def _single_body_quaternion(body_quat_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world quaternion for configs that may resolve to a list or slice."""

    selected_quat_w = body_quat_w[:, body_ids, :]
    if selected_quat_w.ndim == 3:
        return selected_quat_w[:, 0, :]
    return selected_quat_w


def _body_point_w(
    asset: Articulation,
    body_cfg: SceneEntityCfg,
    local_point: tuple[float, float, float],
) -> torch.Tensor:
    """Return a local point on a body transformed into the world frame."""

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

    left_pad_center_w, right_pad_center_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    return 0.5 * (left_pad_center_w + right_pad_center_w)


def _finger_pad_positions_w(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the approximate left and right finger-pad centers in world frame."""

    robot: Articulation = env.scene[left_finger_cfg.name]
    left_pad_center_w = _body_point_w(robot, left_finger_cfg, LEFT_FINGER_PAD_CENTER_LOCAL)
    right_pad_center_w = _body_point_w(robot, right_finger_cfg, RIGHT_FINGER_PAD_CENTER_LOCAL)
    return left_pad_center_w, right_pad_center_w


def _cap_grasp_target_w(
    twist_object: Articulation,
    cap_cfg: SceneEntityCfg,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
) -> torch.Tensor:
    """Return the desired grasp target around the cap center in world frame."""

    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    grasp_offset = torch.zeros_like(cap_pos_w)
    grasp_offset[:, 2] = grasp_height
    return cap_pos_w + grasp_offset


def _cap_approach_target_w(
    twist_object: Articulation,
    cap_cfg: SceneEntityCfg,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
) -> torch.Tensor:
    """Return the approach target above the cap in world frame."""

    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    approach_offset = torch.zeros_like(grasp_target_w)
    approach_offset[:, 2] = approach_clearance
    return grasp_target_w + approach_offset


def cap_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
) -> torch.Tensor:
    """Return the cap position in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    cap_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cap_pos_w)
    return cap_pos_b


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


def gripper_center_to_cap(
    env: ManagerBasedRLEnv,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the cap grasp-target error vector relative to the finger-pad midpoint in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    cap_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cap_target_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return cap_pos_b - gripper_center_b


def finger_pair_to_cap(
    env: ManagerBasedRLEnv,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return cap grasp-target error vectors for the left and right finger pads."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    cap_target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cap_target_w)
    left_pad_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_pad_w)
    right_pad_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_pad_w)
    return torch.cat((cap_target_b - left_pad_b, cap_target_b - right_pad_b), dim=1)


def finger_pair_to_approach_target(
    env: ManagerBasedRLEnv,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return approach-target error vectors for the left and right finger pads."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    approach_target_w = _cap_approach_target_w(
        twist_object,
        cap_cfg,
        approach_clearance=approach_clearance,
        grasp_height=grasp_height,
    )
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    approach_target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, approach_target_w)
    left_pad_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_pad_w)
    right_pad_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_pad_w)
    return torch.cat((approach_target_b - left_pad_b, approach_target_b - right_pad_b), dim=1)


def gripper_center_to_approach_target(
    env: ManagerBasedRLEnv,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the approach-target error vector relative to the finger-pad midpoint in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    approach_target_w = _cap_approach_target_w(
        twist_object,
        cap_cfg,
        approach_clearance=approach_clearance,
        grasp_height=grasp_height,
    )
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    approach_target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, approach_target_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return approach_target_b - gripper_center_b


def cap_joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
) -> torch.Tensor:
    """Return the current cap joint angle."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    return twist_object.data.joint_pos[:, asset_cfg.joint_ids]


def cap_angle_error(
    env: ManagerBasedRLEnv,
    goal_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
) -> torch.Tensor:
    """Return the remaining rotation needed to reach the target cap angle."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    return goal_angle - twist_object.data.joint_pos[:, asset_cfg.joint_ids]
