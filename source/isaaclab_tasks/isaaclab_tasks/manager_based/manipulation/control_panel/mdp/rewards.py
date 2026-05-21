# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import subtract_frame_transforms

from .observations import (
    LEFT_FINGER_PAD_CENTER_LOCAL,
    RIGHT_FINGER_PAD_CENTER_LOCAL,
    _body_point_w,
    _gripper_pad_midpoint_w,
    _panel_target_point_w,
    _single_joint_value,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


EPS = 1e-6


def _stage_gate(score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert a smooth score into a gate that activates after a threshold."""

    return torch.clamp((score - threshold) / max(1.0 - threshold, EPS), min=0.0, max=1.0)


def _tanh_position_score(distance: torch.Tensor, std: float) -> torch.Tensor:
    """Position score used by object-reaching rewards in Isaac Lab examples."""

    return 1.0 - torch.tanh(distance / max(std, EPS))


def _panel_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    panel: Articulation = env.scene[asset_cfg.name]
    return _single_joint_value(panel.data.joint_pos, asset_cfg.joint_ids)


def _panel_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    panel: Articulation = env.scene[asset_cfg.name]
    return _single_joint_value(panel.data.joint_vel, asset_cfg.joint_ids)


def _gripper_open_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Return how open the gripper is, normalized to [0, 1]."""

    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed_ratio = torch.clamp(joint_pos.mean(dim=1) / close_joint_pos, min=0.0, max=1.0)
    return 1.0 - closed_ratio


def _contact_force_score(env: ManagerBasedRLEnv, sensor_name: str, force_scale: float = 5.0) -> torch.Tensor:
    """Return a smooth [0, 1] score from filtered contact force magnitudes."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    if contact_sensor.data.force_matrix_w is not None:
        contact_force_w = torch.nan_to_num(contact_sensor.data.force_matrix_w, nan=0.0).reshape(env.num_envs, -1, 3)
    else:
        contact_force_w = torch.nan_to_num(contact_sensor.data.net_forces_w, nan=0.0).reshape(env.num_envs, -1, 3)

    contact_mag = torch.linalg.norm(contact_force_w, dim=-1).sum(dim=1)
    return 1.0 - torch.exp(-contact_mag / force_scale)


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


def _closest_finger_distance_to_panel_target(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_radius: float = 0.0,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Distance from the better-positioned finger pad to the panel target region."""

    target_w = _panel_target_point_w(
        env,
        body_cfg=body_cfg,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
    )
    left_pad_center_w, right_pad_center_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    left_distance = torch.linalg.norm(target_w - left_pad_center_w, dim=1)
    right_distance = torch.linalg.norm(target_w - right_pad_center_w, dim=1)
    closest_distance = torch.minimum(left_distance, right_distance)
    return torch.clamp(closest_distance - target_radius, min=0.0)


def gripper_to_panel_target_alignment(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: float = 0.08,
    close_joint_pos: float = 0.85,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Reward the gripper pad midpoint for reaching a contact or pre-contact target."""

    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    distance = torch.linalg.norm(target_w - gripper_center_w, dim=1)
    open_bonus = 0.35 + 0.65 * _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    return _tanh_position_score(distance, std) * open_bonus


def gripper_to_panel_approach_alignment(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float],
    std: float = 0.18,
    contact_std: float = 0.055,
    contact_gate_threshold: float = 0.55,
    target_radius: float = 0.0,
    progress_fade_start: float = 0.08,
    start_position: float | None = None,
    goal_position: float | None = None,
    close_joint_pos: float = 0.85,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Coarse approach reward that fades once a finger reaches the contact region or the panel moves."""

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
    approach_distance = torch.linalg.norm(target_w - gripper_center_w, dim=1)
    approach_score = _tanh_position_score(approach_distance, std)

    contact_distance = _closest_finger_distance_to_panel_target(
        env,
        target_local_offset=target_local_offset,
        target_radius=target_radius,
        body_cfg=body_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    contact_score = _tanh_position_score(contact_distance, contact_std)
    pre_contact_fade = 1.0 - _stage_gate(contact_score, contact_gate_threshold)

    if start_position is not None and goal_position is not None:
        progress = panel_joint_progress(env, start_position=start_position, goal_position=goal_position, asset_cfg=asset_cfg)
        progress_fade = torch.clamp(1.0 - progress / max(progress_fade_start, EPS), min=0.0, max=1.0)
    else:
        progress_fade = 1.0

    open_bonus = 0.35 + 0.65 * _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    return approach_score * pre_contact_fade * progress_fade * open_bonus


def finger_to_panel_target_alignment(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: float = 0.055,
    target_radius: float = 0.0,
    close_joint_pos: float = 0.85,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Reward the better side finger, not the gripper midpoint, for reaching the target contact region."""

    distance = _closest_finger_distance_to_panel_target(
        env,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
        target_radius=target_radius,
        body_cfg=body_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    open_bonus = 0.35 + 0.65 * _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    return _tanh_position_score(distance, std) * open_bonus


def panel_joint_progress(
    env: ManagerBasedRLEnv,
    start_position: float,
    goal_position: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Reward normalized progress of the switch or slider joint toward the goal."""

    joint_pos = _panel_joint_pos(env, asset_cfg)
    direction = 1.0 if goal_position >= start_position else -1.0
    span = abs(goal_position - start_position)
    return torch.clamp(direction * (joint_pos - start_position) / max(span, EPS), min=0.0, max=1.0)


def panel_joint_velocity_towards_goal(
    env: ManagerBasedRLEnv,
    start_position: float,
    goal_position: float,
    velocity_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Reward switch/slider velocity in the goal direction."""

    joint_vel = _panel_joint_vel(env, asset_cfg)
    direction = 1.0 if goal_position >= start_position else -1.0
    return torch.clamp(direction * joint_vel / max(velocity_scale, EPS), min=0.0, max=1.0)


def push_action_towards_goal(
    env: ManagerBasedRLEnv,
    push_axis: tuple[float, float, float],
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_std: float = 0.08,
    target_radius: float = 0.0,
    target_gate_threshold: float = 0.3,
    contact_gate_mix: float = 0.35,
    left_sensor_name: str = "contact_left_panel",
    right_sensor_name: str = "contact_right_panel",
    force_scale: float = 5.0,
    min_action_norm: float = 0.004,
    action_name: str = "arm_action",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Reward relative IK commands that push along the task's useful direction."""

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
    finger_distance = _closest_finger_distance_to_panel_target(
        env,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
        target_radius=target_radius,
        body_cfg=body_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    target_score = _tanh_position_score(finger_distance, target_std)
    target_gate = _stage_gate(target_score, target_gate_threshold)
    contact_score = finger_panel_contact(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        force_scale=force_scale,
    )
    contact_gate = contact_gate_mix + (1.0 - contact_gate_mix) * contact_score

    robot: Articulation = env.scene[robot_cfg.name]
    axis_w = gripper_center_w.new_tensor(push_axis).unsqueeze(0).expand_as(gripper_center_w)
    axis_b, _ = subtract_frame_transforms(
        torch.zeros_like(robot.data.root_pos_w),
        robot.data.root_quat_w,
        axis_w,
    )
    axis_b = axis_b / (torch.linalg.norm(axis_b, dim=1, keepdim=True) + EPS)

    arm_action = env.action_manager.get_term(action_name).processed_actions[:, :3]
    action_norm = torch.linalg.norm(arm_action, dim=1, keepdim=True)
    action_dir = arm_action / (action_norm + EPS)
    action_mag = torch.clamp(action_norm.squeeze(-1) / min_action_norm, min=0.0, max=1.0)
    direction_score = torch.clamp(torch.sum(action_dir * axis_b, dim=1), min=0.0, max=1.0)
    return target_gate * contact_gate * action_mag * direction_score


def finger_panel_contact(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "contact_left_panel",
    right_sensor_name: str = "contact_right_panel",
    force_scale: float = 5.0,
) -> torch.Tensor:
    """Reward either finger contacting the active switch or slider body."""

    left_contact = _contact_force_score(env, left_sensor_name, force_scale=force_scale)
    right_contact = _contact_force_score(env, right_sensor_name, force_scale=force_scale)
    return torch.clamp(left_contact + right_contact, min=0.0, max=1.0)


def finger_panel_contact_near_target(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_std: float = 0.07,
    target_radius: float = 0.0,
    target_gate_threshold: float = 0.25,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    left_sensor_name: str = "contact_left_panel",
    right_sensor_name: str = "contact_right_panel",
    force_scale: float = 5.0,
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Reward contact only when one finger is close enough to the intended panel target."""

    contact_score = finger_panel_contact(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        force_scale=force_scale,
    )
    finger_distance = _closest_finger_distance_to_panel_target(
        env,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
        target_radius=target_radius,
        body_cfg=body_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    target_gate = _stage_gate(_tanh_position_score(finger_distance, target_std), target_gate_threshold)
    return contact_score * target_gate


def near_target_without_contact_penalty(
    env: ManagerBasedRLEnv,
    target_local_offset: tuple[float, float, float],
    target_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_std: float = 0.07,
    target_radius: float = 0.0,
    target_gate_threshold: float = 0.45,
    progress_clearance: float = 0.05,
    start_position: float | None = None,
    goal_position: float | None = None,
    body_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
    left_sensor_name: str = "contact_left_panel",
    right_sensor_name: str = "contact_right_panel",
    force_scale: float = 5.0,
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Penalize hovering in the contact region without registering useful contact or progress."""

    finger_distance = _closest_finger_distance_to_panel_target(
        env,
        target_local_offset=target_local_offset,
        target_world_offset=target_world_offset,
        target_radius=target_radius,
        body_cfg=body_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    near_gate = _stage_gate(_tanh_position_score(finger_distance, target_std), target_gate_threshold)
    no_contact = 1.0 - finger_panel_contact(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        force_scale=force_scale,
    )
    if start_position is not None and goal_position is not None:
        progress = panel_joint_progress(env, start_position=start_position, goal_position=goal_position, asset_cfg=asset_cfg)
        no_progress = torch.clamp(1.0 - progress / max(progress_clearance, EPS), min=0.0, max=1.0)
    else:
        no_progress = 1.0
    return near_gate * no_contact * no_progress


def panel_joint_goal_bonus(
    env: ManagerBasedRLEnv,
    start_position: float,
    done_position: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("panel"),
) -> torch.Tensor:
    """Sparse bonus when the switch or slider reaches the demonstrable success threshold."""

    joint_pos = _panel_joint_pos(env, asset_cfg)
    direction = 1.0 if done_position >= start_position else -1.0
    return torch.where(direction * (joint_pos - done_position) >= 0.0, 1.0, 0.0)


def keep_gripper_open(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Reward keeping the parallel gripper open while using a finger side to push."""

    return _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)


def joint_posture_score(
    env: ManagerBasedRLEnv,
    reference_joint_pos: list[float],
    std: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the arm for staying near a comfortable overhead posture."""

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    reference = joint_pos.new_tensor(reference_joint_pos).unsqueeze(0)
    if joint_pos.shape[1] != reference.shape[1]:
        joint_pos = joint_pos[:, : reference.shape[1]]
    posture_error = torch.linalg.norm(joint_pos - reference, dim=1)
    return 1.0 - torch.tanh(posture_error / std)
