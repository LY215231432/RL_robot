# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import matrix_from_quat, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CAP_SIDE_GRASP_HEIGHT = 0.033
CAP_APPROACH_CLEARANCE = 0.055
CAP_RADIUS = 0.026
CAP_HEIGHT = 0.03
CAP_STRADDLE_CLEARANCE = 0.008
CAP_INNER_EXCLUSION_RADIUS = 0.017
STAGE_GATE_EPS = 1e-6
LEFT_FINGER_PAD_CENTER_LOCAL = (0.0, 0.011462, 0.083039)
RIGHT_FINGER_PAD_CENTER_LOCAL = (0.0, -0.011462, 0.083039)


def _single_body_position(body_pos_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world position for configs that may resolve to a list or slice."""

    selected_pos_w = body_pos_w[:, body_ids, :]
    if selected_pos_w.ndim == 3:
        return selected_pos_w[:, 0, :]
    return selected_pos_w


def _single_body_vector(body_vec_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world-frame vector for configs that may resolve to a list or slice."""

    selected_vec_w = body_vec_w[:, body_ids, :]
    if selected_vec_w.ndim == 3:
        return selected_vec_w[:, 0, :]
    return selected_vec_w


def _single_body_quaternion(body_quat_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world-frame quaternion for configs that may resolve to a list or slice."""

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


def _cap_grasp_target_w(
    twist_object: Articulation,
    cap_cfg: SceneEntityCfg,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
) -> torch.Tensor:
    """Return the desired TCP target around the cap side grasp center."""

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
    """Return the desired TCP target above the cap before descending."""

    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    approach_offset = torch.zeros_like(grasp_target_w)
    approach_offset[:, 2] = approach_clearance
    return grasp_target_w + approach_offset


def _ee_position_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the end-effector TCP position in world frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


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


def _ee_vertical_alignment_score(
    env: ManagerBasedRLEnv,
    axis_index: int = 2,
    target_sign: float = -1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return how well a chosen TCP axis aligns with the world vertical direction."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_rot_mat = matrix_from_quat(ee_quat_w)
    ee_axis = ee_rot_mat[..., axis_index]

    target_axis = torch.zeros_like(ee_axis)
    target_axis[:, 2] = target_sign
    alignment = torch.sum(ee_axis * target_axis, dim=1)
    return torch.clamp(alignment, min=0.0, max=1.0).square()


def _gripper_closed_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Return how closed the gripper is, normalized to [0, 1]."""

    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    return torch.clamp(joint_pos.mean(dim=1) / close_joint_pos, min=0.0, max=1.0)


def _gripper_open_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Return how open the gripper is, normalized to [0, 1]."""

    return 1.0 - _gripper_closed_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)


def _joint_posture_score(
    env: ManagerBasedRLEnv,
    reference_joint_pos: list[float],
    std: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a smooth [0, 1] score for keeping the arm near a reference posture."""

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    reference = joint_pos.new_tensor(reference_joint_pos).unsqueeze(0)
    if joint_pos.shape[1] != reference.shape[1]:
        joint_pos = joint_pos[:, : reference.shape[1]]
    posture_error = torch.linalg.norm(joint_pos - reference, dim=1)
    return 1.0 - torch.tanh(posture_error / std)


def _contact_force_score(env: ManagerBasedRLEnv, sensor_name: str, force_scale: float = 6.0) -> torch.Tensor:
    """Return a smooth [0, 1] score from filtered contact force magnitudes."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    if contact_sensor.data.force_matrix_w is not None:
        contact_force_w = torch.nan_to_num(contact_sensor.data.force_matrix_w, nan=0.0).reshape(env.num_envs, -1, 3)
    else:
        contact_force_w = torch.nan_to_num(contact_sensor.data.net_forces_w, nan=0.0).reshape(env.num_envs, -1, 3)

    contact_mag = torch.linalg.norm(contact_force_w, dim=-1).sum(dim=1)
    return 1.0 - torch.exp(-contact_mag / force_scale)


def _finger_contact_scores(
    env: ManagerBasedRLEnv,
    contact_force_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right cap-contact scores for the two finger pads."""

    left_contact = _contact_force_score(env, "contact_left_grasp", force_scale=contact_force_scale)
    right_contact = _contact_force_score(env, "contact_right_grasp", force_scale=contact_force_scale)
    return left_contact, right_contact


def _stage_gate(score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert a smooth score into a gate that activates only past a threshold."""

    return torch.clamp((score - threshold) / max(1.0 - threshold, STAGE_GATE_EPS), min=0.0, max=1.0)


def _position_distance_score(target_w: torch.Tensor, point_w: torch.Tensor, std: float) -> torch.Tensor:
    """Reward a point for reaching a target using the object-grasp tanh distance kernel."""

    distance = torch.linalg.norm(target_w - point_w, dim=1)
    return 1.0 - torch.tanh(distance / max(std, STAGE_GATE_EPS))


def _finger_pair_straddle_score(
    target_w: torch.Tensor,
    left_pad_w: torch.Tensor,
    right_pad_w: torch.Tensor,
    center_std: float,
    span_std: float,
    line_std: float,
    z_std: float,
    cap_radius: float = CAP_RADIUS,
    finger_clearance: float = CAP_STRADDLE_CLEARANCE,
    inner_exclusion_radius: float = CAP_INNER_EXCLUSION_RADIUS,
) -> torch.Tensor:
    """Reward the cap target being between the two finger pads, not under one finger."""

    center_w = 0.5 * (left_pad_w + right_pad_w)
    center_score = _position_distance_score(target_w, center_w, center_std)

    span_xy = left_pad_w[:, :2] - right_pad_w[:, :2]
    span_norm = torch.linalg.norm(span_xy, dim=1)
    span_dir = span_xy / (span_norm.unsqueeze(-1) + STAGE_GATE_EPS)
    target_delta_xy = target_w[:, :2] - center_w[:, :2]
    along_error_signed = torch.sum(target_delta_xy * span_dir, dim=1)
    along_error = torch.abs(along_error_signed)
    perp_error = torch.linalg.norm(target_delta_xy - along_error_signed.unsqueeze(-1) * span_dir, dim=1)

    target_half_span = cap_radius + finger_clearance
    half_span = 0.5 * span_norm
    span_score = 1.0 - torch.tanh(torch.abs(half_span - target_half_span) / max(span_std, STAGE_GATE_EPS))
    balance_score = 1.0 - torch.tanh(along_error / max(line_std, STAGE_GATE_EPS))
    line_score = 1.0 - torch.tanh(perp_error / max(line_std, STAGE_GATE_EPS))

    center_z_error = torch.abs(center_w[:, 2] - target_w[:, 2])
    finger_z_mismatch = torch.abs(left_pad_w[:, 2] - right_pad_w[:, 2])
    z_score = 0.65 * (1.0 - torch.tanh(center_z_error / max(z_std, STAGE_GATE_EPS)))
    z_score = z_score + 0.35 * (1.0 - torch.tanh(finger_z_mismatch / max(z_std, STAGE_GATE_EPS)))

    left_xy_distance = torch.linalg.norm(left_pad_w[:, :2] - target_w[:, :2], dim=1)
    right_xy_distance = torch.linalg.norm(right_pad_w[:, :2] - target_w[:, :2], dim=1)
    side_distance_error = 0.5 * (
        torch.abs(left_xy_distance - target_half_span) + torch.abs(right_xy_distance - target_half_span)
    )
    side_distance_score = 1.0 - torch.tanh(side_distance_error / max(span_std, STAGE_GATE_EPS))
    side_balance_score = 1.0 - torch.tanh(
        torch.abs(left_xy_distance - right_xy_distance) / max(line_std, STAGE_GATE_EPS)
    )
    min_finger_xy_distance = torch.minimum(left_xy_distance, right_xy_distance)
    outside_center_gate = torch.clamp(
        (min_finger_xy_distance - inner_exclusion_radius)
        / max(target_half_span - inner_exclusion_radius, STAGE_GATE_EPS),
        min=0.0,
        max=1.0,
    )

    pair_geometry = (
        0.25 * center_score
        + 0.14 * line_score
        + 0.18 * balance_score
        + 0.13 * span_score
        + 0.18 * side_distance_score
        + 0.12 * z_score
    )
    precision_gate = torch.minimum(
        torch.minimum(balance_score, line_score),
        torch.minimum(span_score, side_distance_score),
    )
    reward = pair_geometry
    reward = reward * (0.15 + 0.85 * outside_center_gate)
    reward = reward * (0.25 + 0.75 * precision_gate.square())
    reward = reward * (0.35 + 0.65 * side_balance_score)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def approach_action_towards_cap(
    env: ManagerBasedRLEnv,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    min_action_norm: float = 0.004,
    active_distance: float = 0.18,
    close_joint_pos: float = 0.85,
    xy_weight: float = 0.7,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Reward position-IK actions that move the finger-pad midpoint toward the cap approach target."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    approach_target_w = _cap_approach_target_w(
        twist_object,
        cap_cfg,
        approach_clearance=approach_clearance,
        grasp_height=grasp_height,
    )

    robot: Articulation = env.scene[robot_cfg.name]
    approach_target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, approach_target_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    desired_delta = approach_target_b - gripper_center_b
    arm_action = env.action_manager.get_term(action_name).processed_actions[:, :3]

    desired_xy = desired_delta.clone()
    desired_xy[:, 2] = 0.0
    action_xy = arm_action.clone()
    action_xy[:, 2] = 0.0

    desired_norm = torch.linalg.norm(desired_delta, dim=1, keepdim=True)
    action_norm = torch.linalg.norm(arm_action, dim=1, keepdim=True)
    desired_xy_norm = torch.linalg.norm(desired_xy, dim=1, keepdim=True)
    action_xy_norm = torch.linalg.norm(action_xy, dim=1, keepdim=True)

    direction_score = torch.sum(
        arm_action / (action_norm + STAGE_GATE_EPS) * desired_delta / (desired_norm + STAGE_GATE_EPS),
        dim=1,
    )
    xy_direction_score = torch.sum(
        action_xy / (action_xy_norm + STAGE_GATE_EPS) * desired_xy / (desired_xy_norm + STAGE_GATE_EPS),
        dim=1,
    )
    direction_score = torch.clamp(direction_score, min=0.0, max=1.0)
    xy_direction_score = torch.clamp(xy_direction_score, min=0.0, max=1.0)

    action_mag = torch.clamp(action_norm.squeeze(-1) / min_action_norm, min=0.0, max=1.0)
    distance_gate = torch.clamp(desired_norm.squeeze(-1) / active_distance, min=0.0, max=1.0)
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    direction_mix = xy_weight * xy_direction_score + (1.0 - xy_weight) * direction_score
    reward = direction_mix * action_mag * distance_gate * (0.5 + 0.5 * gripper_open)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def approach_cap_from_above(
    env: ManagerBasedRLEnv,
    std: float,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    vertical_axis_index: int = 2,
    vertical_target_sign: float = -1.0,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Reward the finger-pad midpoint for first moving to a point above the bottle cap."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_w = 0.5 * (left_pad_w + right_pad_w)
    approach_target_w = _cap_approach_target_w(
        twist_object, cap_cfg, approach_clearance=approach_clearance, grasp_height=grasp_height
    )
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)

    above_cap_score = torch.clamp(
        (gripper_center_w[:, 2] - grasp_target_w[:, 2]) / max(approach_clearance, STAGE_GATE_EPS),
        0.0,
        1.0,
    )
    position_alignment = _position_distance_score(approach_target_w, gripper_center_w, std)
    straddle_alignment = _finger_pair_straddle_score(
        approach_target_w,
        left_pad_w,
        right_pad_w,
        center_std=std,
        span_std=0.028,
        line_std=max(0.5 * std, 0.025),
        z_std=0.04,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )

    alignment = (
        0.12 * position_alignment
        + 0.66 * straddle_alignment
        + 0.12 * above_cap_score
        + 0.10 * vertical_alignment
    )
    return alignment * (0.45 + 0.55 * gripper_open)


def descend_to_cap_grasp(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    hover_gate_threshold: float = 0.45,
    straddle_gate_threshold: float = 0.35,
    reference_joint_pos: list[float] | None = None,
    posture_std: float = 0.45,
    posture_gate_threshold: float = 0.35,
    vertical_axis_index: int = 2,
    vertical_target_sign: float = -1.0,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    robot_posture_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-7]"]),
) -> torch.Tensor:
    """Reward descending straight down after horizontal alignment and a vertical-press posture."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_w = 0.5 * (left_pad_w + right_pad_w)
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)

    position_std = max(xy_std, z_std)
    position_alignment = _position_distance_score(grasp_target_w, gripper_center_w, position_std)
    straddle_alignment = _finger_pair_straddle_score(
        grasp_target_w,
        left_pad_w,
        right_pad_w,
        center_std=position_std,
        span_std=0.018,
        line_std=max(xy_std, 0.012),
        z_std=max(z_std, 0.012),
    )
    hover_score = approach_cap_from_above(
        env,
        std=max(1.35 * xy_std, z_std),
        approach_clearance=CAP_APPROACH_CLEARANCE,
        grasp_height=grasp_height,
        close_joint_pos=close_joint_pos,
        vertical_axis_index=vertical_axis_index,
        vertical_target_sign=vertical_target_sign,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    hover_gate = 0.15 + 0.85 * _stage_gate(hover_score, hover_gate_threshold)
    above_grasp = torch.clamp((gripper_center_w[:, 2] - grasp_target_w[:, 2] + 0.01) / 0.03, min=0.0, max=1.0)
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    posture_gate = 1.0
    if reference_joint_pos is not None:
        posture_score = _joint_posture_score(
            env,
            reference_joint_pos=reference_joint_pos,
            std=posture_std,
            robot_cfg=robot_posture_cfg,
        )
        posture_gate = 0.2 + 0.8 * _stage_gate(posture_score, posture_gate_threshold)

    alignment = (
        0.12 * position_alignment
        + 0.76 * straddle_alignment
        + 0.06 * above_grasp
        + 0.06 * vertical_alignment
    )
    straddle_gate = _stage_gate(straddle_alignment, straddle_gate_threshold)
    return straddle_gate * hover_gate * posture_gate * alignment * (0.75 + 0.25 * gripper_open)


def tcp_vertical_alignment(
    env: ManagerBasedRLEnv,
    axis_index: int = 2,
    target_sign: float = -1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the TCP for keeping the selected axis vertical to the ground."""

    return _ee_vertical_alignment_score(
        env,
        axis_index=axis_index,
        target_sign=target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )


def vertical_press_posture(
    env: ManagerBasedRLEnv,
    reference_joint_pos: list[float],
    std: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-7]"]),
) -> torch.Tensor:
    """Reward the arm for staying in a posture that supports top-down cap pressing."""

    return _joint_posture_score(
        env,
        reference_joint_pos=reference_joint_pos,
        std=std,
        robot_cfg=robot_cfg,
    )


def gripper_center_to_approach_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    vertical_axis_index: int = 2,
    vertical_target_sign: float = -1.0,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
) -> torch.Tensor:
    """Reward a coarse move of the finger-pad midpoint towards the pre-grasp point above the cap."""

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
    gripper_center_w = 0.5 * (left_pad_w + right_pad_w)

    position_alignment = _position_distance_score(approach_target_w, gripper_center_w, std)
    straddle_alignment = _finger_pair_straddle_score(
        approach_target_w,
        left_pad_w,
        right_pad_w,
        center_std=std,
        span_std=0.03,
        line_std=max(0.5 * std, 0.03),
        z_std=0.045,
    )
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    approach_alignment = 0.20 * position_alignment + 0.70 * straddle_alignment + 0.10 * vertical_alignment
    return approach_alignment * (0.5 + 0.5 * gripper_open)


def gripper_center_to_cap_alignment(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    vertical_axis_index: int = 2,
    vertical_target_sign: float = -1.0,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
) -> torch.Tensor:
    """Reward the midpoint of the two finger pads for centering on the cap grasp target."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_w = 0.5 * (left_pad_w + right_pad_w)

    position_std = max(xy_std, z_std)
    position_alignment = _position_distance_score(grasp_target_w, gripper_center_w, position_std)
    straddle_alignment = _finger_pair_straddle_score(
        grasp_target_w,
        left_pad_w,
        right_pad_w,
        center_std=position_std,
        span_std=0.018,
        line_std=max(xy_std, 0.012),
        z_std=max(z_std, 0.012),
    )
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    alignment = 0.12 * position_alignment + 0.76 * straddle_alignment + 0.12 * vertical_alignment
    return alignment * (0.85 + 0.15 * gripper_open)


def finger_pair_side_wall_alignment(
    env: ManagerBasedRLEnv,
    xy_std: float = 0.035,
    z_std: float = 0.030,
    span_std: float = 0.026,
    line_std: float = 0.024,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    vertical_axis_index: int = 2,
    vertical_target_sign: float = -1.0,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
) -> torch.Tensor:
    """Dense reward for placing the two finger pads on opposite sides of the cap side wall."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    center_w = 0.5 * (left_pad_w + right_pad_w)

    center_score = _position_distance_score(grasp_target_w, center_w, max(xy_std, z_std))
    straddle_score = _finger_pair_straddle_score(
        grasp_target_w,
        left_pad_w,
        right_pad_w,
        center_std=max(xy_std, z_std),
        span_std=span_std,
        line_std=line_std,
        z_std=z_std,
    )
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    return (0.18 * center_score + 0.72 * straddle_score + 0.10 * vertical_alignment) * (0.75 + 0.25 * gripper_open)


def close_on_cap_straddle(
    env: ManagerBasedRLEnv,
    xy_std: float = 0.035,
    z_std: float = 0.030,
    straddle_gate_threshold: float = 0.30,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
) -> torch.Tensor:
    """Reward gripper closure only after the fingers roughly straddle the cap side wall."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    straddle_score = _finger_pair_straddle_score(
        grasp_target_w,
        left_pad_w,
        right_pad_w,
        center_std=max(xy_std, z_std),
        span_std=0.026,
        line_std=0.024,
        z_std=z_std,
    )
    gripper_closed = _gripper_closed_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    top_clear = 1.0 - cap_top_contact_penalty(
        env,
        cap_cfg=cap_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    straddle_gate = _stage_gate(straddle_score, straddle_gate_threshold)
    return straddle_gate * gripper_closed * (0.45 + 0.55 * straddle_score) * top_clear


def side_approach_penalty(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    safe_xy_radius: float,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Penalize hovering near the cap sidewall instead of aligning above the cap center."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )

    xy_distance = torch.linalg.norm(grasp_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    z_error = torch.abs(grasp_target_w[:, 2] - gripper_center_w[:, 2])

    side_height_band = 1 - torch.tanh(z_error / z_std)
    xy_misalignment = torch.clamp((xy_distance - safe_xy_radius) / max(xy_std, STAGE_GATE_EPS), min=0.0, max=1.0)
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    return side_height_band * xy_misalignment * (0.4 + 0.6 * gripper_open)


def finger_over_cap_penalty(
    env: ManagerBasedRLEnv,
    inner_radius: float = CAP_INNER_EXCLUSION_RADIUS,
    z_std: float = 0.035,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Penalize placing either finger pad directly over the cap center line."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )

    left_xy_distance = torch.linalg.norm(left_pad_w[:, :2] - grasp_target_w[:, :2], dim=1)
    right_xy_distance = torch.linalg.norm(right_pad_w[:, :2] - grasp_target_w[:, :2], dim=1)
    left_is_closer = left_xy_distance <= right_xy_distance
    closest_xy_distance = torch.where(left_is_closer, left_xy_distance, right_xy_distance)
    closest_z_error = torch.where(
        left_is_closer,
        torch.abs(left_pad_w[:, 2] - grasp_target_w[:, 2]),
        torch.abs(right_pad_w[:, 2] - grasp_target_w[:, 2]),
    )

    inside_cap_axis = torch.clamp((inner_radius - closest_xy_distance) / max(inner_radius, STAGE_GATE_EPS), 0.0, 1.0)
    height_band = 1.0 - torch.tanh(closest_z_error / max(z_std, STAGE_GATE_EPS))
    return inside_cap_axis * height_band


def cap_top_contact_penalty(
    env: ManagerBasedRLEnv,
    top_radius: float = 0.024,
    top_height: float = CAP_HEIGHT,
    side_grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    z_std: float = 0.010,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Penalize finger pads hovering or pressing on the cap top disk instead of its side wall."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )

    cap_top_z = cap_pos_w[:, 2] + top_height
    side_target_z = cap_pos_w[:, 2] + side_grasp_height
    top_band_height = max(top_height - side_grasp_height, STAGE_GATE_EPS)

    def _single_pad_top_score(pad_w: torch.Tensor) -> torch.Tensor:
        xy_distance = torch.linalg.norm(pad_w[:, :2] - cap_pos_w[:, :2], dim=1)
        inside_top_disk = torch.clamp((top_radius - xy_distance) / max(top_radius, STAGE_GATE_EPS), 0.0, 1.0)
        above_side_grasp = torch.clamp((pad_w[:, 2] - side_target_z) / top_band_height, 0.0, 1.0)
        near_cap_top = 1.0 - torch.tanh(torch.abs(pad_w[:, 2] - cap_top_z) / max(z_std, STAGE_GATE_EPS))
        return inside_top_disk * (0.35 * above_side_grasp + 0.65 * near_cap_top)

    left_top_score = _single_pad_top_score(left_pad_w)
    right_top_score = _single_pad_top_score(right_pad_w)
    return torch.maximum(left_top_score, right_top_score)


def single_finger_contact_penalty(
    env: ManagerBasedRLEnv,
    contact_force_scale: float = 4.0,
    xy_std: float = 0.035,
    z_std: float = 0.025,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Penalize touching the cap with only one finger near the intended grasp pose."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    left_pad_w, right_pad_w = _finger_pad_positions_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_w = 0.5 * (left_pad_w + right_pad_w)

    xy_distance = torch.linalg.norm(grasp_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    z_error = torch.abs(grasp_target_w[:, 2] - gripper_center_w[:, 2])
    near_grasp_gate = (1.0 - torch.tanh(xy_distance / max(xy_std, STAGE_GATE_EPS)))
    near_grasp_gate = near_grasp_gate * (1.0 - torch.tanh(z_error / max(z_std, STAGE_GATE_EPS)))

    left_contact, right_contact = _finger_contact_scores(env, contact_force_scale=contact_force_scale)
    one_sided_contact = torch.abs(left_contact - right_contact)
    any_contact = torch.maximum(left_contact, right_contact)
    return near_grasp_gate * any_contact * one_sided_contact


def grasp_cap(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    descend_gate_threshold: float = 0.25,
    straddle_gate_threshold: float = 0.35,
    reference_joint_pos: list[float] | None = None,
    posture_std: float = 0.45,
    posture_gate_threshold: float = 0.35,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    robot_posture_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-7]"]),
) -> torch.Tensor:
    """Reward a stable bilateral grasp on the cap using contact and gripper closure."""

    descent_score = descend_to_cap_grasp(
        env,
        xy_std=xy_std,
        z_std=z_std,
        grasp_height=grasp_height,
        close_joint_pos=close_joint_pos,
        straddle_gate_threshold=straddle_gate_threshold,
        reference_joint_pos=reference_joint_pos,
        posture_std=posture_std,
        posture_gate_threshold=posture_gate_threshold,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
        robot_posture_cfg=robot_posture_cfg,
    )
    gripper_closed = _gripper_closed_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    left_contact, right_contact = _finger_contact_scores(env, contact_force_scale=contact_force_scale)
    bilateral_contact = torch.minimum(left_contact, right_contact)
    contact_balance = 1.0 - torch.tanh(
        torch.abs(left_contact - right_contact) / max(0.25, STAGE_GATE_EPS)
    )
    descend_gate = _stage_gate(descent_score, descend_gate_threshold)
    contact_bonus = bilateral_contact * (0.35 + 0.65 * contact_balance)
    return descend_gate * gripper_closed * contact_bonus


def cap_rotation_progress(
    env: ManagerBasedRLEnv,
    goal_angle: float,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    grasp_gate_threshold: float = 0.6,
) -> torch.Tensor:
    """Reward rotating the cap, gated by having first established a stable grasp."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    cap_angle = twist_object.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    grasp_score = grasp_cap(
        env,
        xy_std=xy_std,
        z_std=z_std,
        contact_force_scale=contact_force_scale,
        close_joint_pos=close_joint_pos,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
    )
    progress = torch.clamp(cap_angle / goal_angle, min=0.0, max=1.0)
    return progress * _stage_gate(grasp_score, grasp_gate_threshold)


def cap_positive_rotation_velocity(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    grasp_gate_threshold: float = 0.6,
) -> torch.Tensor:
    """Provide a small reward for rotating the cap in the unscrewing direction after grasping."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    cap_vel = twist_object.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
    grasp_score = grasp_cap(
        env,
        xy_std=xy_std,
        z_std=z_std,
        contact_force_scale=contact_force_scale,
        close_joint_pos=close_joint_pos,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
    )
    return torch.clamp(cap_vel, min=0.0, max=5.0) * _stage_gate(grasp_score, grasp_gate_threshold)


def twist_tcp_angular_velocity(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    velocity_scale: float = 2.5,
    max_velocity: float = 6.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_tcp"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    grasp_gate_threshold: float = 0.6,
) -> torch.Tensor:
    """Reward the TCP for actively twisting about the bottle-cap axis once the cap is grasped."""

    robot: Articulation = env.scene[robot_cfg.name]
    tcp_ang_vel_w = _single_body_vector(robot.data.body_ang_vel_w, robot_cfg.body_ids)
    positive_twist_speed = torch.clamp(tcp_ang_vel_w[:, 2], min=0.0, max=max_velocity)

    grasp_score = grasp_cap(
        env,
        xy_std=xy_std,
        z_std=z_std,
        contact_force_scale=contact_force_scale,
        close_joint_pos=close_joint_pos,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
    )
    grasp_gate = _stage_gate(grasp_score, grasp_gate_threshold)
    return (1.0 - torch.exp(-positive_twist_speed / velocity_scale)) * grasp_gate
