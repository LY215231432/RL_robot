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


CAP_SIDE_GRASP_HEIGHT = 0.018
CAP_APPROACH_CLEARANCE = 0.055
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

    robot: Articulation = env.scene[left_finger_cfg.name]
    left_pad_center_w = _body_point_w(robot, left_finger_cfg, LEFT_FINGER_PAD_CENTER_LOCAL)
    right_pad_center_w = _body_point_w(robot, right_finger_cfg, RIGHT_FINGER_PAD_CENTER_LOCAL)
    return 0.5 * (left_pad_center_w + right_pad_center_w)


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


def _stage_gate(score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert a smooth score into a gate that activates only past a threshold."""

    return torch.clamp((score - threshold) / max(1.0 - threshold, STAGE_GATE_EPS), min=0.0, max=1.0)


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
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    approach_target_w = _cap_approach_target_w(
        twist_object, cap_cfg, approach_clearance=approach_clearance, grasp_height=grasp_height
    )
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)

    xy_distance = torch.linalg.norm(approach_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    hover_height_error = torch.abs(approach_target_w[:, 2] - gripper_center_w[:, 2])
    above_cap_score = torch.clamp(
        (gripper_center_w[:, 2] - grasp_target_w[:, 2]) / max(approach_clearance, STAGE_GATE_EPS),
        0.0,
        1.0,
    )
    xy_alignment = 1 - torch.tanh(xy_distance / std)
    hover_alignment = 1 - torch.tanh(hover_height_error / (0.75 * std))
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )

    alignment = 0.60 * xy_alignment + 0.18 * hover_alignment + 0.12 * above_cap_score + 0.10 * vertical_alignment
    return alignment * (0.45 + 0.55 * gripper_open)


def descend_to_cap_grasp(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    close_joint_pos: float = 0.85,
    hover_gate_threshold: float = 0.55,
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
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)

    xy_distance = torch.linalg.norm(grasp_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    z_error = torch.abs(grasp_target_w[:, 2] - gripper_center_w[:, 2])

    xy_alignment = 1 - torch.tanh(xy_distance / xy_std)
    z_alignment = 1 - torch.tanh(z_error / z_std)
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

    alignment = 0.55 * xy_alignment + 0.25 * z_alignment + 0.12 * above_grasp + 0.08 * vertical_alignment
    return hover_gate * posture_gate * alignment * (0.35 + 0.65 * gripper_open)


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
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )

    xy_distance = torch.linalg.norm(approach_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    z_error = torch.abs(approach_target_w[:, 2] - gripper_center_w[:, 2])
    xy_alignment = 1 - torch.tanh(xy_distance / std)
    z_alignment = 1 - torch.tanh(z_error / (0.75 * std))
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    approach_alignment = 0.75 * xy_alignment + 0.15 * z_alignment + 0.10 * vertical_alignment
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
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )

    xy_distance = torch.linalg.norm(grasp_target_w[:, :2] - gripper_center_w[:, :2], dim=1)
    z_error = torch.abs(grasp_target_w[:, 2] - gripper_center_w[:, 2])

    xy_alignment = 1 - torch.tanh(xy_distance / xy_std)
    z_alignment = 1 - torch.tanh(z_error / z_std)
    vertical_alignment = _ee_vertical_alignment_score(
        env,
        axis_index=vertical_axis_index,
        target_sign=vertical_target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    alignment = 0.55 * xy_alignment + 0.25 * z_alignment + 0.20 * vertical_alignment
    return alignment * (0.45 + 0.55 * gripper_open)


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


def grasp_cap(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    descend_gate_threshold: float = 0.5,
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
        reference_joint_pos=reference_joint_pos,
        posture_std=posture_std,
        posture_gate_threshold=posture_gate_threshold,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
        robot_posture_cfg=robot_posture_cfg,
    )
    gripper_closed = _gripper_closed_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    left_contact = _contact_force_score(env, "contact_left_grasp", force_scale=contact_force_scale)
    right_contact = _contact_force_score(env, "contact_right_grasp", force_scale=contact_force_scale)
    bilateral_contact = torch.minimum(left_contact, right_contact)
    descend_gate = _stage_gate(descent_score, descend_gate_threshold)
    contact_bonus = 0.12 + 0.88 * bilateral_contact
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
