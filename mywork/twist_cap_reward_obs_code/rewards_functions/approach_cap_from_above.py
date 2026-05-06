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
