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
