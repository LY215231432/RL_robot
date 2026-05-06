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
