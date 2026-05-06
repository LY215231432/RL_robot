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
