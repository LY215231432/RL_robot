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
