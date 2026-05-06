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
    contact_bonus = 0.35 + 0.65 * bilateral_contact
    return descend_gate * gripper_closed * contact_bonus
