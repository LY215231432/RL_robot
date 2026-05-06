def gripper_center_to_cap(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the cap-center error vector relative to the finger-pad midpoint in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    cap_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cap_pos_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return cap_pos_b - gripper_center_b
