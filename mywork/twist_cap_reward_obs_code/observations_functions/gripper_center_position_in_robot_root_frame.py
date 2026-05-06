def gripper_center_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the midpoint of the two finger pads in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return gripper_center_b
