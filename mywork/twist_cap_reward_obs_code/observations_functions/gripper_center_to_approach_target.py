def gripper_center_to_approach_target(
    env: ManagerBasedRLEnv,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Return the approach-target error vector relative to the finger-pad midpoint in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
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
    approach_target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, approach_target_w)
    gripper_center_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, gripper_center_w)
    return approach_target_b - gripper_center_b
