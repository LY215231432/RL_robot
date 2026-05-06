def cap_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
) -> torch.Tensor:
    """Return the cap position in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    twist_object: Articulation = env.scene[cap_cfg.name]
    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    cap_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cap_pos_w)
    return cap_pos_b
