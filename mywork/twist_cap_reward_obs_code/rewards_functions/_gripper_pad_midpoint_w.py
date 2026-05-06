def _gripper_pad_midpoint_w(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
) -> torch.Tensor:
    """Approximate the midpoint between the two finger pads in world frame."""

    robot: Articulation = env.scene[left_finger_cfg.name]
    left_pad_center_w = _body_point_w(robot, left_finger_cfg, LEFT_FINGER_PAD_CENTER_LOCAL)
    right_pad_center_w = _body_point_w(robot, right_finger_cfg, RIGHT_FINGER_PAD_CENTER_LOCAL)
    return 0.5 * (left_pad_center_w + right_pad_center_w)
