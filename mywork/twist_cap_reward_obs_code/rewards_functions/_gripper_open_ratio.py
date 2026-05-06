def _gripper_open_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Return how open the gripper is, normalized to [0, 1]."""

    return 1.0 - _gripper_closed_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
