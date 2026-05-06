def _gripper_closed_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    close_joint_pos: float = 0.85,
) -> torch.Tensor:
    """Return how closed the gripper is, normalized to [0, 1]."""

    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    return torch.clamp(joint_pos.mean(dim=1) / close_joint_pos, min=0.0, max=1.0)
