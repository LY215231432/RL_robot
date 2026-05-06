def _joint_posture_score(
    env: ManagerBasedRLEnv,
    reference_joint_pos: list[float],
    std: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a smooth [0, 1] score for keeping the arm near a reference posture."""

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    reference = joint_pos.new_tensor(reference_joint_pos).unsqueeze(0)
    if joint_pos.shape[1] != reference.shape[1]:
        joint_pos = joint_pos[:, : reference.shape[1]]
    posture_error = torch.linalg.norm(joint_pos - reference, dim=1)
    return 1.0 - torch.tanh(posture_error / std)
