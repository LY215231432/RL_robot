def vertical_press_posture(
    env: ManagerBasedRLEnv,
    reference_joint_pos: list[float],
    std: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-7]"]),
) -> torch.Tensor:
    """Reward the arm for staying in a posture that supports top-down cap pressing."""

    return _joint_posture_score(
        env,
        reference_joint_pos=reference_joint_pos,
        std=std,
        robot_cfg=robot_cfg,
    )
