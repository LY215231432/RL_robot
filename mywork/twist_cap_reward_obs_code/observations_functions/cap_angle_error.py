def cap_angle_error(
    env: ManagerBasedRLEnv,
    goal_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
) -> torch.Tensor:
    """Return the remaining rotation needed to reach the target cap angle."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    return goal_angle - twist_object.data.joint_pos[:, asset_cfg.joint_ids]
