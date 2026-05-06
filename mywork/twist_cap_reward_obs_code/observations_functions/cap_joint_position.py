def cap_joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
) -> torch.Tensor:
    """Return the current cap joint angle."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    return twist_object.data.joint_pos[:, asset_cfg.joint_ids]
