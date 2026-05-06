def _ee_position_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the end-effector TCP position in world frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]
