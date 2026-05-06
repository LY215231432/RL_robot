def tcp_vertical_alignment(
    env: ManagerBasedRLEnv,
    axis_index: int = 2,
    target_sign: float = -1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the TCP for keeping the selected axis vertical to the ground."""

    return _ee_vertical_alignment_score(
        env,
        axis_index=axis_index,
        target_sign=target_sign,
        ee_frame_cfg=ee_frame_cfg,
    )
