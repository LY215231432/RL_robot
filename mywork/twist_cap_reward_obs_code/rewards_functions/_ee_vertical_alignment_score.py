def _ee_vertical_alignment_score(
    env: ManagerBasedRLEnv,
    axis_index: int = 2,
    target_sign: float = -1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return how well a chosen TCP axis aligns with the world vertical direction."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_rot_mat = matrix_from_quat(ee_quat_w)
    ee_axis = ee_rot_mat[..., axis_index]

    target_axis = torch.zeros_like(ee_axis)
    target_axis[:, 2] = target_sign
    alignment = torch.sum(ee_axis * target_axis, dim=1)
    return torch.clamp(alignment, min=0.0, max=1.0).square()
