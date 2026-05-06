def _cap_grasp_target_w(
    twist_object: Articulation,
    cap_cfg: SceneEntityCfg,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
) -> torch.Tensor:
    """Return the desired TCP target around the cap side grasp center."""

    cap_pos_w = _single_body_position(twist_object.data.body_pos_w, cap_cfg.body_ids)
    grasp_offset = torch.zeros_like(cap_pos_w)
    grasp_offset[:, 2] = grasp_height
    return cap_pos_w + grasp_offset
