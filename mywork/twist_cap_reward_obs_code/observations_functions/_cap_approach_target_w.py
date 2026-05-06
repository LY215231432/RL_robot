def _cap_approach_target_w(
    twist_object: Articulation,
    cap_cfg: SceneEntityCfg,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
) -> torch.Tensor:
    """Return the approach target above the cap in world frame."""

    grasp_target_w = _cap_grasp_target_w(twist_object, cap_cfg, grasp_height=grasp_height)
    approach_offset = torch.zeros_like(grasp_target_w)
    approach_offset[:, 2] = approach_clearance
    return grasp_target_w + approach_offset
