def _body_point_w(
    asset: Articulation,
    body_cfg: SceneEntityCfg,
    local_point: tuple[float, float, float],
) -> torch.Tensor:
    """Return a local point on a body transformed into the world frame."""

    body_pos_w = _single_body_position(asset.data.body_pos_w, body_cfg.body_ids)
    body_quat_w = _single_body_quaternion(asset.data.body_quat_w, body_cfg.body_ids)
    local_point_tensor = body_pos_w.new_tensor(local_point).expand_as(body_pos_w)
    return body_pos_w + quat_apply(body_quat_w, local_point_tensor)
