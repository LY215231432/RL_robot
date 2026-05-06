def _single_body_quaternion(body_quat_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world quaternion for configs that may resolve to a list or slice."""

    selected_quat_w = body_quat_w[:, body_ids, :]
    if selected_quat_w.ndim == 3:
        return selected_quat_w[:, 0, :]
    return selected_quat_w
