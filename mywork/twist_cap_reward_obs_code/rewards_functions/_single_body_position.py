def _single_body_position(body_pos_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world position for configs that may resolve to a list or slice."""

    selected_pos_w = body_pos_w[:, body_ids, :]
    if selected_pos_w.ndim == 3:
        return selected_pos_w[:, 0, :]
    return selected_pos_w
