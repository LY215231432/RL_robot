def _single_body_vector(body_vec_w: torch.Tensor, body_ids: list[int] | slice | int) -> torch.Tensor:
    """Return a single body's world-frame vector for configs that may resolve to a list or slice."""

    selected_vec_w = body_vec_w[:, body_ids, :]
    if selected_vec_w.ndim == 3:
        return selected_vec_w[:, 0, :]
    return selected_vec_w
