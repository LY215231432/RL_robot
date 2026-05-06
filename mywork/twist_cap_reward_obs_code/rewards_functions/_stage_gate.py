def _stage_gate(score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert a smooth score into a gate that activates only past a threshold."""

    return torch.clamp((score - threshold) / max(1.0 - threshold, STAGE_GATE_EPS), min=0.0, max=1.0)
