def _contact_force_score(env: ManagerBasedRLEnv, sensor_name: str, force_scale: float = 6.0) -> torch.Tensor:
    """Return a smooth [0, 1] score from filtered contact force magnitudes."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    if contact_sensor.data.force_matrix_w is not None:
        contact_force_w = torch.nan_to_num(contact_sensor.data.force_matrix_w, nan=0.0).reshape(env.num_envs, -1, 3)
    else:
        contact_force_w = torch.nan_to_num(contact_sensor.data.net_forces_w, nan=0.0).reshape(env.num_envs, -1, 3)

    contact_mag = torch.linalg.norm(contact_force_w, dim=-1).sum(dim=1)
    return 1.0 - torch.exp(-contact_mag / force_scale)
