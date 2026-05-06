def cap_positive_rotation_velocity(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", joint_names="cap_joint"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    grasp_gate_threshold: float = 0.6,
) -> torch.Tensor:
    """Provide a small reward for rotating the cap in the unscrewing direction after grasping."""

    twist_object: Articulation = env.scene[asset_cfg.name]
    cap_vel = twist_object.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
    grasp_score = grasp_cap(
        env,
        xy_std=xy_std,
        z_std=z_std,
        contact_force_scale=contact_force_scale,
        close_joint_pos=close_joint_pos,
        cap_cfg=cap_cfg,
        ee_frame_cfg=ee_frame_cfg,
        gripper_cfg=gripper_cfg,
    )
    return torch.clamp(cap_vel, min=0.0, max=5.0) * _stage_gate(grasp_score, grasp_gate_threshold)
