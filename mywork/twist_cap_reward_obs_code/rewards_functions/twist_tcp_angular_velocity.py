def twist_tcp_angular_velocity(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    contact_force_scale: float,
    close_joint_pos: float = 0.85,
    velocity_scale: float = 2.5,
    max_velocity: float = 6.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_tcp"),
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    grasp_gate_threshold: float = 0.6,
) -> torch.Tensor:
    """Reward the TCP for actively twisting about the bottle-cap axis once the cap is grasped."""

    robot: Articulation = env.scene[robot_cfg.name]
    tcp_ang_vel_w = _single_body_vector(robot.data.body_ang_vel_w, robot_cfg.body_ids)
    positive_twist_speed = torch.clamp(tcp_ang_vel_w[:, 2], min=0.0, max=max_velocity)

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
    grasp_gate = _stage_gate(grasp_score, grasp_gate_threshold)
    return (1.0 - torch.exp(-positive_twist_speed / velocity_scale)) * grasp_gate
