def approach_action_towards_cap(
    env: ManagerBasedRLEnv,
    approach_clearance: float = CAP_APPROACH_CLEARANCE,
    grasp_height: float = CAP_SIDE_GRASP_HEIGHT,
    min_action_norm: float = 0.004,
    active_distance: float = 0.18,
    close_joint_pos: float = 0.85,
    xy_weight: float = 0.7,
    cap_cfg: SceneEntityCfg = SceneEntityCfg("twist_object", body_names="cap_link"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint_left_finger", "joint_right_finger"]),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_finger"),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_finger"),
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Reward position-IK actions that move the finger-pad midpoint toward the cap approach target."""

    twist_object: Articulation = env.scene[cap_cfg.name]
    gripper_center_w = _gripper_pad_midpoint_w(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
    )
    approach_target_w = _cap_approach_target_w(
        twist_object,
        cap_cfg,
        approach_clearance=approach_clearance,
        grasp_height=grasp_height,
    )

    desired_delta = approach_target_w - gripper_center_w
    arm_action = env.action_manager.get_term(action_name).processed_actions[:, :3]

    desired_xy = desired_delta.clone()
    desired_xy[:, 2] = 0.0
    action_xy = arm_action.clone()
    action_xy[:, 2] = 0.0

    desired_norm = torch.linalg.norm(desired_delta, dim=1, keepdim=True)
    action_norm = torch.linalg.norm(arm_action, dim=1, keepdim=True)
    desired_xy_norm = torch.linalg.norm(desired_xy, dim=1, keepdim=True)
    action_xy_norm = torch.linalg.norm(action_xy, dim=1, keepdim=True)

    direction_score = torch.sum(
        arm_action / (action_norm + STAGE_GATE_EPS) * desired_delta / (desired_norm + STAGE_GATE_EPS),
        dim=1,
    )
    xy_direction_score = torch.sum(
        action_xy / (action_xy_norm + STAGE_GATE_EPS) * desired_xy / (desired_xy_norm + STAGE_GATE_EPS),
        dim=1,
    )
    direction_score = torch.clamp(direction_score, min=0.0, max=1.0)
    xy_direction_score = torch.clamp(xy_direction_score, min=0.0, max=1.0)

    action_mag = torch.clamp(action_norm.squeeze(-1) / min_action_norm, min=0.0, max=1.0)
    distance_gate = torch.clamp(desired_norm.squeeze(-1) / active_distance, min=0.0, max=1.0)
    gripper_open = _gripper_open_ratio(env, gripper_cfg=gripper_cfg, close_joint_pos=close_joint_pos)
    direction_mix = xy_weight * xy_direction_score + (1.0 - xy_weight) * direction_score
    return direction_mix * action_mag * distance_gate * (0.5 + 0.5 * gripper_open)
