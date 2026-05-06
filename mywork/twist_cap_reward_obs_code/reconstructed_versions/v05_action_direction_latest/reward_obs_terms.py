# Reconstructed version 05: latest localization-focused reward.
# Idea: reduce posture-only reward, add explicit action direction guidance toward the cap approach target.

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.twist_cap import mdp

VERTICAL_PRESS_JOINT_POS = [0.0, -0.38, 0.16, 1.22, 0.0, 0.98, 0.0]
ARM_JOINT_CFG = SceneEntityCfg("robot", joint_names=["joint[1-7]"])


class PolicyObsTerms:
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    cap_position = ObsTerm(func=mdp.cap_position_in_robot_root_frame)
    gripper_center_pos = ObsTerm(func=mdp.gripper_center_position_in_robot_root_frame)
    gripper_center_to_cap = ObsTerm(func=mdp.gripper_center_to_cap)
    gripper_center_to_approach = ObsTerm(func=mdp.gripper_center_to_approach_target)
    cap_angle = ObsTerm(func=mdp.cap_joint_position)
    cap_angle_error = ObsTerm(func=mdp.cap_angle_error, params={"goal_angle": 1.0})
    actions = ObsTerm(func=mdp.last_action)


class RewardTerms:
    vertical_tcp = RewTerm(func=mdp.tcp_vertical_alignment, params={"axis_index": 2, "target_sign": -1.0}, weight=8.0)
    vertical_press_posture = RewTerm(
        func=mdp.vertical_press_posture,
        params={"reference_joint_pos": VERTICAL_PRESS_JOINT_POS, "std": 0.45, "robot_cfg": ARM_JOINT_CFG},
        weight=1.0,
    )
    approach_action_direction = RewTerm(
        func=mdp.approach_action_towards_cap,
        params={"approach_clearance": 0.055, "grasp_height": 0.018, "min_action_norm": 0.004, "active_distance": 0.18, "close_joint_pos": 0.85, "xy_weight": 0.75},
        weight=7.0,
    )
    approach_center_coarse = RewTerm(
        func=mdp.gripper_center_to_approach_alignment,
        params={"std": 0.14, "approach_clearance": 0.055, "grasp_height": 0.018, "close_joint_pos": 0.85, "vertical_axis_index": 2, "vertical_target_sign": -1.0},
        weight=12.0,
    )
    finger_center_on_cap = RewTerm(
        func=mdp.gripper_center_to_cap_alignment,
        params={"xy_std": 0.025, "z_std": 0.02, "grasp_height": 0.018, "close_joint_pos": 0.85, "vertical_axis_index": 2, "vertical_target_sign": -1.0},
        weight=9.0,
    )
    side_approach = RewTerm(
        func=mdp.side_approach_penalty,
        params={"xy_std": 0.025, "z_std": 0.02, "safe_xy_radius": 0.012, "grasp_height": 0.018, "close_joint_pos": 0.85},
        weight=-3.0,
    )
    find_cap_above = RewTerm(
        func=mdp.approach_cap_from_above,
        params={"std": 0.06, "approach_clearance": 0.055, "grasp_height": 0.018, "close_joint_pos": 0.85, "vertical_axis_index": 2, "vertical_target_sign": -1.0},
        weight=6.0,
    )
    descend_to_grasp = RewTerm(
        func=mdp.descend_to_cap_grasp,
        params={"xy_std": 0.02, "z_std": 0.015, "grasp_height": 0.018, "close_joint_pos": 0.85, "hover_gate_threshold": 0.65, "reference_joint_pos": VERTICAL_PRESS_JOINT_POS, "posture_std": 0.45, "posture_gate_threshold": 0.35, "robot_posture_cfg": ARM_JOINT_CFG, "vertical_axis_index": 2, "vertical_target_sign": -1.0},
        weight=7.0,
    )
    grasp_cap = RewTerm(
        func=mdp.grasp_cap,
        params={"xy_std": 0.02, "z_std": 0.015, "contact_force_scale": 4.0, "close_joint_pos": 0.85, "descend_gate_threshold": 0.45, "reference_joint_pos": VERTICAL_PRESS_JOINT_POS, "posture_std": 0.45, "posture_gate_threshold": 0.35, "robot_posture_cfg": ARM_JOINT_CFG},
        weight=10.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
