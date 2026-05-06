# Reconstructed version 01: basic staged cap grasp.
# Idea: provide cap pose, gripper/cap errors, and staged rewards for approach -> descend -> grasp -> rotate.

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.twist_cap import mdp


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
    find_cap_above = RewTerm(
        func=mdp.approach_cap_from_above,
        params={"std": 0.06, "approach_clearance": 0.055, "grasp_height": 0.018, "close_joint_pos": 0.85},
        weight=6.0,
    )
    descend_to_grasp = RewTerm(
        func=mdp.descend_to_cap_grasp,
        params={"xy_std": 0.02, "z_std": 0.015, "grasp_height": 0.018, "close_joint_pos": 0.85},
        weight=7.0,
    )
    grasp_cap = RewTerm(
        func=mdp.grasp_cap,
        params={"xy_std": 0.02, "z_std": 0.015, "contact_force_scale": 4.0, "close_joint_pos": 0.85},
        weight=10.0,
    )
    rotate_cap = RewTerm(
        func=mdp.cap_rotation_progress,
        params={"goal_angle": 1.0, "xy_std": 0.02, "z_std": 0.015, "contact_force_scale": 4.0},
        weight=8.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
