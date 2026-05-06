# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


VERTICAL_PRESS_JOINT_POS = [0.0, -0.38, 0.16, 1.22, 0.0, 0.98, 0.0]
ARM_JOINT_CFG = SceneEntityCfg("robot", joint_names=["joint[1-7]"])


def _resolve_twist_cap_urdf() -> Path:
    """Resolve the local bottle-cap URDF shipped with this task package."""

    return Path(__file__).resolve().parent / "assets" / "bottle_cap.urdf"


def _create_twist_object_cfg() -> ArticulationCfg:
    """Create the articulated bottle-cap object used by the twist skill."""

    asset_path = _resolve_twist_cap_urdf()
    usd_dir = asset_path.parent / ".isaaclab_usd"

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/TwistObject",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            usd_dir=str(usd_dir),
            fix_base=True,
            root_link_name="bottle_link",
            merge_fixed_joints=False,
            make_instanceable=False,
            collider_type="convex_decomposition",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.38, 0.0, 0.0],
            joint_pos={"cap_joint": 0.0},
        ),
        actuators={
            "cap_passive": ImplicitActuatorCfg(
                joint_names_expr=["cap_joint"],
                effort_limit_sim=2.0,
                velocity_limit_sim=8.0,
                stiffness=0.0,
                damping=0.05,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class TwistCapSceneCfg(InteractiveSceneCfg):
    """Scene definition for the bottle-cap twisting skill."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    twist_object: ArticulationCfg = _create_twist_object_cfg()

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """No external commands are used in the first twist-cap skill version."""


@configclass
class ActionsCfg:
    """Action specifications for the twist-cap MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the twist-cap MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cap_position = ObsTerm(func=mdp.cap_position_in_robot_root_frame)
        gripper_center_pos = ObsTerm(func=mdp.gripper_center_position_in_robot_root_frame)
        gripper_center_to_cap = ObsTerm(func=mdp.gripper_center_to_cap)
        gripper_center_to_approach = ObsTerm(func=mdp.gripper_center_to_approach_target)
        cap_angle = ObsTerm(func=mdp.cap_joint_position)
        cap_angle_error = ObsTerm(func=mdp.cap_angle_error, params={"goal_angle": 1.0})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for environment reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class RewardsCfg:
    """Reward terms for the twist-cap skill."""

    vertical_tcp = RewTerm(
        func=mdp.tcp_vertical_alignment,
        params={"axis_index": 2, "target_sign": -1.0},
        weight=8.0,
    )
    vertical_press_posture = RewTerm(
        func=mdp.vertical_press_posture,
        params={
            "reference_joint_pos": VERTICAL_PRESS_JOINT_POS,
            "std": 0.45,
            "robot_cfg": ARM_JOINT_CFG,
        },
        weight=1.0,
    )
    approach_action_direction = RewTerm(
        func=mdp.approach_action_towards_cap,
        params={
            "approach_clearance": 0.055,
            "grasp_height": 0.018,
            "min_action_norm": 0.004,
            "active_distance": 0.18,
            "close_joint_pos": 0.85,
            "xy_weight": 0.75,
        },
        weight=7.0,
    )
    approach_center_coarse = RewTerm(
        func=mdp.gripper_center_to_approach_alignment,
        params={
            "std": 0.14,
            "approach_clearance": 0.055,
            "grasp_height": 0.018,
            "close_joint_pos": 0.85,
            "vertical_axis_index": 2,
            "vertical_target_sign": -1.0,
        },
        weight=12.0,
    )
    finger_center_on_cap = RewTerm(
        func=mdp.gripper_center_to_cap_alignment,
        params={
            "xy_std": 0.025,
            "z_std": 0.02,
            "grasp_height": 0.018,
            "close_joint_pos": 0.85,
            "vertical_axis_index": 2,
            "vertical_target_sign": -1.0,
        },
        weight=9.0,
    )
    side_approach = RewTerm(
        func=mdp.side_approach_penalty,
        params={
            "xy_std": 0.025,
            "z_std": 0.02,
            "safe_xy_radius": 0.012,
            "grasp_height": 0.018,
            "close_joint_pos": 0.85,
        },
        weight=-3.0,
    )
    find_cap_above = RewTerm(
        func=mdp.approach_cap_from_above,
        params={
            "std": 0.06,
            "approach_clearance": 0.055,
            "grasp_height": 0.018,
            "close_joint_pos": 0.85,
            "vertical_axis_index": 2,
            "vertical_target_sign": -1.0,
        },
        weight=6.0,
    )
    descend_to_grasp = RewTerm(
        func=mdp.descend_to_cap_grasp,
        params={
            "xy_std": 0.02,
            "z_std": 0.015,
            "grasp_height": 0.018,
            "close_joint_pos": 0.85,
            "hover_gate_threshold": 0.65,
            "reference_joint_pos": VERTICAL_PRESS_JOINT_POS,
            "posture_std": 0.45,
            "posture_gate_threshold": 0.35,
            "robot_posture_cfg": ARM_JOINT_CFG,
            "vertical_axis_index": 2,
            "vertical_target_sign": -1.0,
        },
        weight=7.0,
    )
    grasp_cap = RewTerm(
        func=mdp.grasp_cap,
        params={
            "xy_std": 0.02,
            "z_std": 0.015,
            "contact_force_scale": 4.0,
            "close_joint_pos": 0.85,
            "descend_gate_threshold": 0.45,
            "reference_joint_pos": VERTICAL_PRESS_JOINT_POS,
            "posture_std": 0.45,
            "posture_gate_threshold": 0.35,
            "robot_posture_cfg": ARM_JOINT_CFG,
        },
        weight=10.0,
    )
    twist_tcp = RewTerm(
        func=mdp.twist_tcp_angular_velocity,
        params={
            "xy_std": 0.02,
            "z_std": 0.015,
            "contact_force_scale": 4.0,
            "close_joint_pos": 0.85,
            "velocity_scale": 1.75,
            "max_velocity": 4.5,
            "grasp_gate_threshold": 0.65,
        },
        weight=0.0,
    )
    rotate_cap = RewTerm(
        func=mdp.cap_rotation_progress,
        params={
            "goal_angle": 1.0,
            "xy_std": 0.02,
            "z_std": 0.015,
            "contact_force_scale": 4.0,
            "close_joint_pos": 0.85,
            "grasp_gate_threshold": 0.65,
        },
        weight=0.0,
    )
    rotate_cap_velocity = RewTerm(
        func=mdp.cap_positive_rotation_velocity,
        params={
            "xy_std": 0.02,
            "z_std": 0.015,
            "contact_force_scale": 4.0,
            "close_joint_pos": 0.85,
            "grasp_gate_threshold": 0.65,
        },
        weight=0.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the twist-cap MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cap_goal_reached = DoneTerm(func=mdp.cap_rotation_reached, params={"goal_angle": 1.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for smoothing actions later in training."""

    vertical_tcp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "vertical_tcp", "weight": 4.0, "num_steps": 20000},
    )

    vertical_press_posture = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "vertical_press_posture", "weight": 0.5, "num_steps": 20000},
    )

    approach_action_direction = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "approach_action_direction", "weight": 2.0, "num_steps": 20000},
    )

    approach_center_coarse = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "approach_center_coarse", "weight": 3.0, "num_steps": 20000},
    )

    finger_center_on_cap = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "finger_center_on_cap", "weight": 4.0, "num_steps": 20000},
    )

    side_approach = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "side_approach", "weight": -1.5, "num_steps": 20000},
    )

    find_cap_above = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "find_cap_above", "weight": 2.5, "num_steps": 20000},
    )

    descend_to_grasp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "descend_to_grasp", "weight": 3.0, "num_steps": 20000},
    )

    grasp_cap = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grasp_cap", "weight": 6.0, "num_steps": 20000},
    )

    twist_tcp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "twist_tcp", "weight": 5.0, "num_steps": 20000},
    )

    rotate_cap = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "rotate_cap", "weight": 16.0, "num_steps": 20000},
    )

    rotate_cap_velocity = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "rotate_cap_velocity", "weight": 1.5, "num_steps": 20000},
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 15000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 15000},
    )


@configclass
class TwistCapEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bottle-cap twisting environment."""

    scene: TwistCapSceneCfg = TwistCapSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
