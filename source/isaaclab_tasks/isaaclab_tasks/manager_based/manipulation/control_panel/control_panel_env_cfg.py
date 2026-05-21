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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp


CONTROL_PANEL_READY_JOINT_POS = [0.0, -0.42, 0.0, 1.18, 0.0, 1.08, 0.0]
ARM_JOINT_CFG = SceneEntityCfg("robot", joint_names=["joint[1-7]"])

SWITCH_START = -0.45
SWITCH_GOAL = 0.45
SWITCH_DONE = 0.40
SWITCH_TARGET_LOCAL = (0.0, 0.0, 0.075)
SWITCH_APPROACH_OFFSET = (-0.065, 0.0, 0.018)
SWITCH_PUSH_AXIS = (1.0, 0.0, 0.0)

SLIDER_START = 0.0
SLIDER_GOAL = 0.09
SLIDER_DONE = 0.085
SLIDER_TARGET_LOCAL = (0.0, -0.022, 0.018)
SLIDER_APPROACH_OFFSET = (0.0, -0.065, 0.012)
SLIDER_PUSH_AXIS = (0.0, 1.0, 0.0)


def _resolve_asset(asset_file_name: str) -> Path:
    """Resolve a local control-panel URDF shipped with this task package."""

    return Path(__file__).resolve().parent / "assets" / asset_file_name


def _create_panel_cfg(
    asset_file_name: str,
    joint_name: str,
    initial_joint_position: float,
    joint_damping: float,
) -> ArticulationCfg:
    """Create a fixed-base articulated panel object from a local URDF."""

    asset_path = _resolve_asset(asset_file_name)
    usd_dir = asset_path.parent / f".isaaclab_usd_{asset_path.stem}"

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Panel",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            usd_dir=str(usd_dir),
            fix_base=True,
            root_link_name="panel_base",
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
            pos=[0.40, 0.0, 0.0],
            joint_pos={joint_name: initial_joint_position},
        ),
        actuators={
            "panel_passive": ImplicitActuatorCfg(
                joint_names_expr=[joint_name],
                effort_limit_sim=4.0,
                velocity_limit_sim=5.0,
                stiffness=0.0,
                damping=joint_damping,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class ControlPanelSceneCfg(InteractiveSceneCfg):
    """Shared scene definition for simple control-panel pushing tasks."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    panel: ArticulationCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.42, 0.0, -0.04]),
        spawn=sim_utils.CuboidCfg(
            size=(0.78, 0.56, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.34), roughness=0.8),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12), roughness=1.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class SwitchPanelSceneCfg(ControlPanelSceneCfg):
    """Scene containing a single revolute switch lever."""

    panel: ArticulationCfg = _create_panel_cfg(
        asset_file_name="switch_panel.urdf",
        joint_name="switch_joint",
        initial_joint_position=SWITCH_START,
        joint_damping=0.02,
    )


@configclass
class SliderPanelSceneCfg(ControlPanelSceneCfg):
    """Scene containing a single prismatic slider knob."""

    panel: ArticulationCfg = _create_panel_cfg(
        asset_file_name="slider_panel.urdf",
        joint_name="slider_joint",
        initial_joint_position=SLIDER_START,
        joint_damping=0.08,
    )


@configclass
class CommandsCfg:
    """No external commands are used for the fixed control-panel prototypes."""


@configclass
class ActionsCfg:
    """Action specifications for xArm7 control-panel tasks."""

    arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class SwitchObservationsCfg:
    """Observation specifications for the switch-panel task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        panel_target = ObsTerm(
            func=mdp.panel_target_position_in_robot_root_frame,
            params={
                "target_local_offset": SWITCH_TARGET_LOCAL,
                "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
            },
        )
        gripper_center_pos = ObsTerm(func=mdp.gripper_center_position_in_robot_root_frame)
        gripper_to_approach = ObsTerm(
            func=mdp.gripper_center_to_panel_target,
            params={
                "target_local_offset": SWITCH_TARGET_LOCAL,
                "target_world_offset": SWITCH_APPROACH_OFFSET,
                "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
            },
        )
        gripper_to_contact = ObsTerm(
            func=mdp.gripper_center_to_panel_target,
            params={
                "target_local_offset": SWITCH_TARGET_LOCAL,
                "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
            },
        )
        panel_joint_pos = ObsTerm(
            func=mdp.panel_joint_position,
            params={"asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint")},
        )
        panel_joint_vel = ObsTerm(
            func=mdp.panel_joint_velocity,
            params={"asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint")},
        )
        panel_joint_error = ObsTerm(
            func=mdp.panel_joint_error,
            params={"goal_position": SWITCH_GOAL, "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SliderObservationsCfg:
    """Observation specifications for the slider-panel task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        panel_target = ObsTerm(
            func=mdp.panel_target_position_in_robot_root_frame,
            params={
                "target_local_offset": SLIDER_TARGET_LOCAL,
                "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
            },
        )
        gripper_center_pos = ObsTerm(func=mdp.gripper_center_position_in_robot_root_frame)
        gripper_to_approach = ObsTerm(
            func=mdp.gripper_center_to_panel_target,
            params={
                "target_local_offset": SLIDER_TARGET_LOCAL,
                "target_world_offset": SLIDER_APPROACH_OFFSET,
                "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
            },
        )
        gripper_to_contact = ObsTerm(
            func=mdp.gripper_center_to_panel_target,
            params={
                "target_local_offset": SLIDER_TARGET_LOCAL,
                "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
            },
        )
        panel_joint_pos = ObsTerm(
            func=mdp.panel_joint_position,
            params={"asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint")},
        )
        panel_joint_vel = ObsTerm(
            func=mdp.panel_joint_velocity,
            params={"asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint")},
        )
        panel_joint_error = ObsTerm(
            func=mdp.panel_joint_error,
            params={"goal_position": SLIDER_GOAL, "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint")},
        )
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
class SwitchRewardsCfg:
    """Reward terms for flicking a switch with one gripper finger."""

    comfortable_posture = RewTerm(
        func=mdp.joint_posture_score,
        params={"reference_joint_pos": CONTROL_PANEL_READY_JOINT_POS, "std": 0.55, "robot_cfg": ARM_JOINT_CFG},
        weight=0.8,
    )
    approach_switch = RewTerm(
        func=mdp.gripper_to_panel_approach_alignment,
        params={
            "target_local_offset": SWITCH_TARGET_LOCAL,
            "target_world_offset": SWITCH_APPROACH_OFFSET,
            "std": 0.18,
            "contact_std": 0.06,
            "contact_gate_threshold": 0.55,
            "target_radius": 0.025,
            "progress_fade_start": 0.08,
            "start_position": SWITCH_START,
            "goal_position": SWITCH_GOAL,
            "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
        weight=1.5,
    )
    contact_switch_target = RewTerm(
        func=mdp.finger_to_panel_target_alignment,
        params={
            "target_local_offset": SWITCH_TARGET_LOCAL,
            "std": 0.06,
            "target_radius": 0.025,
            "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
        },
        weight=5.5,
    )
    push_direction = RewTerm(
        func=mdp.push_action_towards_goal,
        params={
            "push_axis": SWITCH_PUSH_AXIS,
            "target_local_offset": SWITCH_TARGET_LOCAL,
            "target_std": 0.075,
            "target_radius": 0.025,
            "target_gate_threshold": 0.30,
            "contact_gate_mix": 0.35,
            "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
        },
        weight=6.0,
    )
    finger_contact = RewTerm(
        func=mdp.finger_panel_contact_near_target,
        params={
            "target_local_offset": SWITCH_TARGET_LOCAL,
            "target_std": 0.08,
            "target_radius": 0.025,
            "target_gate_threshold": 0.25,
            "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
        },
        weight=4.0,
    )
    hover_without_contact = RewTerm(
        func=mdp.near_target_without_contact_penalty,
        params={
            "target_local_offset": SWITCH_TARGET_LOCAL,
            "target_std": 0.08,
            "target_radius": 0.025,
            "target_gate_threshold": 0.50,
            "progress_clearance": 0.05,
            "start_position": SWITCH_START,
            "goal_position": SWITCH_GOAL,
            "body_cfg": SceneEntityCfg("panel", body_names="switch_lever"),
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
        weight=-1.2,
    )
    switch_velocity = RewTerm(
        func=mdp.panel_joint_velocity_towards_goal,
        params={
            "start_position": SWITCH_START,
            "goal_position": SWITCH_GOAL,
            "velocity_scale": 1.0,
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
        weight=8.0,
    )
    switch_progress = RewTerm(
        func=mdp.panel_joint_progress,
        params={
            "start_position": SWITCH_START,
            "goal_position": SWITCH_GOAL,
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
        weight=30.0,
    )
    switch_success = RewTerm(
        func=mdp.panel_joint_goal_bonus,
        params={
            "start_position": SWITCH_START,
            "done_position": SWITCH_DONE,
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
        weight=20.0,
    )
    keep_gripper_open = RewTerm(func=mdp.keep_gripper_open, weight=0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class SliderRewardsCfg:
    """Reward terms for pushing a slider knob to the marked goal zone."""

    comfortable_posture = RewTerm(
        func=mdp.joint_posture_score,
        params={"reference_joint_pos": CONTROL_PANEL_READY_JOINT_POS, "std": 0.55, "robot_cfg": ARM_JOINT_CFG},
        weight=0.8,
    )
    approach_slider = RewTerm(
        func=mdp.gripper_to_panel_approach_alignment,
        params={
            "target_local_offset": SLIDER_TARGET_LOCAL,
            "target_world_offset": SLIDER_APPROACH_OFFSET,
            "std": 0.18,
            "contact_std": 0.055,
            "contact_gate_threshold": 0.55,
            "target_radius": 0.018,
            "progress_fade_start": 0.08,
            "start_position": SLIDER_START,
            "goal_position": SLIDER_GOAL,
            "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
        weight=1.5,
    )
    contact_slider_target = RewTerm(
        func=mdp.finger_to_panel_target_alignment,
        params={
            "target_local_offset": SLIDER_TARGET_LOCAL,
            "std": 0.055,
            "target_radius": 0.018,
            "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
        },
        weight=5.0,
    )
    push_direction = RewTerm(
        func=mdp.push_action_towards_goal,
        params={
            "push_axis": SLIDER_PUSH_AXIS,
            "target_local_offset": SLIDER_TARGET_LOCAL,
            "target_std": 0.07,
            "target_radius": 0.018,
            "target_gate_threshold": 0.30,
            "contact_gate_mix": 0.35,
            "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
        },
        weight=5.0,
    )
    finger_contact = RewTerm(
        func=mdp.finger_panel_contact_near_target,
        params={
            "target_local_offset": SLIDER_TARGET_LOCAL,
            "target_std": 0.075,
            "target_radius": 0.018,
            "target_gate_threshold": 0.25,
            "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
        },
        weight=3.0,
    )
    hover_without_contact = RewTerm(
        func=mdp.near_target_without_contact_penalty,
        params={
            "target_local_offset": SLIDER_TARGET_LOCAL,
            "target_std": 0.075,
            "target_radius": 0.018,
            "target_gate_threshold": 0.50,
            "progress_clearance": 0.04,
            "start_position": SLIDER_START,
            "goal_position": SLIDER_GOAL,
            "body_cfg": SceneEntityCfg("panel", body_names="slider_knob"),
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
        weight=-1.0,
    )
    slider_velocity = RewTerm(
        func=mdp.panel_joint_velocity_towards_goal,
        params={
            "start_position": SLIDER_START,
            "goal_position": SLIDER_GOAL,
            "velocity_scale": 0.35,
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
        weight=6.0,
    )
    slider_progress = RewTerm(
        func=mdp.panel_joint_progress,
        params={
            "start_position": SLIDER_START,
            "goal_position": SLIDER_GOAL,
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
        weight=24.0,
    )
    slider_success = RewTerm(
        func=mdp.panel_joint_goal_bonus,
        params={
            "start_position": SLIDER_START,
            "done_position": SLIDER_DONE,
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
        weight=16.0,
    )
    keep_gripper_open = RewTerm(func=mdp.keep_gripper_open, weight=0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class SwitchTerminationsCfg:
    """Termination terms for the switch-panel task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    switch_goal_reached = DoneTerm(
        func=mdp.panel_joint_reached,
        params={
            "goal_position": SWITCH_DONE,
            "start_position": SWITCH_START,
            "asset_cfg": SceneEntityCfg("panel", joint_names="switch_joint"),
        },
    )


@configclass
class SliderTerminationsCfg:
    """Termination terms for the slider-panel task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    slider_goal_reached = DoneTerm(
        func=mdp.panel_joint_reached,
        params={
            "goal_position": SLIDER_DONE,
            "start_position": SLIDER_START,
            "asset_cfg": SceneEntityCfg("panel", joint_names="slider_joint"),
        },
    )


@configclass
class CurriculumCfg:
    """No curriculum is needed for the first control-panel prototypes."""


@configclass
class SwitchPanelEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the switch-panel pushing environment."""

    scene: SwitchPanelSceneCfg = SwitchPanelSceneCfg(num_envs=1024, env_spacing=2.0)
    observations: SwitchObservationsCfg = SwitchObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SwitchRewardsCfg = SwitchRewardsCfg()
    terminations: SwitchTerminationsCfg = SwitchTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SliderPanelEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the slider-panel pushing environment."""

    scene: SliderPanelSceneCfg = SliderPanelSceneCfg(num_envs=1024, env_spacing=2.0)
    observations: SliderObservationsCfg = SliderObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SliderRewardsCfg = SliderRewardsCfg()
    terminations: SliderTerminationsCfg = SliderTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
