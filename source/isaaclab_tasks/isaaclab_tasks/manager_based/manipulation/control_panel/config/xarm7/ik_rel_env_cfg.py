# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.xarm7 import XARM7_CFG

from ... import mdp
from ...control_panel_env_cfg import (
    CONTROL_PANEL_READY_JOINT_POS,
    SliderPanelEnvCfg,
    SwitchPanelEnvCfg,
)


ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
READY_JOINT_POS_BY_NAME = dict(zip(ARM_JOINT_NAMES, CONTROL_PANEL_READY_JOINT_POS))


def _configure_xarm7_common(env_cfg, active_panel_body: str):
    """Attach xArm7, IK actions, and finger contact sensors to a panel environment."""

    env_cfg.scene.robot = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            **READY_JOINT_POS_BY_NAME,
            "joint_left_finger": 0.0,
            "joint_right_finger": 0.0,
        }
    )

    env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint[1-7]"],
        body_name="link_tcp",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
        scale=(0.05, 0.05, 0.035),
    )
    env_cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left_finger", "joint_right_finger"],
        open_command_expr={
            "joint_left_finger": 0.0,
            "joint_right_finger": 0.0,
        },
        close_command_expr={
            "joint_left_finger": 0.85,
            "joint_right_finger": 0.85,
        },
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    env_cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_base",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_tcp",
                name="end_effector",
            ),
        ],
    )

    env_cfg.scene.contact_left_panel = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_finger",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Panel/{active_panel_body}"],
    )
    env_cfg.scene.contact_right_panel = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Panel/{active_panel_body}"],
    )


@configclass
class XArm7SwitchPanelEnvCfg(SwitchPanelEnvCfg):
    """xArm7 configuration for flicking a switch lever."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.0
        self.viewer.eye = (1.05, -0.85, 0.70)
        self.viewer.lookat = (0.40, 0.0, 0.08)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        _configure_xarm7_common(self, active_panel_body="switch_lever")


@configclass
class XArm7SwitchPanelEnvCfg_PLAY(XArm7SwitchPanelEnvCfg):
    """Smaller play configuration for switch-panel rollout and recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False


@configclass
class XArm7SliderPanelEnvCfg(SliderPanelEnvCfg):
    """xArm7 configuration for pushing a slider knob."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.0
        self.viewer.eye = (1.05, -0.85, 0.70)
        self.viewer.lookat = (0.40, 0.0, 0.08)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        _configure_xarm7_common(self, active_panel_body="slider_knob")


@configclass
class XArm7SliderPanelEnvCfg_PLAY(XArm7SliderPanelEnvCfg):
    """Smaller play configuration for slider-panel rollout and recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
