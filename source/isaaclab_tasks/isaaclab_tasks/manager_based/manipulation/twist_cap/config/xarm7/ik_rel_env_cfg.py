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
from ...twist_cap_env_cfg import TwistCapEnvCfg, VERTICAL_PRESS_JOINT_POS


ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
VERTICAL_PRESS_JOINT_POS_BY_NAME = dict(zip(ARM_JOINT_NAMES, VERTICAL_PRESS_JOINT_POS))


@configclass
class XArm7TwistCapEnvCfg(TwistCapEnvCfg):
    """xArm7 configuration for the bottle-cap twisting task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5
        self.viewer.eye = (1.2, -0.9, 0.75)
        self.viewer.lookat = (0.38, 0.0, 0.14)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        self.scene.robot = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.init_state.joint_pos.update(
            {
                **VERTICAL_PRESS_JOINT_POS_BY_NAME,
                "joint_left_finger": 0.0,
                "joint_right_finger": 0.0,
            }
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-7]"],
            body_name="link_tcp",
            controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
            scale=(0.05, 0.05, 0.035),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
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
        self.scene.ee_frame = FrameTransformerCfg(
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

        self.scene.contact_left_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_finger",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/TwistObject/cap_link"],
        )
        self.scene.contact_right_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_finger",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/TwistObject/cap_link"],
        )


@configclass
class XArm7TwistCapEnvCfg_PLAY(XArm7TwistCapEnvCfg):
    """Smaller play configuration for policy rollout and recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
