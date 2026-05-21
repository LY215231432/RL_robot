# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the locally stored UFactory xArm7 with gripper."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _resolve_xarm7_urdf() -> Path:
    """Resolve the user-local xArm7 URDF from the current workspace."""

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "mywork" / "mymujoco" / "xarm7_urdf_base" / "xarm7.urdf"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Unable to locate mywork/mymujoco/xarm7_urdf_base/xarm7.urdf from the current workspace.")


XARM7_URDF_PATH = _resolve_xarm7_urdf()
XARM7_USD_DIR = XARM7_URDF_PATH.parent / ".isaaclab_usd"

XARM7_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(XARM7_URDF_PATH),
        usd_dir=str(XARM7_USD_DIR),
        fix_base=True,
        root_link_name="link_base",
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        collider_type="convex_decomposition",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.247,
            "joint3": 0.0,
            "joint4": 0.909,
            "joint5": 0.0,
            "joint6": 1.15644,
            "joint7": 0.0,
            "joint_left_finger": 0.0,
            "joint_right_finger": 0.0,
        },
    ),
    actuators={
        "xarm7_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim={
                "joint[1-2]": 50.0,
                "joint[3-5]": 30.0,
                "joint[6-7]": 20.0,
            },
            velocity_limit_sim=3.14,
            stiffness=80.0,
            damping=4.0,
        ),
        "xarm7_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_left_finger",
                "joint_right_finger",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the locally stored xArm7 articulation."""
