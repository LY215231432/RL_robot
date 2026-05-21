# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class XArm7ControlPanelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared PPO defaults for simple xArm7 control-panel skills."""

    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    check_for_nan = True
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.5e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7SwitchPanelPPORunnerCfg(XArm7ControlPanelPPORunnerCfg):
    """PPO defaults for the switch-panel task."""

    experiment_name = "xarm7_switch_panel"


@configclass
class XArm7SliderPanelPPORunnerCfg(XArm7ControlPanelPPORunnerCfg):
    """PPO defaults for the slider-panel task."""

    experiment_name = "xarm7_slider_panel"

