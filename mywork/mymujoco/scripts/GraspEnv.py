
import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if sys.platform.startswith("linux") else "glfw"

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from project_paths import BASE_DIR

class GraspEnv(gym.Env):
    """xArm7 single-block grasping environment."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        scene_path = os.path.join(BASE_DIR, "grasping_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=self._ctrl_low, high=self._ctrl_high, dtype=np.float32)

        self._table_top_z = 0.12
        self._cube_half = 0.025

        self._object_body = "object_to_grasp"
        self._object_free_joint = "object_free"
        self._object_geom = "object_geom"
        self._object_qpos_adr = int(np.asarray(self.model.joint(self._object_free_joint).qposadr).reshape(-1)[0])
        self._object_dof_adr = int(np.asarray(self.model.joint(self._object_free_joint).dofadr).reshape(-1)[0])
        self._object_geom_id = int(np.asarray(self.model.geom(self._object_geom).id).reshape(-1)[0])
        self._left_finger_pad_geom_ids = {
            int(np.asarray(self.model.geom("left_finger_pad_1").id).reshape(-1)[0]),
            int(np.asarray(self.model.geom("left_finger_pad_2").id).reshape(-1)[0]),
        }
        self._right_finger_pad_geom_ids = {
            int(np.asarray(self.model.geom("right_finger_pad_1").id).reshape(-1)[0]),
            int(np.asarray(self.model.geom("right_finger_pad_2").id).reshape(-1)[0]),
        }
        self._finger_pad_geom_ids = self._left_finger_pad_geom_ids | self._right_finger_pad_geom_ids

        obs_dim = self.model.nq + self.model.nv + 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        self.max_steps = 200
        self._step_count = 0
        self._prev_distance = None
        self._init_object_xy = None
        self._pre_grasp_height_target = 0.08
        self._pre_grasp_height_window = 0.05
        self._pre_grasp_xy_gate = 0.10
        self._grasp_xy_gate = 0.04
        self._lift_target_z = 0.25
        self._last_reward_terms = {}

        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self, initial=False):
        qpos = self.data.qpos
        qvel = self.data.qvel
        tcp_pos = np.asarray(self.data.site("link_tcp").xpos).reshape(-1)
        object_pos = np.asarray(self.data.body(self._object_body).xpos).reshape(-1)
        object_quat = np.asarray(self.data.body(self._object_body).xquat).reshape(-1)
        vec = object_pos - tcp_pos
        obs = np.concatenate([qpos.flatten(), qvel.flatten(), tcp_pos, object_pos, object_quat, vec], dtype=np.float64)
        if initial:
            return np.zeros(obs.shape, dtype=np.float64)
        return obs

    def _has_finger_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            if (g1 == self._object_geom_id and g2 in self._finger_pad_geom_ids) or (g2 == self._object_geom_id and g1 in self._finger_pad_geom_ids):
                return True
        return False

    def _finger_contacts(self):
        left = False
        right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            if g1 == self._object_geom_id:
                other = g2
            elif g2 == self._object_geom_id:
                other = g1
            else:
                continue
            if other in self._left_finger_pad_geom_ids:
                left = True
            if other in self._right_finger_pad_geom_ids:
                right = True
        return left, right

    def _get_reward(self):
        tcp_site = self.data.site("link_tcp")
        tcp_pos = np.asarray(tcp_site.xpos).reshape(-1)
        tcp_xmat = np.asarray(tcp_site.xmat).reshape(3, 3)
        tool_z = tcp_xmat[:, 2]
        obj = self.data.body(self._object_body)
        object_pos = np.asarray(obj.xpos).reshape(-1)
        object_vel = np.asarray(obj.cvel[3:6]).reshape(-1)
        object_z = float(object_pos[2])
        distance = float(np.linalg.norm(object_pos - tcp_pos))
        xy_dist = float(np.linalg.norm((object_pos - tcp_pos)[:2]))
        tcp_above_obj = float(tcp_pos[2] - object_pos[2])
        table_object_z = self._table_top_z + self._cube_half + 0.001
        gripper_cmd = float(np.asarray(self.data.ctrl[-1]).reshape(-1)[0])
        gripper_ratio = gripper_cmd / max(1e-6, float(self._ctrl_high[-1]))
        gripper_closed = gripper_cmd > 0.85 * float(self._ctrl_high[-1])
        reward_terms = {
            "distance_penalty": -2.5 * distance,
            "distance_progress": 0.0,
            "xy_alignment": 0.0,
            "anti_slide_penalty": 0.0,
            "vertical_alignment": 0.0,
            "pre_grasp_height": 0.0,
            "below_object_penalty": 0.0,
            "contact_any": 0.0,
            "contact_both": 0.0,
            "one_sided_contact_penalty": 0.0,
            "lifted_with_contact": 0.0,
            "lift_progress": 0.0,
            "lifted_without_contact": 0.0,
            "close_gripper_near_object": 0.0,
            "premature_close_penalty": 0.0,
            "stable_grasp_bonus": 0.0,
            "target_height_bonus": 0.0,
            "low_speed_bonus": 0.0,
            "time_penalty": -0.01,
        }
        if self._prev_distance is not None:
            reward_terms["distance_progress"] = 8.0 * float(self._prev_distance - distance)

        reward_terms["xy_alignment"] = 4.0 * (1.0 - min(xy_dist / 0.12, 1.0))

        if self._init_object_xy is not None:
            slide = float(np.linalg.norm(object_pos[:2] - self._init_object_xy))
            on_table = object_z <= (self._table_top_z + self._cube_half + 0.01)
            if on_table:
                reward_terms["anti_slide_penalty"] = -25.0 * slide

        vertical = float(np.clip(np.dot(tool_z, np.array([0.0, 0.0, -1.0])), 0.0, 1.0))
        if xy_dist < self._pre_grasp_xy_gate:
            reward_terms["vertical_alignment"] = 1.0 * vertical

        if xy_dist < self._pre_grasp_xy_gate:
            height_error = abs(tcp_above_obj - self._pre_grasp_height_target)
            normalized_height = 1.0 - min(height_error / self._pre_grasp_height_window, 1.0)
            reward_terms["pre_grasp_height"] = 3.0 * normalized_height
        if tcp_above_obj < 0.01:
            reward_terms["below_object_penalty"] = -3.0 * min(abs(tcp_above_obj - 0.01) / 0.06, 1.0)

        contact_left, contact_right = self._finger_contacts()
        contact_any = contact_left or contact_right
        contact_both = contact_left and contact_right
        if contact_any:
            reward_terms["contact_any"] = 2.0
        if contact_both:
            reward_terms["contact_both"] = 8.0
        elif contact_any:
            reward_terms["one_sided_contact_penalty"] = -2.0 * min(xy_dist / 0.08, 1.0)

        lifted = object_z > 0.22
        if lifted and contact_both:
            reward_terms["lifted_with_contact"] = 15.0
        if contact_both and gripper_closed:
            lift_amount = max(object_z - table_object_z, 0.0)
            reward_terms["lift_progress"] = 20.0 * min(lift_amount / 0.10, 1.0)
        if lifted and (not contact_any):
            reward_terms["lifted_without_contact"] = -10.0

        good_grasp_region = (xy_dist < self._grasp_xy_gate) and (0.02 < tcp_above_obj < 0.10)
        if good_grasp_region:
            reward_terms["close_gripper_near_object"] = 3.0 * gripper_ratio
        elif gripper_ratio > 0.6 and (xy_dist > 0.08 or tcp_above_obj < 0.02):
            reward_terms["premature_close_penalty"] = -3.0 * gripper_ratio

        grasp_pose_ok = (xy_dist < 0.035) and (tcp_above_obj > 0.05) and (vertical > 0.8)
        if contact_both and gripper_closed and grasp_pose_ok and lifted and (distance < 0.07):
            speed = float(np.linalg.norm(object_vel))
            reward_terms["stable_grasp_bonus"] = 120.0
            target_z = self._lift_target_z
            reward_terms["target_height_bonus"] = 10.0 * (
                1.0 - min(abs(object_z - target_z) / 0.05, 1.0)
            )
            reward_terms["low_speed_bonus"] = 5.0 * (1.0 - min(speed / 0.3, 1.0))

        reward = sum(reward_terms.values())
        self._last_reward_terms = reward_terms
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._prev_distance = None
        self._init_object_xy = None

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        mujoco.mj_forward(self.model, self.data)

        xy = None
        if isinstance(options, dict):
            if "object_xy" in options:
                xy = np.asarray(options["object_xy"], dtype=np.float64).reshape(-1)[:2]
            elif "object_pos" in options:
                xy = np.asarray(options["object_pos"], dtype=np.float64).reshape(-1)[:2]
        if xy is None:
            xy = np.array([0.28, 0.0], dtype=np.float64)
        z = self._table_top_z + self._cube_half + 0.001
        self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        tcp_pos = np.asarray(self.data.site("link_tcp").xpos).reshape(-1)
        object_pos = np.asarray(self.data.body(self._object_body).xpos).reshape(-1)
        self._prev_distance = float(np.linalg.norm(object_pos - tcp_pos))
        self._init_object_xy = object_pos[:2].copy()

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        ctrl = np.asarray(action, dtype=np.float32)
        ctrl = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=10)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._get_reward()
        tcp_pos = np.asarray(self.data.site("link_tcp").xpos).reshape(-1)
        object_pos = np.asarray(self.data.body(self._object_body).xpos).reshape(-1)
        self._prev_distance = float(np.linalg.norm(object_pos - tcp_pos))
        object_z = float(self.data.body(self._object_body).xpos[2])
        contact_left, contact_right = self._finger_contacts()
        contact_both = contact_left and contact_right
        gripper_cmd = float(np.asarray(self.data.ctrl[-1]).reshape(-1)[0])
        gripper_closed = gripper_cmd > 0.85 * float(self._ctrl_high[-1])
        success = (object_z > 0.22) and contact_both and gripper_closed
        terminated = (object_z < 0.08) or success
        truncated = self._step_count >= self.max_steps

        info = {
            "success": bool(success),
            "contact_left": bool(contact_left),
            "contact_right": bool(contact_right),
            "object_z": float(object_z),
            "distance": float(self._prev_distance),
            "reward_terms": dict(self._last_reward_terms),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if not self.renderer:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
