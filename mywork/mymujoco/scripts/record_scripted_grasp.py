import argparse
import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if sys.platform.startswith("linux") else "glfw"

import imageio
import mujoco
import numpy as np

from project_paths import BASE_DIR, VIDEOS_DIR, ensure_artifact_dirs


MODEL_PATH = os.path.join(BASE_DIR, "grasping_scene.xml")
OUTPUT_PATH = os.path.join(VIDEOS_DIR, "scripted_grasp_demo.mp4")


class ScriptedGraspRecorder:
    def __init__(self, model_path, framerate=60):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.framerate = framerate
        self.frames = []
        self._next_frame_time = 0.0

        self._arm_ctrl_low = self.model.actuator_ctrlrange[:7, 0].astype(np.float64)
        self._arm_ctrl_high = self.model.actuator_ctrlrange[:7, 1].astype(np.float64)
        self._open_gripper_ctrl = float(self.model.actuator_ctrlrange[7, 0])
        self._closed_gripper_ctrl = float(self.model.actuator_ctrlrange[7, 1])

        self._home_ctrl = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], dtype=np.float64)
        self._grasp_ctrl = np.zeros(7, dtype=np.float64)
        self._arm_ctrl = self._home_ctrl.copy()

        self._object_qpos_adr = int(np.asarray(self.model.joint("object_free").qposadr).reshape(-1)[0])
        self._object_dof_adr = int(np.asarray(self.model.joint("object_free").dofadr).reshape(-1)[0])
        self._object_geom_id = int(np.asarray(self.model.geom("object_geom").id).reshape(-1)[0])
        self._left_pad_geom_id = int(np.asarray(self.model.geom("left_finger_pad_1").id).reshape(-1)[0])
        self._right_pad_geom_id = int(np.asarray(self.model.geom("right_finger_pad_1").id).reshape(-1)[0])
        self._left_pad_geom_ids = {
            int(np.asarray(self.model.geom("left_finger_pad_1").id).reshape(-1)[0]),
            int(np.asarray(self.model.geom("left_finger_pad_2").id).reshape(-1)[0]),
        }
        self._right_pad_geom_ids = {
            int(np.asarray(self.model.geom("right_finger_pad_1").id).reshape(-1)[0]),
            int(np.asarray(self.model.geom("right_finger_pad_2").id).reshape(-1)[0]),
        }
        self._arm_dof_ids = np.arange(7)

    def reset(self, object_xy=(0.28, 0.0)):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        object_z = 0.12 + 0.025 + 0.001
        self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = [object_xy[0], object_xy[1], object_z]
        self.data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.frames = []
        self._next_frame_time = 0.0
        self._arm_ctrl = self._home_ctrl.copy()
        self._capture_frame(force=True)

    def close(self):
        self.renderer.close()

    def _capture_frame(self, force=False):
        if force or self.data.time >= self._next_frame_time:
            self.renderer.update_scene(self.data)
            self.frames.append(self.renderer.render())
            if force:
                self._next_frame_time = self.data.time + 1.0 / self.framerate
            else:
                while self._next_frame_time <= self.data.time:
                    self._next_frame_time += 1.0 / self.framerate

    def _step(self, nstep=1):
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
            self._capture_frame()

    def _set_ctrl(self, arm_ctrl, gripper_ctrl):
        self._arm_ctrl = np.clip(np.asarray(arm_ctrl, dtype=np.float64), self._arm_ctrl_low, self._arm_ctrl_high)
        self.data.ctrl[:7] = self._arm_ctrl
        self.data.ctrl[7] = float(gripper_ctrl)

    def _interpolate_arm(self, target_ctrl, gripper_ctrl, duration, nstep=1):
        start_ctrl = self._arm_ctrl.copy()
        steps = max(1, int(duration / (self.model.opt.timestep * nstep)))
        target_ctrl = np.asarray(target_ctrl, dtype=np.float64)
        for i in range(steps):
            alpha = (i + 1) / steps
            arm_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            self._set_ctrl(arm_ctrl, gripper_ctrl)
            self._step(nstep=nstep)

    def _ramp_gripper(self, start_ctrl, end_ctrl, duration, arm_ctrl=None, nstep=1):
        if arm_ctrl is None:
            arm_ctrl = self._arm_ctrl
        arm_ctrl = np.asarray(arm_ctrl, dtype=np.float64)
        steps = max(1, int(duration / (self.model.opt.timestep * nstep)))
        for i in range(steps):
            alpha = (i + 1) / steps
            grip = (1.0 - alpha) * start_ctrl + alpha * end_ctrl
            self._set_ctrl(arm_ctrl, grip)
            self._step(nstep=nstep)

    def _pad_midpoint(self):
        left = np.asarray(self.data.geom("left_finger_pad_1").xpos, dtype=np.float64).copy()
        right = np.asarray(self.data.geom("right_finger_pad_1").xpos, dtype=np.float64).copy()
        return 0.5 * (left + right)

    def _pad_midpoint_jacobian(self):
        jacp_left = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr_left = np.zeros((3, self.model.nv), dtype=np.float64)
        jacp_right = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr_right = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(self.model, self.data, jacp_left, jacr_left, self._left_pad_geom_id)
        mujoco.mj_jacGeom(self.model, self.data, jacp_right, jacr_right, self._right_pad_geom_id)
        return 0.5 * (jacp_left + jacp_right)[:, self._arm_dof_ids]

    def _contact_both(self):
        left = False
        right = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self._object_geom_id:
                other = contact.geom2
            elif contact.geom2 == self._object_geom_id:
                other = contact.geom1
            else:
                continue
            if other in self._left_pad_geom_ids:
                left = True
            if other in self._right_pad_geom_ids:
                right = True
        return left and right

    def _lift_object(self, lift_height, duration, nstep=5):
        steps = max(1, int(duration / (self.model.opt.timestep * nstep)))
        start_mid = self._pad_midpoint()
        target_mid = start_mid + np.array([0.0, 0.0, lift_height], dtype=np.float64)
        damping = 1e-3
        gain = 0.8
        for i in range(steps):
            alpha = (i + 1) / steps
            desired_mid = (1.0 - alpha) * start_mid + alpha * target_mid
            current_mid = self._pad_midpoint()
            error = desired_mid - current_mid
            jacobian = self._pad_midpoint_jacobian()
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error)
            arm_ctrl = np.asarray(self.data.qpos[:7], dtype=np.float64) + gain * dq
            self._set_ctrl(arm_ctrl, self._closed_gripper_ctrl)
            self._step(nstep=nstep)

    def run(self, object_xy=(0.28, 0.0), lift_height=0.12):
        self.reset(object_xy=object_xy)

        self._set_ctrl(self._home_ctrl, self._open_gripper_ctrl)
        self._step(nstep=60)

        self._interpolate_arm(self._grasp_ctrl, self._open_gripper_ctrl, duration=2.0, nstep=2)
        self._step(nstep=120)

        self._ramp_gripper(
            start_ctrl=self._open_gripper_ctrl,
            end_ctrl=self._closed_gripper_ctrl,
            duration=1.6,
            arm_ctrl=self._grasp_ctrl,
            nstep=2,
        )
        self._step(nstep=120)

        self._lift_object(lift_height=lift_height, duration=2.8, nstep=5)
        self._step(nstep=180)

        object_pos = np.asarray(self.data.body("object_to_grasp").xpos, dtype=np.float64).copy()
        success = bool(object_pos[2] > 0.22 and self._contact_both())
        return {
            "success": success,
            "object_pos": object_pos,
            "pad_midpoint": self._pad_midpoint(),
            "frame_count": len(self.frames),
            "sim_time": float(self.data.time),
        }

    def save_video(self, output_path):
        imageio.mimsave(output_path, self.frames, fps=self.framerate)


def record_scripted_grasp(output_path, object_xy=(0.28, 0.0), lift_height=0.12):
    ensure_artifact_dirs()
    recorder = ScriptedGraspRecorder(MODEL_PATH)
    try:
        result = recorder.run(object_xy=object_xy, lift_height=lift_height)
        recorder.save_video(output_path)
    finally:
        recorder.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a deterministic xArm7 grasp demo.")
    parser.add_argument("--out", type=str, default=OUTPUT_PATH)
    parser.add_argument("--object-x", type=float, default=0.28)
    parser.add_argument("--object-y", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.12)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result = record_scripted_grasp(
        output_path=args.out,
        object_xy=(args.object_x, args.object_y),
        lift_height=args.lift_height,
    )
    print(
        "Scripted grasp finished:",
        f"success={result['success']}",
        f"object_pos={np.round(result['object_pos'], 4).tolist()}",
        f"frames={result['frame_count']}",
        f"sim_time={result['sim_time']:.2f}s",
    )
