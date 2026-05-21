import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if sys.platform.startswith("linux") else "glfw"

import imageio
import mujoco

from project_paths import BASE_DIR, VIDEOS_DIR, ensure_artifact_dirs


MODEL_PATH = os.path.join(BASE_DIR, "ufactory_xarm7_mjcf", "scene.xml")
OUTPUT_PATH = os.path.join(VIDEOS_DIR, "simulation_video.mp4")


def record_simulation(duration=5, framerate=60):
    ensure_artifact_dirs()
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    print("Running headless MuJoCo simulation and recording video...")
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            frames.append(renderer.render())

    print(f"Saving simulation video to {OUTPUT_PATH}...")
    imageio.mimsave(OUTPUT_PATH, frames, fps=framerate)
    print("Video saved successfully.")


if __name__ == "__main__":
    try:
        record_simulation()
    except FileNotFoundError:
        print(f"Error: model file not found at '{MODEL_PATH}'")
    except Exception as exc:
        print(f"Failed to load or run the model: {exc}")
