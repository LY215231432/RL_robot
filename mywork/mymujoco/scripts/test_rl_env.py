
from GraspEnv import GraspEnv
import os
import imageio
from project_paths import VIDEOS_DIR, ensure_artifact_dirs

def test_random_actions():
    """Tests the GraspEnv with random actions and records a video."""
    ensure_artifact_dirs()

    # Instantiate the environment
    env = GraspEnv(render_mode="rgb_array")

    # Reset the environment
    obs, info = env.reset()

    frames = []
    total_reward = 0
    terminated = False
    truncated = False
    max_steps = 500

    print("Running environment with random actions...")

    for step in range(max_steps):
        if terminated or truncated:
            break

        # Sample a random action from the action space
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        total_reward += reward

        # Render and store the frame
        frame = env.render()
        frames.append(frame)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{max_steps}, Current Total Reward: {total_reward:.2f}")

    # --- Save video file ---
    output_path = os.path.join(VIDEOS_DIR, "rl_random_actions.mp4")
    print(f"\nSaving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=env.metadata["render_fps"])
    print("Video saved successfully!")
    print(f"Final Total Reward: {total_reward:.2f}")

    # Clean up
    env.close()

if __name__ == "__main__":
    test_random_actions()
