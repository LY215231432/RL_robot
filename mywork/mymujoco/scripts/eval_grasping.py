
import os
import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from GraspEnv import GraspEnv
from project_paths import BASE_DIR, VIDEOS_DIR, ensure_artifact_dirs
from sb3_compat import install_numpy_compat_aliases, load_vec_normalize

install_numpy_compat_aliases()
ensure_artifact_dirs()

def _render_first_env(env):
    target = env
    if hasattr(target, "venv"):
        target = target.venv
    if hasattr(target, "envs"):
        return target.envs[0].render()
    return target.render()

def evaluate(model_path, stats_path, output_video, num_episodes):
    # 1. Path to model and normalization statistics
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run train_grasping.py first.")
        return

    # 2. Create the environment (with rendering enabled)
    def make_env():
        return GraspEnv(render_mode="rgb_array")

    env = DummyVecEnv([make_env])

    # 3. Load normalization statistics
    if os.path.exists(stats_path):
        env = load_vec_normalize(stats_path, env)
        # Stop updating normalization statistics during evaluation
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: Normalization statistics not found. Results might be poor.")

    # 4. Load the trained model
    model = PPO.load(model_path, env=env)

    # 5. Run evaluation episodes
    frames = []

    print(f"Starting evaluation for {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Render and store frame
            frame = _render_first_env(env)
            frames.append(frame)
            step += 1
            
        total_reward = float(np.asarray(episode_reward).reshape(-1)[0])
        print(f"Episode {episode + 1} finished: steps={step}, total_reward={total_reward:.2f}")

    # 6. Save video
    print(f"Saving evaluation video to {output_video}...")
    imageio.mimsave(output_video, frames, fps=60)
    print("Video saved successfully!")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "models", "ppo_grasping_final.zip"))
    parser.add_argument("--stats", type=str, default=os.path.join(BASE_DIR, "models", "vec_normalize.pkl"))
    parser.add_argument("--out", type=str, default=os.path.join(VIDEOS_DIR, "eval_grasping.mp4"))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    evaluate(model_path=args.model, stats_path=args.stats, output_video=args.out, num_episodes=args.episodes)
