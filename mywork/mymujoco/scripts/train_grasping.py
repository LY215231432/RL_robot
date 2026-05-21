
import os
import argparse
import tempfile
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from GraspEnv import GraspEnv
from project_paths import BASE_DIR
from sb3_compat import install_numpy_compat_aliases, load_vec_normalize

install_numpy_compat_aliases()

def build_env(log_dir, seed, num_envs, stats_path=None):
    def make_env(rank, env_seed):
        def _init():
            env = GraspEnv()
            env.reset(seed=env_seed + rank)
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}.csv"))
            return env
        return _init

    if num_envs == 1:
        env = DummyVecEnv([make_env(0, seed)])
    else:
        env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])

    if stats_path:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
        env = load_vec_normalize(stats_path, env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded normalization stats from {stats_path}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return env


def train(total_timesteps, save_dir, log_dir, seed, num_envs, load_model_path=None, load_stats_path=None, device="cpu", reset_num_timesteps=None):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if load_model_path and not load_stats_path:
        print("Warning: continuing training without VecNormalize stats. Learning quality may drop.")

    env = build_env(log_dir=log_dir, seed=seed, num_envs=num_envs, stats_path=load_stats_path)

    if load_model_path:
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {load_model_path}")
        print(f"Loading existing PPO model from {load_model_path}")
        model = PPO.load(load_model_path, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log=log_dir,
            device=device,
        )

    if reset_num_timesteps is None:
        reset_num_timesteps = load_model_path is None

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_grasping"
    )

    print(f"Starting training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_model_path = os.path.join(save_dir, "ppo_grasping_final")
    stats_output_path = os.path.join(save_dir, "vec_normalize.pkl")
    model.save(final_model_path)
    env.save(stats_output_path)
    env.close()
    print(f"Training finished! Model saved to {final_model_path}.zip")
    print(f"Normalization stats saved to {stats_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "models"))
    parser.add_argument("--log-dir", type=str, default=os.path.join(BASE_DIR, "logs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-model", type=str, default=None, help="Path to an existing PPO .zip checkpoint to continue training from.")
    parser.add_argument("--load-stats", type=str, default=None, help="Path to VecNormalize statistics (.pkl) matching the loaded model.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device, e.g. cpu, cuda, or auto.")
    parser.add_argument("--reset-timesteps", action="store_true", help="Reset the timestep counter when continuing training.")
    args = parser.parse_args()
    load_model_path = os.path.abspath(args.load_model) if args.load_model else None
    load_stats_path = os.path.abspath(args.load_stats) if args.load_stats else None
    if load_model_path and not load_stats_path:
        candidate_stats_path = os.path.join(os.path.dirname(load_model_path), "vec_normalize.pkl")
        if os.path.exists(candidate_stats_path):
            load_stats_path = candidate_stats_path

    train(
        total_timesteps=args.steps,
        save_dir=os.path.abspath(args.save_dir),
        log_dir=os.path.abspath(args.log_dir),
        seed=args.seed,
        num_envs=int(args.n_envs),
        load_model_path=load_model_path,
        load_stats_path=load_stats_path,
        device=args.device,
        reset_num_timesteps=args.reset_timesteps if args.load_model else None,
    )
