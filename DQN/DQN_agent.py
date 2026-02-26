import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


class SnakeDqnAgent:
    def __init__(self, env):
        self.name = "SnakeDQN"

        #Modify these!
        self.model = DQN(
            "MlpPolicy",
            env,
            verbose = 1,
            learning_rate = 1e-4,
            buffer_size = 100_000,
            learning_starts = 5_000,
            batch_size = 64,
            gamma = 0.97,
            train_freq = 4,
            target_update_interval = 10_000,
            exploration_fraction = 0.15,
            exploration_final_eps = 0.01,
            policy_kwargs=dict(net_arch= (128, 128)),
        )

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        if self.model is None:
            return 1  # default: straight
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def train(self, env, total_timesteps, save_path) -> None:
        self.model.set_env(env)

        if save_path is None:
            save_path = os.path.join("models", self.name)
        os.makedirs(save_path, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=500_000,
            save_path=save_path,
            name_prefix=f"{self.name}_model",
            save_replay_buffer=True,
            save_vecnormalize=False,
        )

        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.save(os.path.join(save_path, f"{self.name}_final_model"))

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save (train or load first).")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path)
        print(f"Model loaded from {path}")