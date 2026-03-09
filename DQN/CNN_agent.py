import os
import numpy as np
import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 10x10 -> 5x5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


class SmallCombinedExtractor(BaseFeaturesExtractor):
    """
    For Dict obs: {"image": (C,H,W) uint8, "state": (k,) float32}
    Concats CNN(image) + MLP(state).
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        self.image_extractor = SmallGridCNN(observation_space["image"], features_dim=128)

        state_dim = observation_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        self._features_dim = 128 + 64

    def forward(self, obs) -> th.Tensor:
        img_feat = self.image_extractor(obs["image"])
        state_feat = self.state_mlp(obs["state"].float())
        return th.cat([img_feat, state_feat], dim=1)


class SnakeDqnAgent:
    def __init__(self, env):
        self.name = "SnakeDQN_CNN"

        self.model = DQN(
            "MultiInputPolicy",
            env,
            verbose = 1,
            learning_rate = 1e-4,
            buffer_size = 100_000,
            learning_starts = 2_000,
            batch_size = 64,
            gamma = 0.97,
            train_freq = 4,
            target_update_interval = 5_000,
            exploration_fraction = 0.1,
            exploration_final_eps = 0.05,
            policy_kwargs=dict(
                features_extractor_class=SmallCombinedExtractor,
                net_arch=[256, 256],
            ),
        )

    def act(self, observation, deterministic: bool = True) -> int:
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
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path)
        print(f"Model loaded from {path}")