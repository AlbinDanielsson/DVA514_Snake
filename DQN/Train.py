#Run by: python Train.py
#maybe add: --timesteps 300000 --render True
import os
import argparse
import time
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from Environment import SnakeDqnEnv
from DQN_agent import SnakeDqnAgent

def make_env(render):
    def _thunk():
        env = SnakeDqnEnv()
        env.visible = render
        return Monitor(env)
    return _thunk

def train_snake(timesteps, render, save_path):
    env = DummyVecEnv([make_env(render)])

    agent = SnakeDqnAgent(env)

    start_time = time.time()
    agent.train(env, timesteps, save_path)
    dt = time.time() - start_time

    final_dir = save_path or os.path.join("models", agent.name)
    print(f"Training completed in {dt:.2f}s")
    print(f"Models saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate snake DQN agent")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    train_snake(args.timesteps, args.render, args.save_path)