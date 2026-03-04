#Run by:
#python Test.py --model_path models/SnakeDQN/SnakeDQN_final_model --render True

import os
import argparse

from CNN_Environment import SnakeDqnEnv
from CNN_agent import SnakeDqnAgent

def test_snake(model_path, num_episodes, render):
    env = SnakeDqnEnv()
    env.visible = render
    env.isTraining = False
    agent = SnakeDqnAgent(env)

    if not model_path.endswith(".zip"):
        model_path += ".zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    agent.load(model_path)

    total_reward = 0.0
    total_apples = 0.0

    for ep in range(num_episodes):
        print(f"\nRunning episode {ep+1}/{num_episodes}")
        obs, info = env.reset()
        env.visible = render
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            action = agent.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

        apples = info.get("length", 3) - 3
        print(f"Episode {ep+1} finished after {steps} steps | reward={ep_reward:.2f} | apples={apples}")
        total_reward += ep_reward
        total_apples += apples

    print("\nAverage number of apples:", total_apples / max(num_episodes, 1))
    env.close()
    print("Testing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test snake DQN agent")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--render", type=bool, default = False)
    parser.add_argument("--episodes", type=int, default=5)

    args = parser.parse_args()
    #args.episodes = 100

    test_snake(args.model_path, args.episodes, args.render)