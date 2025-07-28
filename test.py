"""
Created: 2025-07-28
"""
import torch
from src.agent import Agent
from src.enviroment import Enviroment
import argparse

def test(path=None):
    agent = Agent()
    if path:
        print(f"Loading model from {path}...")
        agent.policy_network.load_state_dict(torch.load(path))
        agent.update_target_network()

    env = Enviroment(render="human")
    obs, _ = env.reset()

    while True:

        state = env.get_state(obs)
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Flappy Bird agent model.")
    parser.add_argument("--path", type=str, help="Path to the saved model")
    args = parser.parse_args()
    test(path=args.path)