import torch
import argparse
from src.agent import Agent
from src.enviroment import Enviroment
import numpy as np

def train(path=None):
    agent = Agent()
    agent.train = True
    steps = 0

    game_reward = 0

    if path:
        print(f"Loading model from {path}...")
        agent.policy_network.load_state_dict(torch.load(path))
        agent.update_target_network()

    env = Enviroment()
    obs, _ = env.reset()

    while True:

        state = env.get_state(obs)

        action = agent.get_action(state)

        obs, reward, terminated, truncated, info = env.step(action)

        score = info['score']

        state_new = env.get_state(obs)

        agent.memory.append((state, action, reward, state_new, terminated))

        if len(agent.memory) > agent.batch_size:
            agent.train_step()

        steps += 1
        game_reward += reward

        if steps % agent.update_threshold == 0: 
            agent.update_target_network()
            steps = 0

        if terminated or truncated:
            agent.n_games += 1
            agent.scores.append(score)

            if agent.reward_record < game_reward:
                agent.reward_record = game_reward
                print(f'New best reward: {agent.reward_record:.2f}')
                agent.policy_network.save()
            
            game_reward = 0    

            if score > agent.score_record:
                agent.score_record = score

            if agent.n_games % agent.print_info == 0:
                mean_score = np.mean(agent.scores)
                print(f'Game {agent.n_games}, Mean Score: {mean_score:.2f}, Record: {agent.score_record}')

            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Flappy Bird agent.")
    parser.add_argument("--path", type=str, help="Path to the saved model to continue training.")
    args = parser.parse_args()
    train(path=args.path)