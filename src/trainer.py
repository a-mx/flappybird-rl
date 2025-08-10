import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Trainer:
    def __init__(self, target_network, policy_network, lr, gamma):
        self.lr = lr

        self.target_network = target_network
        self.policy_network = policy_network

        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(policy_network.parameters(), lr=self.lr)

        self.loss_fn = nn.SmoothL1Loss()


    
    def train_step(self, states, actions, rewards, next_states, terminations):

        self.policy_network.train()

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.array(terminations), dtype=torch.bool, device=self.device)

        with torch.no_grad():

            next_actions = self.policy_network(next_states).argmax(dim=1, keepdim=True)

            next_q_value = self.target_network(next_states).gather(1, next_actions).squeeze()

            q_target = rewards + (~terminations) * self.gamma * next_q_value

        current_q = self.policy_network(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()