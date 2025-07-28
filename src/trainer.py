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
        #self.loss_accumulator = 0
        #self.loss_count = 0

    
    def train_step(self, states, actions, rewards, next_states, terminations):

        self.policy_network.train()

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.array(terminations), dtype=torch.bool, device=self.device)


        with torch.no_grad():
            target_values = self.target_network(next_states).max(dim=1).values
            q_target = rewards + (~terminations) * self.gamma * target_values

        current_q = self.policy_network(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        #self.loss_accumulator += loss.item()
        #self.loss_count += 1

        #if self.loss_count % 10000 == 0:
        #    avg_loss = self.loss_accumulator / 10000
        #    print(f"Average loss (last 10000 steps): {avg_loss:.4f}")
        #    self.loss_accumulator = 0 
        #    self.loss_count = 0