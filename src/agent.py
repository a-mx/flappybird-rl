import torch
import random
from collections import deque
from src.model import Network
from src.trainer import Trainer
from src.memory import Memory
class Agent:
    def __init__(self):
        self.n_games = 0

        self.epsilon = 0.1
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.00001

        self.gamma = 0.98

        self.batch_size = 128

        self.max_memory = 100_000
        self.memory = Memory(self.max_memory)

        self.policy_network = Network(12, 256, 2)
        self.target_network = Network(12,256,2)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.scores = deque(maxlen=100)

        self.reward_record = -99
        self.score_record = 0

        self.update_threshold = 500

        self.print_info = 100

        self.lr = 1e-4
        self.trainer = Trainer(target_network=self.target_network,
                               policy_network=self.policy_network,
                               lr=self.lr,gamma=self.gamma)
        self.train = False


    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
    

    def train_step(self):
        sample = self.memory.sample(self.batch_size)
        if len(sample):
            states, actions, rewards, new_states, terminates = zip(*sample)
            self.trainer.train_step(states, actions, rewards, new_states, terminates)
    

    def get_action(self, state):    
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if (random.random() < self.epsilon) and self.train:
            action = random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float, device=self.trainer.device).unsqueeze(0)
            with torch.no_grad():
                prediction = self.policy_network(state)
                action = prediction.argmax().item()
        return action


if __name__ == "__main__":
    pass