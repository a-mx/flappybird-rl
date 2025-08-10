"""
Created: 2025-07-26
"""
from collections import deque
import random
import numpy as np

class Memory():
    def __init__(self, max_length, alpha=0.6, beta_start=0.4, beta_increment=0.001, epsilon=1e-5):
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)
        self.priorities = deque(maxlen=max_length)
        self.alpha = alpha  
        self.beta = beta_start  
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def append(self, x):
        self.memory.append(x)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [], [], []
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        priorities = np.array(self.priorities)
        priorities = np.maximum(priorities, 1e-6)

        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=True)
        
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
        
        samples = [self.memory[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    pass