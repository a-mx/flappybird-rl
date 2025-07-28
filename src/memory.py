"""
Created: 2025-07-26
"""
from collections import deque
import random

class Memory():
    def __init__(self, max_length):
        self.memory = deque(maxlen=max_length)

    def append(self, x):
        self.memory.append(x)

    def sample(self, size):
        size = min(size, len(self.memory))
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    pass