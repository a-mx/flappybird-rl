"""
Created: 2025-07-27
"""
import numpy as np
import flappy_bird_gymnasium
import gymnasium

class Enviroment():
    def __init__(self, env="FlappyBird-v0", render=None, lidar=False):
        self.game = gymnasium.make(env, render_mode=render, use_lidar=lidar)

    def reset(self):
        obs, _ = self.game.reset()
        return obs, _
    
    def get_state(self, obs):
        state = np.array(obs, dtype=float)
        state /= np.max(np.abs(state), axis=0)
        return state

    def step(self, action):
        return self.game.step(action)

if __name__ == '__main__':
    pass