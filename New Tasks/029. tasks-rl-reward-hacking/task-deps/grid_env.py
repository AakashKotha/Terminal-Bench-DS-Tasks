import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (4, 4)
        self.trap = (2, 2)
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.state = (0, 0)
        self.current_step = 0
        return self._get_state_index()

    def _get_state_index(self):
        return self.state[0] * self.size + self.state[1]

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        x, y = self.state
        
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Right
            y = min(self.size - 1, y + 1)
        elif action == 2: # Down
            x = min(self.size - 1, x + 1)
        elif action == 3: # Left
            y = max(0, y - 1)
            
        self.state = (x, y)
        self.current_step += 1
        
        done = False
        state_idx = self._get_state_index()
        
        # Check Goal
        if self.state == self.goal:
            done = True
            
        # Check Max Steps (Battery Dead)
        if self.current_step >= self.max_steps:
            done = True
            
        return state_idx, self.state, done