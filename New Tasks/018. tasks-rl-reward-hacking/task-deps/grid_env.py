import numpy as np

class WarehouseEnv:
    def __init__(self):
        # 5x5 Grid
        # S . . . .
        # . X . . . (X = Recharge Station)
        # . . . . .
        # . . . G . (G = Goal/Package)
        self.grid_size = 5
        self.state = (0, 0)
        self.goal = (3, 3)
        self.recharge_station = (1, 1)
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.state = (0, 0)
        self.current_step = 0
        return self._get_state_idx()

    def _get_state_idx(self):
        return self.state[0] * self.grid_size + self.state[1]

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.current_step += 1
        x, y = self.state
        
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.grid_size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.grid_size - 1, y + 1)
        
        self.state = (x, y)
        done = False
        
        # --- THE REWARD HACK ---
        reward = 0
        
        # Flaw: Logic grants +10 EVERY time it lands here.
        # Agent learns to step IN, step OUT, step IN...
        if self.state == self.recharge_station:
            reward += 10 
            
        # Goal
        if self.state == self.goal:
            reward += 100
            done = True
            
        # Time limit
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_state_idx(), reward, done, {}