import random
import numpy as np

class DoubleIntegrator1D:
    def __init__(self, 
                 delta_t=0.05, 
                 target_x=0, 
                 goal_reward=1e2, 
                 out_of_bound_penalty=1e2,
                 x_bound=[-10, 10], 
                 x_init_bound=[-5, 5],
                 v_bound=[-5, 5], 
                 v_init_bound=[-1, 1],
                 v_penalty=0.1, 
                 action_range=[-1, 1],
                 time_penalty=0.01, 
                 action_penalty=0.1, 
                 action_change_panelty=0.1, 
                 action_smooth=0.7, 
                 x_epsilon=0.1, 
                 vx_epsilon=0.1,
                 noise={'x': 0, 'vx': 0, 'action': 0},
                 bias={'x':0, 'vx': 0, 'action': 0},
                 random_bias=None,
                 debug=False):
        self.delta_t = delta_t
        self.x = 0
        self.vx = 0
        self.target_x = target_x
        self.goal_reward = goal_reward
        self.out_of_bound_penalty = out_of_bound_penalty
        self.v_penalty = v_penalty
        self.action_range = action_range
        self.time_penalty = time_penalty
        self.action_penalty = action_penalty
        self.action_change_panelty = action_change_panelty
        self.action_smooth = action_smooth
        self.x_bound = x_bound
        self.x_init_bound = x_init_bound
        self.v_bound = v_bound
        self.v_init_bound = v_init_bound
        self.x_epsilon = x_epsilon
        self.vx_epsilon = vx_epsilon
        self.noise = noise
        self.bias = bias
        self.random_bias = random_bias
        self.debug = debug
        self.prev_action = 0

    def reset(self):
        if self.debug:
            self.x = 5
            self.vx = 0.5
        else:
            self.x = random.uniform(self.x_init_bound[0], self.x_init_bound[1])
            self.vx = random.uniform(self.v_init_bound[0], self.v_init_bound[1]) 

        if self.random_bias:
            self.bias = {
                'x': np.random.uniform(-self.random_bias['x'], self.random_bias['x']), 
                'vx': np.random.uniform(-self.random_bias['vx'], self.random_bias['vx']), 
                'action': np.random.uniform(-self.random_bias['action'], self.random_bias['action']), 
            }

        self.prev_action = 0
        
        return self.get_state(), 'reset'
    
    def set_state(self, x, vx, action=0):
        self.x = x
        self.vx = vx
        self.prev_action = action

    def get_state(self):
        return np.array([np.random.normal(self.x, self.noise['x'] + self.bias['x']), 
                         np.random.normal(self.vx, self.noise['vx'] + self.bias['vx'])])  
    
    def get_action(self):
        return self.prev_action

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]

        percent = (action + 1) / 2
        action = percent * (self.action_range[1] - self.action_range[0]) + self.action_range[0]

        action = np.random.normal(action, self.noise['action']) + self.bias['action']
        self.prev_action = (1-self.action_smooth) * self.prev_action + self.action_smooth * action
        self.x += self.vx * self.delta_t
        self.vx += self.prev_action * self.delta_t
        self.vx = max(min(self.vx, self.v_bound[1]), self.v_bound[0])

        x_penalty = (self.x - self.target_x)**2
        v_penalty = self.v_penalty * self.vx**2
        action_penalty = self.action_penalty + action**2
        action_change_penalty = self.action_change_panelty * (action - self.prev_action)**2

        reward = -(x_penalty + v_penalty + self.time_penalty + action_penalty + action_change_penalty)* self.delta_t

        terminated = False
        truncated = False
        info = ""
        if self.x < self.x_bound[0] or self.x > self.x_bound[1]:
            reward -= self.out_of_bound_penalty
            terminated = True
            info = "Out of bounds"

        if self.goal_reached():
            reward += self.goal_reward
            terminated = True
            info = "Goal reached"

        return np.array([self.x, self.vx]), reward, terminated, truncated, info
    
    def goal_reached(self):
        return np.abs(self.x - self.target_x) < self.x_epsilon and np.abs(self.vx) < self.vx_epsilon
