import random
import numpy as np

class DoubleIntegrator1D:
    def __init__(self, delta_t=0.05, target_x=0, x_bound=[-10, 10], v_bound=[-5, 5], x_epsilon=0.1, vx_epsilon=0.05, debug=False):
        self.delta_t = delta_t
        self.x = 0
        self.vx = 0
        self.target_x = target_x
        self.x_bound = x_bound
        self.v_bound = v_bound
        self.x_epsilon = x_epsilon
        self.vx_epsilon = vx_epsilon
        self.debug = debug

    def reset(self):
        if not self.debug:
            self.x = random.uniform(self.x_bound[0]/2, self.x_bound[1]/2)
            self.vx = random.uniform(self.v_bound[0]/2, self.v_bound[1]/2) 
        else:
            self.x = 5
            self.vx = 0.5
        return self.get_state()

    def get_state(self):
        return np.array([self.x, self.vx])  

    def step(self, action):
        self.x += self.vx * self.delta_t
        self.vx += action * self.delta_t
        self.vx = max(min(self.vx, self.v_bound[1]), self.v_bound[0])

        reward = -((self.x - self.target_x)**2 + 0.1 * self.vx**2)* self.delta_t

        done = False
        if self.x < self.x_bound[0] or self.x > self.x_bound[1]:
            reward -= 100
            done = True
            print("Out of bounds")

        if np.abs(self.x - self.target_x) < self.x_epsilon and np.abs(self.vx) < self.vx_epsilon:
            reward += 10
            done = True
            print("Target reached")

        return np.array([self.x, self.vx]), reward, done
