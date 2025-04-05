import numpy as np

class DiscountedSumRunningStats:
    def __init__(self, beta=0.99, gamma=0.99):
        self.beta = beta  # For EMA
        self.gamma = gamma  # For discounted sum

        self.discounted_sum = 0
        self.discounted_sum_mean = 0
        self.var = 0
        self.count = 0

    def update(self, x):
        self.discounted_sum = x + self.gamma * self.discounted_sum 
        self.discounted_sum_mean = self.discounted_sum_mean * self.beta + (1 - self.beta) * self.discounted_sum
        self.var = self.beta * self.var + (1 - self.beta) * (self.discounted_sum - self.discounted_sum_mean)**2
        self.count += 1

    def get_std(self):
        return np.sqrt(self.var / self.count)  # Population std
    
    def get_mean(self):
        return self.discounted_sum_mean
    
    def get_count(self):
        return self.count