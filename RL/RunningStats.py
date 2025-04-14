import numpy as np

class RunningStats:
    def __init__(self, beta=0.95):
        self.beta = beta
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0

    def update(self, x):
        # Update mean
        new_mean = self.beta * self.mean + (1 - self.beta) * x
        # Update variance using previous mean (unbiased)
        new_variance = self.beta * self.variance + (1 - self.beta) * (x - self.mean) ** 2
        self.mean = new_mean
        self.variance = new_variance
        self.count += 1

    def get_mean(self):
        return self.mean 

    def get_std(self):
        return np.sqrt(self.variance + 1e-5)
    
    def get_count(self):
        return self.count