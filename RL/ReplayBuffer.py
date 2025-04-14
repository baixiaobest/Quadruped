import numpy as np
from typing import Dict, List

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Dict] = [{}] * capacity  # Pre-allocate list of empty dicts
        self.ptr = 0  # Pointer to next write position
        self.size = 0  # Current number of transitions

    def add_list(self, transitions: List[Dict]):
        """Add a list of transitions (dictionaries) to the buffer."""
        for transition in transitions:
            self.add(transition)
    
    def add(self, transition: Dict):
        """Add a transition (dictionary) to the buffer."""
        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of transitions uniformly."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return self.size

# Example usage
if __name__ == "__main__":
    # Initialize buffer with capacity 5
    buffer = ReplayBuffer(5)

    # Add transitions (dictionaries)
    for i in range(7):  # Add 7 transitions (will overwrite oldest 2)
        buffer.add({
            "state": np.random.randn(4),
            "action": i % 2,
            "reward": float(i),
            "next_state": np.random.randn(4),
            "done": False
        })

    # Sample a batch
    batch = buffer.sample(3)
    print("Sampled transitions:", batch)