import numpy as np
from typing import Dict

class ReplayBuffer:
    def __init__(self, capacity: int, structure: Dict[str, tuple]):
        """
        Initialize ReplayBuffer with pre-allocated numpy arrays.
        
        :param capacity: Maximum number of transitions stored
        :param structure: Dictionary specifying buffer structure {
            'key_name': (dtype, shape), 
            e.g. 'state': (np.float32, (4,))
        }
        """
        self.capacity = capacity
        self.structure = structure
        self.ptr = 0  # Current write pointer
        self.size = 0  # Current buffer size
        
        # Pre-allocate numpy arrays for each key
        self.data = {}
        for key, (dtype, shape) in structure.items():
            self.data[key] = np.empty((capacity, *shape), dtype=dtype)
    
    def add(self, transitions: Dict[str, np.ndarray]):
        """
        Add batch of transitions to buffer using vectorized operations.
        
        :param transitions: Dictionary of numpy arrays where each array's
        first dimension is the batch size
        """
        batch_size = transitions[next(iter(transitions))].shape[0]
        start_ptr = self.ptr
        
        if self.ptr + batch_size <= self.capacity:
            end_ptr = self.ptr + batch_size
            for key in self.data:
                self.data[key][start_ptr:end_ptr] = transitions[key]
            self.ptr = end_ptr
        else:
            # Handle wrap-around case
            remaining = self.capacity - start_ptr
            for key in self.data:
                # Fill end of buffer
                self.data[key][start_ptr:self.capacity] = transitions[key][:remaining]
                # Fill beginning of buffer
                self.data[key][0:batch_size - remaining] = transitions[key][remaining:]
            self.ptr = batch_size - remaining
        
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of transitions using vectorized indexing"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {key: arr[indices] for key, arr in self.data.items()}
    
    def __len__(self):
        return self.size

# Updated example usage
if __name__ == "__main__":
    # Define buffer structure (user-configurable)
    buffer_structure = {
        'state': (np.float32, (4,)),
        'action': (np.int32, ()),
        'reward': (np.float32, ()),
        'next_state': (np.float32, (4,)),
        'done': (np.bool_, ())
    }
    
    # Initialize buffer with capacity 5
    buffer = ReplayBuffer(5, buffer_structure)

    # Create batch of 7 transitions (will overwrite oldest 2)
    batch_size = 7
    transitions = {
        'state': np.random.randn(batch_size, 4).astype(np.float32),
        'action': np.arange(batch_size) % 2,
        'reward': np.arange(batch_size, dtype=np.float32),
        'next_state': np.random.randn(batch_size, 4).astype(np.float32),
        'done': np.zeros(batch_size, dtype=np.bool_)
    }
    
    # Add entire batch in one vectorized operation
    buffer.add(transitions)
    print(f"Buffer size after adding 7: {len(buffer)}")  # Should be 5

    # Sample and inspect
    batch = buffer.sample(3)
    print("\nSampled states shape:", batch['state'].shape)
    print("Sampled actions:", batch['action'])
    print("Sampled rewards:", batch['reward'])