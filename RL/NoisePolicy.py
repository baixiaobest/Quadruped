import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from RL.OUNoise import OUNoise
import numpy as np

class GaussianNoisePolicy:
    """Adds Gaussian noise to the actions of a base policy using pre-allocated tensor cache."""
    def __init__(self, base_policy, noise_scale, noise_clip, action_dim, cache_size=1000):
        self.base_policy = base_policy
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.action_dim = action_dim
        self.cache_size = cache_size
        self.cache = torch.empty((cache_size, action_dim))
        self.cache_index = 0
        self.regenerate_noise()
    
    def regenerate_noise(self):
        """Precompute Gaussian noise samples into a tensor cache without computation graph."""
        with torch.no_grad():
            self.cache = torch.normal(mean=0, std=self.noise_scale, 
                                    size=(self.cache_size, self.action_dim))
            if self.noise_clip is not None:
                self.cache = torch.clamp(self.cache, -self.noise_clip, self.noise_clip)
        self.cache_index = 0
    
    def __call__(self, state):
        """Compute noisy action, handling both batched and non-batched states."""
        # Handle input dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
            batched = False
        else:
            batched = True

        action = self.base_policy(state)  # (batch_size, action_dim)
        batch_size = action.size(0)

        # Preallocate noise tensor
        noise = torch.empty((batch_size, self.action_dim))

        # Determine how much noise to take from the cache
        available = min(batch_size, self.cache_size - self.cache_index)

        with torch.no_grad():
            if available > 0:
                # Retrieve cached noise
                cached_noise = self.cache[self.cache_index : self.cache_index + available]
                noise[:available] = cached_noise
                self.cache_index += available

            remaining = batch_size - available
            if remaining > 0:
                # Generate remaining noise in bulk
                new_noise = torch.normal(mean=0, std=self.noise_scale, 
                                       size=(remaining, self.action_dim))
                if self.noise_clip is not None:
                    new_noise = torch.clamp(new_noise, -self.noise_clip, self.noise_clip)
                noise[available:] = new_noise

        # Add noise to action (part of computation graph)
        noisy_action = action + noise
        noisy_action = torch.clamp(noisy_action, -1, 1)

        # Remove batch dimension if input was non-batched
        if not batched:
            noisy_action = noisy_action.squeeze(0)

        return noisy_action

class OUNoisePolicy:
    """Adds Ornstein-Uhlenbeck noise to the actions of a base policy with state-aware caching."""
    def __init__(self, base_policy, action_dim, noise_scale=0.2, dt=0.01, theta=0.15, 
                 noise_clip=None, cache_size=1000):
        self.base_policy = base_policy
        self.action_dim = action_dim
        self.noise_scale = noise_scale
        self.dt = dt
        self.theta = theta
        self.noise_clip = noise_clip
        self.cache_size = cache_size
        self.ou_noise = OUNoise(theta=theta, mu=np.zeros(action_dim), sigma=noise_scale, dt=dt)
        self.cache = torch.empty((cache_size, action_dim))
        self.cache_index = 0
        self.regenerate_noise()
    
    def regenerate_noise(self):
        """Precompute OU noise sequence into a tensor cache using numpy backend."""
        self.ou_noise.reset()
        noise_cache = np.zeros((self.cache_size, self.action_dim))
        for i in range(self.cache_size):
            noise = self.ou_noise.sample()
            if self.noise_clip is not None:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                self.ou_noise.set_noise(noise)
            noise_cache[i] = noise
        self.cache = torch.tensor(noise_cache, dtype=torch.float32)
        self.cache_index = 0
    
    def __call__(self, state):
        """Compute noisy action, handling both batched and non-batched states."""
        # Handle input dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
            batched = False
        else:
            batched = True

        action = self.base_policy(state)  # (batch_size, action_dim)
        batch_size = action.size(0)

        # Preallocate noise tensor
        noise = torch.empty((batch_size, self.action_dim))

        # Determine available cache
        available = min(batch_size, self.cache_size - self.cache_index)

        with torch.no_grad():
            if available > 0:
                cached_noise = self.cache[self.cache_index : self.cache_index + available]
                noise[:available] = cached_noise

                # Update OU state to the last used cache sample and invalidate remaining cache
                last_noise = cached_noise[-1].cpu().numpy()  # Move to CPU for numpy operations
                self.ou_noise.set_noise(last_noise)
                self.cache_index = self.cache_size  # Invalidate remaining cache

            remaining = batch_size - available
            if remaining > 0:
                # Generate remaining noise samples in sequence
                for i in range(remaining):
                    noise_sample = self.ou_noise.sample()
                    if self.noise_clip is not None:
                        noise_sample = np.clip(noise_sample, -self.noise_clip, self.noise_clip)
                        self.ou_noise.set_noise(noise_sample)
                    # Convert to tensor and assign to noise tensor
                    noise_tensor = torch.tensor(noise_sample, dtype=torch.float32)
                    noise[available + i] = noise_tensor

        # Add noise to action (part of computation graph)
        noisy_action = action + noise
        noisy_action = torch.clamp(noisy_action, -1, 1)

        if not batched:
            noisy_action = noisy_action.squeeze(0)

        return noisy_action

class UniformPolicy:
    """Generates uniform random actions using a precomputed tensor cache."""
    def __init__(self, action_dim, cache_size=1000):
        self.action_dim = action_dim
        self.cache_size = cache_size
        self.cache = torch.empty((cache_size, action_dim))
        self.cache_index = 0
        self.regenerate_noise()
    
    def regenerate_noise(self):
        """Precompute uniform random actions into a tensor cache without computation graph."""
        with torch.no_grad():
            self.cache = torch.rand((self.cache_size, self.action_dim)) * 2.0 - 1.0
            self.cache = torch.clamp(self.cache, -1, 1)
        self.cache_index = 0
    
    def __call__(self, state):
        """Return uniform actions matching input state batch size."""
        # Handle input dimensions
        if state is not None:
            if state.dim() == 1:
                batch_size = 1
                batched = False
            else:
                batch_size = state.size(0)
                batched = True
        else:
            # Handle case where state is None (unlikely)
            batch_size = 1
            batched = False

        # Preallocate noise tensor
        noise = torch.empty((batch_size, self.action_dim))

        # Determine available cache
        available = min(batch_size, self.cache_size - self.cache_index)

        with torch.no_grad():
            if available > 0:
                cached_noise = self.cache[self.cache_index : self.cache_index + available]
                noise[:available] = cached_noise
                self.cache_index += available

            remaining = batch_size - available
            if remaining > 0:
                # Generate remaining noise in bulk
                new_noise = torch.rand((remaining, self.action_dim)) * 2.0 - 1.0
                new_noise = torch.clamp(new_noise, -1, 1)
                noise[available:] = new_noise

        # Remove batch dimension if input was non-batched
        if not batched:
            noise = noise.squeeze(0)

        return noise

def main():
    """Test function for noise policies with batched/non-batched inputs."""
    action_dim = 2
    state_dim = 3
    cache_size = 5
    
    # Dummy base policy that returns zero actions
    class ZeroPolicy(torch.nn.Module):
        def forward(self, state):
            if state.dim() == 1:
                return torch.zeros(action_dim)
            else:
                return torch.zeros(state.size(0), action_dim)
    
    zero_policy = ZeroPolicy()
    
    # Test GaussianNoisePolicy
    print("Testing GaussianNoisePolicy:")
    gaussian_policy = GaussianNoisePolicy(zero_policy, noise_scale=0.5, noise_clip=0.3, 
                                        action_dim=action_dim, cache_size=cache_size)
    # Single state
    state = torch.randn(state_dim)
    action = gaussian_policy(state)
    print(f"Single state output shape: {action.shape}, value:\n{action}")
    # Batched state
    states = torch.randn(3, state_dim)
    actions = gaussian_policy(states)
    print(f"Batched states output shape: {actions.shape}, values:\n{actions}")
    
    # Test OUNoisePolicy
    print("\nTesting OUNoisePolicy:")
    ou_policy = OUNoisePolicy(zero_policy, action_dim=action_dim, cache_size=cache_size)
    # Single state
    action = ou_policy(state)
    print(f"Single state output shape: {action.shape}, value:\n{action}")
    # Batched state
    actions = ou_policy(states)
    print(f"Batched states output shape: {actions.shape}, values:\n{actions}")
    
    # Test UniformPolicy
    print("\nTesting UniformPolicy:")
    uniform_policy = UniformPolicy(action_dim=action_dim, cache_size=cache_size)
    # Single state (state is dummy input)
    action = uniform_policy(torch.randn(state_dim))
    print(f"Single state output shape: {action.shape}, value:\n{action}")
    # Batched state
    actions = uniform_policy(torch.randn(3, state_dim))
    print(f"Batched states output shape: {actions.shape}, values:\n{actions}")

if __name__ == "__main__":
    main()