import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy import std
import numpy as np
import torch
from RL.PolicyNetwork import ActionType
import itertools
from typing import Dict, List
import threading

class SimpleRollout:
    def __init__(self, env, action_type) -> None:
        self.env = env
        self.action_type = action_type

    def rollout(self, num_steps, policy, state=None, exact=True, max_steps_per_episode=1000):
        if state is None:
            state, _ = self.env.reset()

        transitions = []
        steps_in_episode = 0

        for steps in itertools.count():

            state_t = torch.tensor(state, dtype=torch.float32)
            if self.action_type == ActionType.DETERMINISTIC_CONTINUOUS:
                action = policy(state_t).detach().numpy()
            elif self.action_type == ActionType.GAUSSIAN:
                mean, std = policy(state_t)
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                action = action.detach().numpy()

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_max_step_reached = steps_in_episode >= max_steps_per_episode

            curr_transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'step_in_episode': steps_in_episode,
                'episode_max_step_reached': episode_max_step_reached,
            }

            # Gaussian policy has additional information
            if self.action_type == ActionType.GAUSSIAN:
                curr_transition['action_log_prob'] = action_log_prob.item()
                curr_transition['mean'] = mean.detach().numpy()
                curr_transition['std'] = std.detach().numpy()

            transitions.append(curr_transition)

            state = next_state

            # Terminate current episode
            
            if done or episode_max_step_reached:
                state, _ = self.env.reset()
                steps_in_episode = 0
    
            # Terminate current rollout
            if (exact or done or episode_max_step_reached) and steps > num_steps:
                break

            steps_in_episode += 1

        return transitions
    
    def eval_rollout(self, n_episode, policy, max_steps_per_episode=1000, gamma=0.99):
        episode_returns = []
        episode_length = []

        for episode in range(n_episode):

            state, _ = self.env.reset()
            rewards = []
            for step in range(max_steps_per_episode):

                state_t = torch.tensor(state, dtype=torch.float32)
                if policy.get_action_type() == ActionType.DETERMINISTIC_CONTINUOUS:
                    action = policy(state_t).detach().numpy()
                elif policy.get_action_type() == ActionType.GAUSSIAN:
                    mean, std = policy(state_t)
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    action = action.detach().numpy()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                rewards.append(reward)

                state = next_state

                if done:
                    break

            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
            episode_returns.append(G)
            episode_length.append(len(rewards))

        return episode_returns, episode_length

class FastRollout:
    def __init__(self, env, action_type, state_shape: tuple, action_shape: tuple):
        """
        Optimized rollout class with preallocated numpy arrays
        
        :param env: Environment to interact with
        :param action_type: Type of action policy (DETERMINISTIC_CONTINUOUS/GAUSSIAN)
        :param state_shape: Shape of state vectors
        :param action_shape: Shape of action vectors
        """
        self.env = env
        self.action_type = action_type
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.state = np.zeros(state_shape)
        
        # Initialize transition structure compatible with ReplayBuffer
        self.transition_structure = self.get_structure()

    def get_structure(self):
        transition_structure = {
            'state': (np.float32, self.state_shape),
            'action': (np.float32, self.action_shape),
            'reward': (np.float32, ()),
            'next_state': (np.float32, self.state_shape),
            'done': (np.bool_, ()),
            'step_in_episode': (np.int32, ()),
            'episode_max_step_reached': (np.bool_, ())
        }
        if self.action_type == ActionType.GAUSSIAN:
            transition_structure.update({
                'action_log_prob': (np.float32, ()),
                'mean': (np.float32, self.action_shape),
                'std': (np.float32, self.action_shape)
            })

        return transition_structure

    def rollout(self, num_steps: int, policy, reset=False, max_steps_per_episode: int = 1000) -> Dict[str, np.ndarray]:
        """
        Perform exact rollout with num_steps transitions
        Returns dictionary of numpy arrays compatible with ReplayBuffer
        """
        # Preallocate numpy arrays
        transitions = {
            key: np.empty((num_steps, *shape), dtype=dtype)
            for key, (dtype, shape) in self.transition_structure.items()
        }
        
        if reset:
            self.state, _ = self.env.reset()
        steps_in_episode = 0
        
        for step in range(num_steps):
            # Convert state to tensor
            state_t = torch.as_tensor(self.state, dtype=torch.float32)
            
            # Get action from policy
            with torch.no_grad():
                if self.action_type == ActionType.DETERMINISTIC_CONTINUOUS:
                    action = policy(state_t).numpy()
                elif self.action_type == ActionType.GAUSSIAN:
                    mean, std = policy(state_t)
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum()
                    action = action.numpy()
                    
                    # Store Gaussian-specific values
                    transitions['action_log_prob'][step] = log_prob.numpy()
                    transitions['mean'][step] = mean.numpy()
                    transitions['std'][step] = std.numpy()
            
            # Environment step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_max_step = steps_in_episode >= max_steps_per_episode - 1
            
            # Store transition
            transitions['state'][step] = self.state
            transitions['action'][step] = action
            transitions['reward'][step] = reward
            transitions['next_state'][step] = next_state
            transitions['done'][step] = done
            transitions['step_in_episode'][step] = steps_in_episode
            transitions['episode_max_step_reached'][step] = episode_max_step
            
            # Update state and episode tracking
            self.state = next_state
            steps_in_episode += 1
            
            # Reset environment if episode ended
            if done or episode_max_step:
                self.state, _ = self.env.reset()
                steps_in_episode = 0

        return transitions

class VectorizedRollout:
    def __init__(self, envs: List, action_type, state_shape: tuple, action_shape: tuple):
        self.envs = envs
        self.num_envs = len(envs)
        self.action_type = action_type
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.states = np.zeros((self.num_envs, *state_shape))
        
        # Define data structure compatible with ReplayBuffer
        self.transition_structure = self.get_structure()
        self.global_counter = 0

    def get_structure(self):
        transition_structure = {
            'state': (np.float32, self.state_shape),
            'action': (np.float32, self.action_shape),
            'reward': (np.float32, ()),
            'next_state': (np.float32, self.state_shape),
            'done': (np.bool_, ()),
            'step_in_episode': (np.int32, ()),
            'episode_max_step_reached': (np.bool_, ())
        }
        if self.action_type == ActionType.GAUSSIAN:
            transition_structure.update({
                'action_log_prob': (np.float32, ()),
                'mean': (np.float32, self.action_shape),
                'std': (np.float32, self.action_shape)
            })

        return transition_structure

    def rollout(self, num_steps: int, policy: List, reset=False, max_steps_per_episode: int = 1000) -> Dict[str, np.ndarray]:
        assert len(policy) == self.num_envs, "Number of policies must match number of environments"
        
        # Calculate initial cache and chunk sizes
        cache_size = max(1, num_steps // self.num_envs)
        chunk_size = max(1, cache_size // 2)
        
        # Shared state
        self.global_counter = 0
        lock = threading.Lock()
        results = []
        threads = []
        
        # Create worker threads
        for thread_id, env in enumerate(self.envs):
            thread = threading.Thread(
                target=self._env_worker,
                args=(thread_id, env, policy[thread_id], num_steps, max_steps_per_episode,
                     cache_size, chunk_size, lock, results, reset)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
            
        return self._merge_results(results)

    def _env_worker(self, thread_id, env, policy, total_steps, max_steps,
                   cache_size, chunk_size, lock, results, reset):
        # Initialize thread-local storage
        cache = self._init_cache(cache_size)
        current_step = 0
        if reset:
            self.states[thread_id], _ = env.reset()
        steps_in_episode = 0
        
        while True:
            # Atomic step allocation
            with lock:
                if self.global_counter >= total_steps:
                    break
                remaining = total_steps - self.global_counter
                allocated = min(chunk_size, remaining)
                if allocated <= 0:
                    break
                self.global_counter += allocated
                # print(f"Thread: {thread_id}, curr step {current_step} , global_counter: {self.global_counter}")
            
            # Dynamic cache expansion
            if current_step + allocated > cache['state'].shape[0]:
                new_size = current_step + allocated
                cache = self._resize_cache(cache, new_size)
            
            # Process allocated steps
            for i in range(allocated):
                # Policy inference
                state_t = torch.as_tensor(self.states[thread_id], dtype=torch.float32)
                with torch.no_grad():
                    if self.action_type == ActionType.DETERMINISTIC_CONTINUOUS:
                        action = policy(state_t).numpy()
                    elif self.action_type == ActionType.GAUSSIAN:
                        mean, std = policy(state_t)
                        action_dist = torch.distributions.Normal(mean, std)
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action).sum()
                        action = action.numpy()
                        
                        # Store policy parameters
                        cache['action_log_prob'][current_step + i] = log_prob.item()
                        cache['mean'][current_step + i] = mean.numpy()
                        cache['std'][current_step + i] = std.numpy()
                
                # Environment interaction
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_max = steps_in_episode >= max_steps - 1
                
                # Store transition
                cache['state'][current_step + i] = self.states[thread_id]
                cache['action'][current_step + i] = action
                cache['reward'][current_step + i] = reward
                cache['next_state'][current_step + i] = next_state
                cache['done'][current_step + i] = done
                cache['step_in_episode'][current_step + i] = steps_in_episode
                
                # Update state tracking
                if done or episode_max:
                    self.states[thread_id], _ = env.reset()
                    steps_in_episode = 0
                else:
                    self.states[thread_id] = next_state
                    steps_in_episode += 1
            
            current_step += allocated
        
        # Trim and store results
        cache = self._resize_cache(cache, current_step)
        with lock:
            results.append(cache)

    def _init_cache(self, size: int) -> Dict[str, np.ndarray]:
        """Initialize pre-allocated numpy arrays"""
        return {
            key: np.empty((size, *shape), dtype=dtype)
            for key, (dtype, shape) in self.transition_structure.items()
        }

    def _resize_cache(self, cache: Dict[str, np.ndarray], new_size: int) -> Dict[str, np.ndarray]:
        """Resize cache while preserving existing data"""
        if new_size <= cache['state'].shape[0]:
            return {k: v[:new_size] for k, v in cache.items()}
        
        new_cache = self._init_cache(new_size)
        for key in cache:
            new_cache[key][:cache[key].shape[0]] = cache[key]
        return new_cache

    def _merge_results(self, results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Concatenate results from all threads"""
        merged = {}
        for key in self.transition_structure:
            merged[key] = np.concatenate([r[key] for r in results], axis=0)
        return merged