import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy import std
import numpy as np
import torch
from RL.PolicyNetwork import ActionType
import itertools
from typing import Dict

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
            'state': (np.float32, (self.state_shape,)),
            'action': (np.float32, (self.action_shape,)),
            'reward': (np.float32, ()),
            'next_state': (np.float32, (self.state_shape,)),
            'done': (np.bool_, ()),
            'step_in_episode': (np.int32, ()),
            'episode_max_step_reached': (np.bool_, ())
        }
        if self.action_type == ActionType.GAUSSIAN:
            transition_structure.update({
                'action_log_prob': (np.float32, ()),
                'mean': (np.float32, (self.action_shape,)),
                'std': (np.float32, (self.action_shape,))
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
    