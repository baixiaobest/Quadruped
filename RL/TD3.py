import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from RL.Rollout import SimpleRollout
from RL.ReplayBuffer import ReplayBuffer
import copy

class TD3:
    def __init__(self, env, policy, policy_optimizer, Q1, Q1_optimizer, Q2, Q2_optimizer, logger,
                 n_epoch=10000, max_steps_per_episode=1000, update_after=1000, update_every=50, 
                 eval_every=5, eval_episode=5, batch_size=64, replay_buffer_size=1e6, policy_delay=2, gamma=0.99, polyak=0.995, 
                 action_noise=0.2, target_noise=0.2, noise_clip=0.5):
        """
        Initialize the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
        
        Args:
            env: The environment to interact with
            policy: The actor network that maps states to actions
            policy_optimizer: Optimizer for the policy network
            Q1: The critic network that estimates Q-values 
            Q1_optimizer: Optimizer for the critic network
            Q2: The second critic network
            Q2_optimizer: Optimizer for the second critic network
            logger: Logger for tracking training metrics
            
            n_epoch (int): Total number of training epochs
            max_steps_per_episode (int): Maximum number of steps per episode before termination
            update_after (int): Number of env steps to take before starting to update networks
            update_every (int): Number of env steps between network update rounds
            eval_every (int): Number of training epochs between evaluation runs
            batch_size (int): Size of minibatch sampled from replay buffer for updates
            replay_buffer_size (float): Maximum size of replay buffer
            policy_delay (int): Number of critic updates per actor update
            gamma (float): Discount factor for future rewards (0,1)
            polyak (float): Interpolation factor for target network updates (0,1)
            action_noise (float): Standard deviation of noise added to actions during training
            target_noise (float): Standard deviation of noise added to target actions
            noise_clip (float): Maximum absolute value of target policy noise
        """
        self.env = env
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.Q1 = Q1
        self.Q1_optimizer = Q1_optimizer
        self.Q2 = Q2
        self.Q2_optimizer = Q2_optimizer
        self.logger = logger
        self.n_epoch = n_epoch
        self.max_steps_per_episode = max_steps_per_episode
        self.update_after = update_after
        self.update_every = update_every
        self.eval_every = eval_every
        self.eval_episode = eval_episode
        self.batch_size = batch_size
        self.replay_buffer_size = int(replay_buffer_size)
        self.policy_delay = policy_delay
        self.gamma = gamma
        self.polyak = polyak
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.max_grad_norm = 0.5

        # Create target networks for stable learning
        self.policy_target = copy.deepcopy(policy)
        self.Q_target_1 = copy.deepcopy(Q1)
        self.Q_target_2 = copy.deepcopy(Q2)
        self.policy_target.eval()  # Set target networks to evaluation mode
        self.Q_target_1.eval()
        self.Q_target_2.eval()

        # Initialize replay buffer to store agent experiences
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)


    def train(self):
        state, _ = self.env.reset()
        a = self.policy(torch.tensor(state, dtype=torch.float32))
        action_dim = a.size()[0]

        uniform_policy = self._create_uniform_policy(action_dim)

        simple_rollout = SimpleRollout(self.env)
        initial_rollout = simple_rollout.rollout(
            num_steps=self.update_after,
            policy=uniform_policy,
            max_steps_per_episode=self.max_steps_per_episode)
        self.replay_buffer.add_list(initial_rollout)

        self.policy.train()
        self.Q1.train()
        self.Q2.train()

        for epoch in range(self.n_epoch):
            # Interact with the environment using current policy
            noisy_rollout_policy = self._create_gaussian_noise_policy(self.policy, self.action_noise, None)
            self.replay_buffer.add_list(
                simple_rollout.rollout(num_steps=self.update_every, 
                                policy=noisy_rollout_policy, 
                                max_steps_per_episode=self.max_steps_per_episode))

            # Sample a batch of transitions
            batch = self.replay_buffer.sample(self.batch_size)
            states = torch.tensor(np.array([transition['state'] for transition in batch]), dtype=torch.float32)
            actions = torch.tensor(np.array([transition['action'] for transition in batch]), dtype=torch.float32)
            rewards = torch.tensor(np.array([transition['reward'] for transition in batch]), dtype=torch.float32)
            next_states = torch.tensor(np.array([transition['next_state'] for transition in batch]), dtype=torch.float32)
            dones = torch.tensor(np.array([transition['done'] for transition in batch]), dtype=torch.float32)

            # Compute target with target policy noise for smoothing
            target_policy_noise = self._create_gaussian_noise_policy(self.policy_target, self.target_noise, self.noise_clip)
            min_Q_target = torch.min(
                self.Q_target_1(next_states, target_policy_noise(next_states)), 
                self.Q_target_2(next_states, target_policy_noise(next_states))
            ).detach()
            target = rewards + (1-dones) * self.gamma * min_Q_target

            # Q functions update
            Q1_loss = torch.nn.functional.mse_loss(self.Q1(states, actions), target)
            self.Q1_optimizer.zero_grad()
            Q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), self.max_grad_norm)
            self.Q1_optimizer.step()

            Q2_loss = torch.nn.functional.mse_loss(self.Q2(states, actions), target)
            self.Q2_optimizer.zero_grad()
            Q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), self.max_grad_norm)
            self.Q2_optimizer.step()

            self.logger.log_update('Q1_loss', [Q1_loss.item()], update_round=epoch)
            self.logger.log_update('Q2_loss', [Q2_loss.item()], update_round=epoch)

            # Delay policy updates
            if epoch > 0 and epoch % self.policy_delay == 0:
                self._set_network_grad(self.Q1, False)
                policy_loss = -self.Q1(states, self.policy(states)).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self._set_network_grad(self.Q1, True)

                # Soft update of target networks
                for target_param, param in zip(self.Q_target_1.parameters(), self.Q1.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

                for target_param, param in zip(self.Q_target_2.parameters(), self.Q2.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

                for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)
                
                self.logger.log_update('policy_loss', [policy_loss.item()], update_round=epoch)

            # Evaluate the policy every eval_every epochs
            if epoch % self.eval_every == 0:
                episode_returns, episode_length = simple_rollout.eval_rollout(
                    n_episode=self.eval_episode, 
                    policy=self.policy, 
                    max_steps_per_episode=self.max_steps_per_episode, 
                    gamma=self.gamma)
                eval_mean_return = np.mean(episode_returns)
                episode_length = np.mean(episode_length)
                self.logger.log_update('eval_return', [eval_mean_return], update_round=epoch)
                self.logger.log_update('eval_episode_length', [episode_length], update_round=epoch)

                print(f"Epoch: {epoch}, return: {eval_mean_return:.2f}, episode length: {episode_length:.2f}")


    def _create_gaussian_noise_policy(self, policy, noise_scale, noise_clip=None):
        """Create a noisy version of the policy for exploration."""
        def noisy_policy(state):
            state_t = torch.tensor(state, dtype=torch.float32)
            action_t = policy(state_t)
            noise_t = torch.normal(mean=0, std=noise_scale, size=action_t.size())
            if noise_clip is not None:
                noise_t = torch.clamp(noise_t, -noise_clip, noise_clip)
            noisy_action_t = action_t + noise_t
            noisy_action_t = torch.clamp(noisy_action_t, -1, 1)
            return noisy_action_t
        return noisy_policy
    
    def _create_uniform_policy(self, action_dim):
        """Create a uniform policy for exploration."""
        def uniform_policy(state):
            return torch.clamp(torch.rand(action_dim) * 2.0 - 1.0, -1, 1)
        return uniform_policy
        
    def _set_network_grad(self, network, requires_grad):
        """Set the gradient requirement for the network."""
        for param in network.parameters():
            param.requires_grad = requires_grad