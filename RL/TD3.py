import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from matplotlib.backend_bases import NonGuiException
from sympy import Q
import torch
import numpy as np
from RL.Rollout import SimpleRollout, FastRollout
from RL.ReplayBuffer import ReplayBuffer
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from RL.NoisePolicy import GaussianNoisePolicy, OUNoisePolicy, UniformPolicy

class TD3:
    def __init__(self, env, eval_env, state_dim, action_dim, policy, policy_optimizer, Q1, Q1_optimizer, Q2, Q2_optimizer, 
                 tensorboard_log_dir, log_every=1000,
                 n_epoch=10000, max_steps_per_episode=500, init_buffer_size=1000, init_policy='uniform', 
                 rollout_steps=100, update_per_rollout=100, eval_every=1000, eval_episode=1, batch_size=100, 
                 replay_buffer_size=1e6, policy_delay=2, gamma=0.99, polyak=0.995, 
                 action_noise_config={'type': 'gaussian', 'sigma': 0.2, 'noise_clip': 0.5}, 
                 target_noise=0.2, target_noise_clip=0.5, verbose_logging=False, 
                 # Visualization
                 visualize_every=1000, visualize_env=None):
        """
        Initialize the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
        
        Args:
            env: training environment.
            eval_env: evaluation environment.
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            policy: The actor network that maps states to actions
            policy_optimizer: Optimizer for the policy network
            Q1: The critic network that estimates Q-values 
            Q1_optimizer: Optimizer for the critic network
            Q2: The second critic network
            Q2_optimizer: Optimizer for the second critic network
            log_dir (str): Directory for TensorBoard logs
            
            log_every (int): Number of training epochs between logging
            n_epoch (int): Total number of training epochs
            max_steps_per_episode (int): Maximum number of steps per episode before termination
            init_buffer_size (int): Number of env steps to take before starting to update networks
            rollout_steps (int): Number of env steps per rollout
            update_per_rollout (int): Number of updates per rollout
            eval_every (int): Number of training epochs between evaluation runs
            batch_size (int): Size of minibatch sampled from replay buffer for updates
            replay_buffer_size (float): Maximum size of replay buffer
            policy_delay (int): Number of critic updates per actor update
            gamma (float): Discount factor for future rewards (0,1)
            polyak (float): Interpolation factor for target network updates (0,1)
            action_noise_config (float): configuration for action noise. There are two types, gaussian and OU noise.
                Gaussian Noise example: {'type': 'gaussian', 'sigma': 0.2, 'noise_clip': 0.2}
                OU Noise example: {'type': 'OU', 'sigma': 0.2, 'dt': 0.002, 'theta': 0.15, 'noise_clip': 0.2}
            target_noise (float): Standard deviation of noise added to target actions
            target_noise_clip (float): Maximum absolute value of target policy noise
            noise_clip (float): Maximum absolute value of target policy noise
            visualize_every (int): Number of epochs between visualizations of the policy
            visualize_env: Environment for visualization. If None, no visualization is performed.
        """
        self.env = env
        self.eval_env = eval_env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.Q1 = Q1
        self.Q1_optimizer = Q1_optimizer
        self.Q2 = Q2
        self.Q2_optimizer = Q2_optimizer
        self.log_every = log_every
        self.n_epoch = n_epoch
        self.max_steps_per_episode = max_steps_per_episode
        self.init_buffer_size = init_buffer_size
        self.init_policy = init_policy
        self.rollout_steps = rollout_steps
        self.update_per_rollout = update_per_rollout
        self.eval_every = eval_every
        self.eval_episode = eval_episode
        self.batch_size = batch_size
        self.replay_buffer_size = int(replay_buffer_size)
        self.policy_delay = policy_delay
        self.gamma = gamma
        self.polyak = polyak
        self.action_noise_config = action_noise_config
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.verbose_logging = verbose_logging
        self.visualize_every = visualize_every
        self.visualize_env = visualize_env

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        self.max_grad_norm = 0.5

        # Create target networks for stable learning
        self.policy_target = copy.deepcopy(policy)
        self.Q_target_1 = copy.deepcopy(Q1)
        self.Q_target_2 = copy.deepcopy(Q2)
        self.policy_target.eval()  # Set target networks to evaluation mode
        self.Q_target_1.eval()
        self.Q_target_2.eval()

        self.fast_rollout = FastRollout(
            env=self.env, 
            action_type=self.policy.get_action_type(), 
            state_shape=self.state_dim, 
            action_shape=self.action_dim)

        # Initialize replay buffer to store agent experiences
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_size, 
            structure=self.fast_rollout.get_structure())

    def train(self):

        # initial buffer filling
        if self.init_policy == 'uniform':
            uniform_policy = UniformPolicy(self.action_dim, cache_size=self.init_buffer_size)
            initial_rollout = self.fast_rollout.rollout(
                num_steps=self.init_buffer_size,
                reset=True,
                policy=uniform_policy,
                max_steps_per_episode=self.max_steps_per_episode)
        elif self.init_policy == 'current':
            initial_rollout = self.fast_rollout.rollout(
                num_steps=self.init_buffer_size,
                reset=True,
                policy=self._get_noisy_rollout_policy(self.policy, self.init_buffer_size),
                max_steps_per_episode=self.max_steps_per_episode)
        self.replay_buffer.add(initial_rollout)

        self.policy.train()
        self.Q1.train()
        self.Q2.train()

        total_steps = 0
        
        for epoch in range(self.n_epoch):
            # Interact with the environment using current policy
            noisy_rollout_policy = self._get_noisy_rollout_policy(self.policy, self.rollout_steps)

            if self.update_per_rollout > 0 and epoch % self.update_per_rollout == 0:
                transitions = self.fast_rollout.rollout(
                                        num_steps=self.rollout_steps, 
                                        reset=False,
                                        policy=noisy_rollout_policy, 
                                        max_steps_per_episode=self.max_steps_per_episode)
                self.replay_buffer.add(transitions)
                total_steps += transitions['state'].shape[0]

            with torch.no_grad():
                # Sample a batch of transitions
                batch = self.replay_buffer.sample(self.batch_size)
                states = torch.tensor(batch['state'], dtype=torch.float32)
                actions = torch.tensor(batch['action'], dtype=torch.float32)
                rewards = torch.tensor(batch['reward'], dtype=torch.float32)
                next_states = torch.tensor(batch['next_state'], dtype=torch.float32)
                dones = torch.tensor(batch['done'], dtype=torch.float32)

                # Compute target with target policy noise for smoothing
                target_policy_noise = self._get_noisy_rollout_policy(self.policy_target, self.batch_size)
                min_Q_target = torch.min(
                    self.Q_target_1(next_states, target_policy_noise(next_states)), 
                    self.Q_target_2(next_states, target_policy_noise(next_states))
                )
                target = rewards + (1-dones) * self.gamma * min_Q_target

            # Q1 function update
            Q1_values = self.Q1(states, actions)
            Q1_td_error = target - Q1_values
            Q1_loss = Q1_td_error.pow(2).mean()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward()
            q1_grads_norm = torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), self.max_grad_norm)
            self.Q1_optimizer.step()

            # Q2 function update
            Q2_values = self.Q2(states, actions)
            Q2_td_error = target - Q2_values
            Q2_loss = Q2_td_error.pow(2).mean()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward()
            q2_grads_norm = torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), self.max_grad_norm)
            self.Q2_optimizer.step()

            # Delay policy updates
            if epoch > 0 and epoch % self.policy_delay == 0:
                self._set_network_grad(self.Q1, False)
                policy_loss = -self.Q1(states, self.policy(states)).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self._set_network_grad(self.Q1, True)

                # Soft update of target networks
                for target_param, param in zip(self.Q_target_1.parameters(), self.Q1.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

                for target_param, param in zip(self.Q_target_2.parameters(), self.Q2.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

                for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                    target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)
                
                # Logging
                policy_parameters_norm = torch.nn.utils.get_total_norm(self.policy.parameters())
                self.writer.add_scalar('policy/loss', policy_loss.item(), epoch)
                self.writer.add_scalar('policy/param_norm', policy_parameters_norm, epoch)
                self.writer.add_scalar('policy/grad_norm', policy_grad_norm, epoch)

            # Evaluate the policy and log every eval_every epochs
            if epoch % self.eval_every == 0:
                simple_rollout = SimpleRollout(self.eval_env, self.policy.get_action_type())
                episode_returns, episode_length = simple_rollout.eval_rollout(
                    n_episode=self.eval_episode, 
                    policy=self.policy, 
                    max_steps_per_episode=self.max_steps_per_episode, 
                    gamma=self.gamma)
                eval_mean_return = np.mean(episode_returns)
                episode_length = np.mean(episode_length)

                self.writer.add_scalar('eval/return--epoch', eval_mean_return, epoch)
                self.writer.add_scalar('eval/episode_length--epoch', episode_length, epoch)
                self.writer.add_scalar('eval/return--steps', eval_mean_return, total_steps)
                self.writer.add_scalar('eval/episode_length--steps', episode_length, total_steps)
                print(f"Epoch: {epoch}, return: {eval_mean_return:.2f}, episode length: {episode_length:.2f}")

            # Logging
            if epoch % self.log_every == 0:
                Q1_parameters_norm = torch.nn.utils.get_total_norm(self.Q1.parameters())
                Q2_parameters_norm = torch.nn.utils.get_total_norm(self.Q2.parameters())
                self.writer.add_scalar('Q1/loss', Q1_loss.item(), epoch)
                self.writer.add_scalar('Q2/loss', Q2_loss.item(), epoch)
                self.writer.add_scalar('Q1/param_norm', Q1_parameters_norm, epoch)
                self.writer.add_scalar('Q2/param_norm', Q2_parameters_norm, epoch)
                self.writer.add_scalar('Q1/grad_norm', q1_grads_norm, epoch)
                self.writer.add_scalar('Q2/grad_norm', q2_grads_norm, epoch)
                # This is a lot of data, so we log it only if verbose_logging is enabled
                if self.verbose_logging:
                    self.writer.add_histogram('target', target, epoch)
                    self.writer.add_histogram('Q1/td_errors', Q1_td_error, epoch)
                    self.writer.add_histogram('Q2/td_errors', Q2_td_error, epoch)
                    self.writer.add_histogram('Q1/values', Q1_values, epoch)
                    self.writer.add_histogram('Q2/values', Q2_values, epoch)
            
                self.writer.add_scalar('target/mean', target.mean().item(), epoch)
                self.writer.add_scalar('Q1/td_errors_mean', Q1_td_error.mean().item(), epoch)
                self.writer.add_scalar('Q2/td_errors_mean', Q2_td_error.mean().item(), epoch)
                self.writer.add_scalar('Q1/values_mean', Q1_values.mean().item(), epoch)
                self.writer.add_scalar('Q2/values_mean', Q2_values.mean().item(), epoch)

            # Visualization
            if self.visualize_env is not None and \
                self.visualize_every > 0 and epoch % self.visualize_every == 0:
                visualize_rollout = SimpleRollout(self.visualize_env, self.policy.get_action_type())
                visualize_rollout.eval_rollout(
                    n_episode=1, 
                    policy=self.policy, 
                    max_steps_per_episode=self.max_steps_per_episode, 
                    gamma=self.gamma)

    def _get_noisy_rollout_policy(self, policy, cache_size):
        if self.action_noise_config['type'] == 'gaussian':
            return GaussianNoisePolicy(
                base_policy=policy, 
                noise_scale=self.action_noise_config['sigma'], 
                noise_clip=self.action_noise_config['noise_clip'],
                action_dim=self.action_dim,
                cache_size=cache_size)
        elif self.action_noise_config['type'] == 'OU':
            return OUNoisePolicy(
                base_policy=policy,
                action_dim=self.action_dim,
                noise_scale=self.action_noise_config['sigma'],
                dt=self.action_noise_config['dt'],
                theta=self.action_noise_config['theta'],
                noise_clip=self.action_noise_config['noise_clip'],
                cache_size=cache_size)
        else:
            raise ValueError(f"Unsupported action noise type: {self.action_noise_config['type']}")
        
    def _set_network_grad(self, network, requires_grad):
        """Set the gradient requirement for the network."""
        for param in network.parameters():
            param.requires_grad = requires_grad