import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from RL.GradientOperators import GradientOperator as GO
from RL.PolicyNetwork import ActionType
from RL.Rollout import SimpleRollout
import numpy as np
import random
from RL.RunningStats import RunningStats
import math
import itertools

class PPO:
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, logger,
                 total_num_steps=10000, max_steps_per_episode=1000, gamma=0.99, lambda_decay=0.95, 
                 entropy_coef=0.1, n_step_per_update=2048, batch_size=64, n_epoch=10, max_norm=0.5, epsilon=0.2, 
                 value_func_epsilon=None, kl_threshold=None, visualize_every=10, visualize_env=None):
        """
        Initialize the Proximal Policy Optimization (PPO) algorithm.
        This implementation includes features like Generalized Advantage Estimation (GAE),
        policy clipping, and  optionalvalue function clipping and KL divergence thresholding.
        Parameters
        ----------
        env : gym.Env
            The environment to train on
        policy : nn.Module or similar
            The policy network that maps states to action distributions
        policy_optimizer : torch.optim.Optimizer
            Optimizer for the policy network
        value_func : nn.Module or similar
            The value function network that estimates state values
        value_optimizer : torch.optim.Optimizer
            Optimizer for the value function network
        logger : object
            Logger for tracking and saving training metrics
        total_num_steps : int, optional
            Total number of timesteps to train for (default: 10000)
        max_steps_per_episode : int, optional
            Maximum number of steps per episode (default: 1000)
        gamma : float, optional
            Discount factor for future rewards (default: 0.99)
        lambda_decay : float, optional
            GAE lambda parameter for advantage calculation (default: 0.95)
        entropy_coef : float, optional
            Coefficient for entropy bonus to encourage exploration (default: 0.1)
        n_step_per_update : int, optional
            Number of steps to collect before each policy update (default: 2048)
        batch_size : int, optional
            Minibatch size for policy updates (default: 64)
        n_epoch : int, optional
            Number of epochs to optimize on the same data (default: 10)
        max_norm : float, optional
            Maximum gradient norm for gradient clipping (default: 0.5)
        epsilon : float, optional
            PPO clipping parameter (default: 0.2)
        value_func_epsilon : float, optional
            Value function clipping parameter, if None no clipping (default: None)
        kl_threshold : float, optional
            KL divergence threshold for early stopping, if None no threshold (default: None)
        visualize_every : int, optional
            Visualize training every n update round (default: 10)
        visualize_env : gym.Env, optional
            Separate environment for visualization (default: None)
        """
        self.env = env
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_func = value_func
        self.value_optimizer = value_optimizer
        self.logger = logger
        self.total_num_steps = total_num_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.entropy_coef = entropy_coef
        self.n_step_per_update = n_step_per_update
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.epsilon = epsilon
        self.value_func_epsilon = value_func_epsilon
        self.kl_threshold = kl_threshold
        self.visualize_every = visualize_every
        self.visualize_env = visualize_env

        self.max_norm = max_norm
        self.reward_clip = 10
        self.std_min = 1e-3

        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        update_round_count = 0

        rollout = SimpleRollout(self.env, self.policy.get_action_type())
        steps_elapsed = 0

        episode = 0

        while steps_elapsed < self.total_num_steps:
                
            transitions = rollout.rollout(
                self.n_step_per_update, self.policy, exact=False, max_steps_per_episode=self.max_steps_per_episode)

            steps_elapsed += self.n_step_per_update

            episode_rewards = []
            update_round_returns = []
            for idx, trans in enumerate(transitions):
                reward = trans['reward']
                state_t = torch.tensor(trans['state'], dtype=torch.float32)
                next_state_t = torch.tensor(trans['next_state'], dtype=torch.float32)
                done = trans['done']
                episode_max_step_reached = trans['episode_max_step_reached']
                step = trans['step_in_episode']
                std = trans['std']

                episode_rewards.append(reward)
                
                target = reward + self.gamma * self.value_func(next_state_t).detach() * (1-done)
                value = self.value_func(state_t).detach()
                td_error =  target - value

                # old_value will be used for value clipping
                transitions[idx]['old_value'] = value

                # Logging
                self.logger.log_episode('td_error', td_error.item(), step=step, episode=episode)
                self.logger.log_episode('target', target.item(), step=step, episode=episode)
                self.logger.log_episode('value', value.item(), step=step, episode=episode)
                self.logger.log_episode('reward', reward, step=step, episode=episode)
                for idx, p_std in enumerate(std):
                    self.logger.log_episode(f'policy_output_std_{idx}', p_std.item(), step=step, episode=episode)
                self.logger.log_episode('policy_output_std_mean', std.mean().item(), step=step, episode=episode)

                if done or episode_max_step_reached or idx == len(transitions) - 1:
                    G = 0
                    for r in reversed(episode_rewards):
                        G = r + self.gamma * G
                    self.return_list.append(G)
                    update_round_returns.append(G)

                    self.logger.log_episode('return', [G], episode=episode)
                    print(f"episode {episode} return: {G}")
                    episode_rewards = []
                    episode += 1

            curr_round_avg_ret = np.mean(update_round_returns)
            self.logger.log_update('rollout_return_mean', [curr_round_avg_ret], update_round=update_round_count)
            print(f"Round {update_round_count} average return: {curr_round_avg_ret}")
            print(f"steps elapsed: {steps_elapsed}")

            td_errors, _ = self._compute_td_errors(transitions)
            # Compute Generalized Advantage Estimation (GAE) in reverse order
            GAE_tensor = self._compute_GAE_tensor(td_errors, transitions)

            for epoch in range(self.n_epoch):

                # Mini-batch update
                # n_step_per_update can be smaller than the size of transitions.
                # We discard the data that exceeds n_step_per_update.
                indices = torch.randperm(self.n_step_per_update)
                
                continue_update = True
                for batch_start in range(0, self.n_step_per_update, self.batch_size):
                    
                    batch_indices = indices[batch_start : batch_start + self.batch_size]
                    batched_traj = [transitions[i] for i in batch_indices]

                    batched_td_error, batched_clipped_td_error = self._compute_td_errors(batched_traj)
                    batched_td_error = torch.cat(batched_td_error)
                    if batched_clipped_td_error is not None:
                        batched_clipped_td_error = torch.cat(batched_clipped_td_error)
                    batched_GAE = torch.tensor([GAE_tensor[i] for i in batch_indices])

                    # Revisit the state buffer and compute new policy probability
                    # and policy loss.
                    replay_state = torch.tensor(np.array([traj['state'] for traj in batched_traj]), dtype=torch.float32)
                    replay_action = torch.tensor(np.array([traj['action'] for traj in batched_traj]), dtype=torch.float32)
                    log_prob_old = torch.tensor(np.array([traj['action_log_prob'] for traj in batched_traj]), dtype=torch.float32)

                    mean_new, std_new = self.policy(replay_state)
                    std_new = torch.clamp(std_new, min=self.std_min)
                    new_dist = torch.distributions.Normal(mean_new, std_new)
                    log_prob_new = new_dist.log_prob(replay_action).sum(dim=-1)
                    entropy = new_dist.entropy().mean()
                    logr = log_prob_new - log_prob_old
                    r = torch.exp(logr)
                    L = r.mul(batched_GAE)
                    kl_div_est_mean = ((r-1) - logr).mean()
                    L_clamped = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon).mul(batched_GAE)
                    policy_loss = -torch.min(L, L_clamped).mean() - self.entropy_coef * entropy

                    # If KL is too big, stop the minibatch and terminate current update round
                    if self.kl_threshold is not None and kl_div_est_mean > self.kl_threshold:
                        continue_update = False
                        break

                    # Optimize policy and value function
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm)
                    policy_grad = [p.grad for p in self.policy.parameters() if p.grad is not None]
                    policy_grad_norm = torch.nn.utils.get_total_norm(policy_grad)
                    self.policy_optimizer.step()

                    value_loss_unclipped = batched_td_error.pow(2).mean()
                    if self.value_func_epsilon is not None:
                        value_loss_clipped = batched_clipped_td_error.pow(2).mean()
                        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                    else:
                        value_loss = value_loss_unclipped

                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_func.parameters(), self.max_norm)
                    value_grad = [p.grad for p in self.value_func.parameters() if p.grad is not None]
                    value_grad_norm = torch.nn.utils.get_total_norm(value_grad)
                    self.value_optimizer.step()

                    # Logging
                    policy_param_norm = torch.nn.utils.get_total_norm(self.policy.parameters())
                    value_func_param_norm = torch.nn.utils.get_total_norm(self.value_func.parameters())
                    index = int(epoch * self.n_step_per_update / self.batch_size + batch_start / self.batch_size)

                    self.logger.log_update('policy_loss', policy_loss.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('value_loss', value_loss.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('policy_ratio', r.mean().item(), update_round=update_round_count, step=index)
                    self.logger.log_update('policy_param_norm', policy_param_norm.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('value_param_norm', value_func_param_norm.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('policy_grad_norm', policy_grad_norm.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('value_grad_norm', value_grad_norm.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('entropy', entropy.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('kl_div_est_mean', kl_div_est_mean.item(), update_round=update_round_count, step=index)
                    self.logger.log_update('epoch_number', epoch, update_round=update_round_count, step=index)
                # End minbatch loop
                            
                if not continue_update:
                    break

            # End epoch loop

            if self.visualize_env is not None and update_round_count % self.visualize_every == 0:
                visualize_rollout = SimpleRollout(self.visualize_env, self.policy)
                visualize_rollout.eval_rollout(1, self.policy, max_steps_per_episode=self.max_steps_per_episode)

            update_round_count += 1

        # End training loop


    def _compute_td_errors(self, trajectory):
        # Recompute td errors given current value function on every update
        td_errors = []
        clipped_td_errors = []
        for idx, traj in enumerate(trajectory):
            state_t = torch.tensor(traj['state'], dtype=torch.float32)
            next_state_t = torch.tensor(traj['next_state'], dtype=torch.float32)
            new_target = traj['reward'] \
                + self.gamma * self.value_func(next_state_t).detach() * (1 - traj['done'])
                
            # Recomputed td errors given current value function
            new_value = self.value_func(state_t)
            new_td_error = new_target - new_value
            td_errors.append(new_td_error)
            # Compute clipped td errors
            if self.value_func_epsilon is not None:
                clipped_value = torch.clamp(new_value, traj['old_value'] - self.value_func_epsilon, traj['old_value'] + self.epsilon)
                clipped_td_error = new_target - clipped_value
                clipped_td_errors.append(clipped_td_error)
            else:
                clipped_td_errors = None
        
        return td_errors, clipped_td_errors
    
    def _compute_GAE_tensor(self, td_errors, trajectory):
        GAE = 0
        GAE_list = []
        for td_error, traj in zip(reversed(td_errors), reversed(trajectory)):
            # When episode ends, GAE does not propagate from one episode to another.
            GAE = td_error.detach() + self.gamma * self.lambda_decay * GAE * (1 - traj['done'])
            GAE_list.append(GAE)
        GAE_list.reverse()
        GAE_tensor = torch.stack(GAE_list)

        # Normalize GAE to have mean 0 and std 1
        if len(GAE_tensor) > 1:
            GAE_tensor = (GAE_tensor - GAE_tensor.mean()) / (GAE_tensor.std() + 1e-5)
        else:
            GAE_tensor[0] = torch.tensor(1, dtype=torch.float32)

        return GAE_tensor

    def get_returns_list(self):
        return self.return_list