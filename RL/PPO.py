import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from RL.GradientOperators import GradientOperator as GO
from RL.PolicyNetwork import ActionType
import numpy as np
import random
from RL.RunningStats import RunningStats
import math
import itertools

class PPO:
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, 
                 total_num_steps=10000, max_steps_per_episode=1000, gamma=0.99, lambda_decay=0.95, 
                 entropy_coef=0.1, n_step_per_update=2048, batch_size=64, n_epoch=10, max_norm=0.5, epsilon=0.2, 
                 value_func_epsilon=None, kl_threshold=None, print_info=True, logger=None):
        self.env = env
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_func = value_func
        self.value_optimizer = value_optimizer
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
        self.print_info = print_info
        self.logger = logger

        self.max_norm = max_norm
        self.reward_clip = 10
        self.std_min = 1e-3

        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        steps_elapsed = 0

        transitions = []
        update_round_count = 0
        update_round_returns = []

        for episode in itertools.count():
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)

            episode_rewards = []
            
            for step in range(self.max_steps_per_episode):
                action = None

                mean, std = self.policy.forward(state)
                std = torch.clamp(std, min=self.std_min)
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                action = action.detach()

                next_state, reward, terminated, truncated, info = self.env.step(action.numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32)
                done = terminated or truncated

                # reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                
                episode_rewards.append(reward)
                
                target = reward + self.gamma * self.value_func(next_state).detach() * (1-done)
                value = self.value_func(state).detach()
                td_error =  target - value

                transitions.append({
                    'state': state, 
                    'next_state': next_state,
                    'action': action, 
                    'log_prob_old': action_log_prob.detach(),
                    'old_value': value,
                    'reward': reward,
                    'done': done})
                
                self.logger.log('td_error', td_error.item(), step=step, episode=episode)
                self.logger.log('target', target.item(), step=step, episode=episode)
                self.logger.log('value', value.item(), step=step, episode=episode)
                self.logger.log('reward', reward, step=step, episode=episode)

                # Iterate of actions
                for idx, p_std in enumerate(std):
                    self.logger.log(f'policy_output_std_{idx}', p_std.item(), step=step, episode=episode)
                self.logger.log('policy_output_std_mean', std.mean().item(), step=step, episode=episode)

                # Stop and compute td errors and advantage
                if steps_elapsed > 0 and steps_elapsed % self.n_step_per_update == 0:

                    curr_round_avg_ret = np.mean(update_round_returns)
                    self.logger.log('rollout_return_mean->update_round', [curr_round_avg_ret], episode=update_round_count)
                    update_round_returns = []
                    print(f"Round {update_round_count} average return: {curr_round_avg_ret}")
                    print(f"steps elapsed: {steps_elapsed}")

                    td_errors, _ = self._compute_td_errors(transitions)
                    # Compute Generalized Advantage Estimation (GAE) in reverse order
                    GAE_tensor = self._compute_GAE_tensor(td_errors, transitions)

                    for epoch in range(self.n_epoch):

                        # Mini-batch update
                        indices = torch.randperm(len(transitions))
                        
                        continue_update = True
                        for batch_start in range(0, len(transitions), self.batch_size):
                            
                            batch_indices = indices[batch_start : batch_start + self.batch_size]
                            batched_traj = [transitions[i] for i in batch_indices]

                            batched_td_error, batched_clipped_td_error = self._compute_td_errors(batched_traj)
                            batched_td_error = torch.cat(batched_td_error)
                            if batched_clipped_td_error is not None:
                                batched_clipped_td_error = torch.cat(batched_clipped_td_error)
                            batched_GAE = torch.tensor([GAE_tensor[i] for i in batch_indices])

                            # Revisit the state buffer and compute new policy probability
                            # and policy loss.
                            replay_state = torch.stack([traj['state'] for traj in batched_traj])
                            replay_action = torch.stack([traj['action'] for traj in batched_traj])
                            log_prob_old = torch.stack([traj['log_prob_old'] for traj in batched_traj])

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
                            policy_grad_norm = torch.nn.utils.get_total_norm(self.policy.parameters())
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
                            value_func_grad_norm = torch.nn.utils.get_total_norm(self.value_func.parameters())
                            self.value_optimizer.step()

                            index = int(epoch * self.n_step_per_update / self.batch_size + batch_start / self.batch_size)
                            self.logger.log('policy_loss->update_round->batch_round', policy_loss.item(), episode=update_round_count, step=index)
                            self.logger.log('value_loss->update_round->batch_round', value_loss.item(), episode=update_round_count, step=index)
                            self.logger.log('policy_ratio->update_round->batch_round', r.mean().item(), episode=update_round_count, step=index)
                            self.logger.log('policy_grad_norm->update_round->batch_round', policy_grad_norm.item(), episode=update_round_count, step=index)
                            self.logger.log('value_grad_norm->update_round->batch_round', value_func_grad_norm.item(), episode=update_round_count, step=index)
                            self.logger.log('entropy->update_round->batch_round', entropy.item(), episode=update_round_count, step=index)
                            self.logger.log('kl_div_est_mean->update_round->batch_round', kl_div_est_mean.item(), episode=update_round_count, step=index)
                            self.logger.log('epoch_number->update_round->batch_round', epoch, episode=update_round_count, step=index)
                        # End minbatch loop
                            
                        if not continue_update:
                            break

                    # End epoch loop
                        
                    transitions = []
                    update_round_count += 1

                # End update clause

                state = next_state
                steps_elapsed += 1

                if done:
                    # print(info)
                    break 

            # End step loop

            if step == self.max_steps_per_episode:
                print(f"max step reached at episode {episode}")

            G = 0
            for r in reversed(episode_rewards):
                G = r + self.gamma * G
            self.return_list.append(G)
            update_round_returns.append(G)

            self.logger.log('return', [G], episode=episode)

            print(f"episode {episode} return: {G}")

            # Stops the whole training if we reach maximum total number of steps
            if steps_elapsed >= self.total_num_steps:
                break

        # End episode loop

    def _compute_td_errors(self, trajectory):
        # Recompute td errors given current value function on every update
        td_errors = []
        clipped_td_errors = []
        for idx, traj in enumerate(trajectory):
            new_target = traj['reward'] \
                + self.gamma * self.value_func(traj['next_state']).detach() * (1 - traj['done'])
                
            # Recomputed td errors given current value function
            new_value = self.value_func(traj['state'])
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