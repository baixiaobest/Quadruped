import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from RL.GradientOperators import GradientOperator as GO
from RL.PolicyNetwork import ActionType
import numpy as np
import random
from RL.RunningStats import DiscountedSumRunningStats

class PPO:
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, 
                 num_episodes=1000, max_steps=100, gamma=0.99, lambda_decay=1.0, n_step=30,
                 batch_size=10, n_epoch=5, epsilon=0.2, print_info=True, 
                 logger=None, improve_callback=None):
        self.env = env
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_func = value_func
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.n_step = n_step
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.epsilon = epsilon
        self.print_info = print_info
        self.logger = logger
        self.improve_callback = improve_callback

        self.policy_max_norm = 1e-2
        self.value_max_norm = 1e-2
        self.reward_clip = 10

        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        max_return = -np.inf

        reward_stats = DiscountedSumRunningStats(beta=0.9, gamma=self.gamma)

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)

            trajectory = []
            episode_rewards = []

            policy_update_round = 0
            
            for step in range(self.max_steps):
                action = None
                mean, std = self.policy.forward(state)
                if step == 0:
                    print(f"mean: {mean}")
                    print(f"std: {std}\n")
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                action = action.detach()

                next_state, reward, terminated, truncated, info = self.env.step(action.numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32)
                done = terminated or truncated

                # reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                
                episode_rewards.append(reward)

                # Scale down the reward to stablize the training
                reward_stats.update(reward)
                scaled_reward = reward
                if reward_stats.get_count() > 10:
                    scale = np.clip(reward_stats.get_std(), 1.0, np.inf)
                    scaled_reward /= scale
                
                target = reward + self.gamma * self.value_func(next_state).detach() * (1-done)
                value = self.value_func(state).detach()
                td_error =  target - value

                trajectory.append({
                    'state': state, 
                    'action': action, 
                    'log_prob_old': action_log_prob.detach(),
                    'old_value': value,
                    'reward': reward,
                    'scaled_reward': scaled_reward,
                    'next_state': next_state})
                
                self.logger.log('td_error', td_error.item(), step=step, episode=episode)
                self.logger.log('target', target.item(), step=step, episode=episode)
                self.logger.log('value', value.item(), step=step, episode=episode)
                self.logger.log('reward', reward, step=step, episode=episode)
                self.logger.log('scaled_reward', scaled_reward, step=step, episode=episode)
                self.logger.log('discounted_sum_mean', reward_stats.get_mean(), step=step, episode=episode)
                self.logger.log('discounted_sum_std', reward_stats.get_std(), step=step, episode=episode)
                # Iterate of actions
                for idx, p_std in enumerate(std):
                    self.logger.log(f'policy_output_std_{idx}', p_std.item(), step=step, episode=episode)
                self.logger.log('policy_output_std_mean', std.mean().item(), step=step, episode=episode)

                # Stop and compute advantages
                if (not step == 0) and (step % self.n_step == 0) \
                    or step == self.max_steps - 1 \
                    or done:

                    for epoch in range(self.n_epoch):
                        
                        # Recompute td errors given current value function on every update
                        td_errors = []
                        clipped_td_errors = []
                        for idx, traj in enumerate(trajectory):
                            if done and (idx == len(trajectory) - 1):
                                new_target = traj['scaled_reward']
                            else:
                                # Target is constant therefore it needs to be detached.
                                new_target = traj['scaled_reward'] \
                                    + self.gamma * self.value_func(traj['next_state']).detach()
                                
                            # Recomputed td errors given current value function
                            new_value = self.value_func(traj['state'])
                            new_td_error = new_target - new_value
                            td_errors.append(new_td_error)
                            # Compute clipped td errors
                            clipped_value = torch.clamp(new_value, traj['old_value'] - self.epsilon, traj['old_value'] + self.epsilon)
                            clipped_td_error = new_target - clipped_value
                            clipped_td_errors.append(clipped_td_error)

                        # Compute advantages from td errors
                        A = 0
                        A_list = []
                        for td_error in reversed(td_errors):
                            A = td_error.detach() + self.gamma * self.lambda_decay * A
                            A_list.append(A)
                        A_list.reverse()
                        A_tensor = torch.stack(A_list)
                        if len(A_tensor) > 1:
                            A_tensor = (A_tensor - A_tensor.mean()) / (A_tensor.std() + 1e-5)
                        else:
                            A_tensor[0] = torch.tensor(1, dtype=torch.float32)

                        for idx, traj in enumerate(trajectory):
                            traj['advantage'] = A_tensor[idx]
                            
                            if epoch == 0:
                                self.logger.log('advantage', traj['advantage'].item(), 
                                                step=(policy_update_round * self.n_epoch + idx), episode=episode)

                        # Revisit the state buffer and compute new policy probability
                        # and policy loss.
                        replay_state = torch.stack([traj['state'] for traj in trajectory])
                        replay_action = torch.stack([traj['action'] for traj in trajectory])
                        log_prob_old = torch.stack([traj['log_prob_old'] for traj in trajectory])
                        advantage = torch.tensor([traj['advantage'] for traj in trajectory])
                        mean_new, std_new = self.policy(replay_state)
                        new_dist = torch.distributions.Normal(mean_new, std_new)
                        log_prob_new = new_dist.log_prob(replay_action).sum(dim=-1)
                        r = torch.exp(log_prob_new - log_prob_old)
                        L = r.mul(advantage)
                        L_clamped = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon).mul(advantage)
                        policy_loss = -torch.min(L, L_clamped).mean()

                        # Optimize policy and value function
                        self.policy_optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_max_norm)
                        policy_grad_norm = torch.nn.utils.get_total_norm(self.policy.parameters())
                        self.policy_optimizer.step()

                        value_loss_unclipped = torch.stack(td_errors).pow(2).mean()
                        value_loss_clipped = torch.stack(clipped_td_errors).pow(2).mean()
                        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                        self.value_optimizer.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.value_func.parameters(), self.value_max_norm)
                        value_func_grad_norm = torch.nn.utils.get_total_norm(self.value_func.parameters())
                        self.value_optimizer.step()

                        self.logger.log('policy_loss_epoch', policy_loss.item(), episode=episode, 
                                        step=(policy_update_round*self.n_epoch + epoch))
                        self.logger.log('value_loss_epoch', value_loss.item(), episode=episode,
                                        step=(policy_update_round*self.n_epoch + epoch))
                        self.logger.log('policy_ratio_epoch', r.mean().item(), episode=episode,
                                        step=(policy_update_round*self.n_epoch + epoch))
                        self.logger.log('policy_grad_norm_epoch', policy_grad_norm, episode=episode,
                                        step=(policy_update_round*self.n_epoch + epoch))
                        self.logger.log('value_grad_norm_epoch', value_func_grad_norm, episode=episode,
                                        step=(policy_update_round*self.n_epoch + epoch))

                    trajectory = []
                    policy_update_round += 1

                if done:
                    print(info)
                    break

                state = next_state
                
            if step == self.max_steps:
                print(f"max step reached at episode {episode}")

            G = 0
            for r in reversed(episode_rewards):
                G = r + self.gamma * G
            self.return_list.append(G)

            self.logger.log('return', [G], episode=episode)

            if self.improve_callback and G > max_return:
                max_return = G
                self.improve_callback(G)

            print(f"episode {episode} return: {G}")

    def get_returns_list(self):
        return self.return_list