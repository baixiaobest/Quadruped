import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from RL.GradientOperators import GradientOperator as GO
from RL.PolicyNetwork import ActionType
import numpy as np

class ActorCriticOneStep:
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, 
                 num_episodes=1000, max_steps=100, gamma=0.99, print_info=True):
        self.env = env
        self.policy = policy
        self.value_func = value_func
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.print_info = print_info

        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            discount = 1
            rewards = []

            for step in range(self.max_steps):
                action_log_prob = torch.tensor(0, dtype=torch.float32, requires_grad=True)
                action = None
                if self.policy.get_action_type() == ActionType.DISTRIBUTION:
                    actions_prob = self.policy.forward(state.detach())
                    dist = torch.distributions.Categorical(actions_prob)
                    action_idx = dist.sample()
                    action_log_prob = dist.log_prob(action_idx)

                    # Map distribution to [-1, 1]
                    delta_action = 2 / self.policy.get_action_dim()
                    action = -1 + delta_action * action_idx.item() + delta_action / 2
                
                elif self.policy.get_action_type() == ActionType.GAUSSIAN:
                    mean, std = self.policy.forward(state.detach())
                    if step == 0:
                        print(f"mean: {mean}")
                        print(f"std: {std}\n")
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                    action = action.detach().numpy()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                if info and self.print_info:
                    print(info)

                rewards.append(reward)

                td_target = reward + self.gamma * self.value_func.forward(next_state).detach()
                td_error = td_target - self.value_func.forward(state)

                 # Update step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                policy_loss = -action_log_prob * td_error.detach() * discount
                value_loss = td_error.pow(2)

                policy_loss.backward()
                self.policy_optimizer.step()
                
                value_loss.backward()
                self.value_optimizer.step()

                if terminated or truncated:
                    break

                state = next_state
                discount *= self.gamma
                
            if step == self.max_steps:
                print(f"max step reached at episode {episode}")

            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
            self.return_list.append(G)

            print(f"episode {episode} return: {G}")

    def get_returns_list(self):
        return self.return_list


class ActorCriticEligibilityTrace(ActorCriticOneStep):
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, 
                 num_episodes=1000, max_steps=100, gamma=0.99, lambda_policy=0.9, lambda_value=0.9,
                 policy_trace_max=1, value_trace_max=1):
        self.env = env
        self.policy = policy
        self.value_func = value_func
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.lambda_policy = lambda_policy
        self.lambda_value = lambda_value
        self.policy_trace_max = policy_trace_max
        self.value_trace_max = value_trace_max

        self.return_list = []

    def train(self):
        self.policy.train()
        self.return_list = []
        torch.autograd.set_detect_anomaly(True)

        for episode in range(self.num_episodes):
            self.policy.reset() # Reset the internal state of the policy
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            discount = 1
            rewards = []

            policy_trace = None
            value_trace = None

            for step in range(self.max_steps):
                action_log_prob = torch.tensor(0, dtype=torch.float32, requires_grad=True)
                action = None
                if self.policy.get_action_type() == ActionType.DISTRIBUTION:
                    actions_prob = self.policy.forward(state.detach())
                    dist = torch.distributions.Categorical(actions_prob)
                    action_idx = dist.sample()
                    action_log_prob = dist.log_prob(action_idx)

                    # Map distribution to [-1, 1]
                    delta_action = 2 / self.policy.get_action_dim()
                    action = -1 + delta_action * action_idx.item() + delta_action / 2
                
                elif self.policy.get_action_type() == ActionType.GAUSSIAN:
                    mean, std = self.policy.forward(state.detach())
                    # if step == 0:
                    #     print(f"mean: {mean}")
                    #     print(f"std: {std}\n")
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action).sum(dim=-1)
                    action = action.detach().numpy()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                if info:
                    print(info)

                rewards.append(reward)

                done = terminated or truncated

                # Policy forward pass and gradient computation
                td_error = reward + self.gamma * self.value_func.forward(next_state).detach() - self.value_func.forward(state).detach() * (1-done)

                action_log_grad = torch.autograd.grad(action_log_prob, self.policy.parameters())

                if policy_trace is None:
                    policy_trace = GO.grad_const_mul(action_log_grad, discount)
                else:
                    policy_trace = GO.grad_add(GO.grad_const_mul(policy_trace, self.gamma * self.lambda_value), GO.grad_const_mul(action_log_grad, discount))
                
                policy_trace = GO.clamp_grad(policy_trace, -self.policy_trace_max, self.policy_trace_max)
                policy_grad = GO.grad_const_mul(policy_trace, -td_error)

                self.policy_optimizer.zero_grad()
                GO.set_grad(self.policy.parameters(), policy_grad)
                self.policy_optimizer.step()

                # print(f"action_log: {GO.grad_norm(action_log_grad):.3f}")
                # print(f"trace: {GO.grad_norm(policy_trace):.3f}")
                # print(f"grad: {GO.grad_norm(policy_grad):.3f}")
                # print(f"td error: {td_error.item():.3f}\n")

                # if GO.grad_norm(policy_trace).item() > 10:
                #     print("---")

                # # Value forward pass and gradient computation
                curr_value_grad = torch.autograd.grad(self.value_func.forward(state), self.value_func.parameters())
                threshold2=1
                curr_value_grad = GO.clamp_grad(curr_value_grad, -threshold2, threshold2)

                if value_trace is None:
                    value_trace = curr_value_grad
                else:
                    value_trace = GO.grad_add(GO.grad_const_mul(value_trace, self.gamma * self.lambda_value), curr_value_grad)
                
                value_trace = GO.clamp_grad(value_trace, -self.value_trace_max, self.value_trace_max)
                value_grad = GO.grad_const_mul(value_trace, -td_error)

                self.policy_optimizer.zero_grad()
                GO.set_grad(self.value_func.parameters(), value_grad)
                self.value_optimizer.step()

                # print(f"value grad: {GO.grad_norm(curr_value_grad):.3f}")
                # print(f"trace: {GO.grad_norm(value_trace):.3f}")
                # print(f"grad: {GO.grad_norm(value_grad):.3f}")
                # print(f"td error: {td_error.item():.3f}\n")

                # if GO.grad_norm(value_trace).item() > 10:
                #     print("---")

                if terminated or truncated:
                    break

                state = next_state
                discount *= self.gamma                
            if step == self.max_steps:
                print(f"max step reached at episode {episode}")

            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
            self.return_list.append(G)

            print(f"episode {episode} return: {G}")

    def get_returns_list(self):
        return self.return_list
    
