import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from RL.ActorCritic import ActorCriticOneStep, ActorCriticEligibilityTrace
from RL.Environments import DoubleIntegrator1D
from RL.PolicyNetwork import DoubleIntegratorPolicy, DoubleIntegratorPolicyLSTM, GaussianStateDependentPolicy
from RL.ValueNetwork import SimpleValuePolicy
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from RL.training.common_double_integrator import *

def train(load, seed, file_name, 
          num_episodes=1000, max_steps=200, 
          x_epsilon=0.5, vx_epsilon=0.1, 
          show=False, algorithm="one_step", random_action_bias=0, policy_type='simple'):
    
    random.seed(seed)
    torch.manual_seed(seed)
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=1e2,
        x_bound=[-10, 10], 
        v_bound=[-5, 5], 
        action_range=[-5, 5],
        v_penalty=0.1, 
        time_penalty=0.1, 
        action_penalty=0.5, 
        action_change_panelty=0,
        action_smooth=0.7, 
        x_epsilon=x_epsilon, 
        vx_epsilon=vx_epsilon,
        random_bias={'x': 0, 'vx': 0, 'action': random_action_bias})

    # Create the policy network
    
    policy = create_policy(policy_type)

    if load:
        policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    # Create optimizer
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Value network
    value_net = SimpleValuePolicy(state_dim=2, hidden_dims=[16, 64])

    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Create the REINFORCE agent
    actor_critic = None

    if algorithm == "one_step":
        actor_critic = ActorCriticOneStep(
            env, 
            policy, 
            policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99)
    
    elif algorithm == "eligibility_trace":
        actor_critic = ActorCriticEligibilityTrace(
            env, 
            policy, 
            policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99,
            lambda_policy=0.1, 
            lambda_value=0.1,
            policy_trace_max=1,
            value_trace_max=1)
    else:
        print("Invalid algorithm")

    # Train the agent
    actor_critic.train()

    # Save the policy
    torch.save(policy.state_dict(), f'RL/training/models/{file_name}.pth')

    if show:
        plot_returns(actor_critic.get_returns_list())
        visualize_policy(policy)
        plt.show()

def create_policy(policy_type='simple'):
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=100, hidden_dims=[16, 64])
    if policy_type == 'lstm':
        policy = DoubleIntegratorPolicyLSTM(state_dim=2, action_dim=100, hidden_dims=[16, 64])
    if policy_type == 'gaussian':
        policy = GaussianStateDependentPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64])
    
    return policy

def load_policy(file_name, policy_type='simple'):
    policy = create_policy(policy_type)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
    policy.eval()
    return policy

if __name__ == '__main__':
    # policy1 = load_policy("double_integrator_actor_critic_trace", policy_type='simple')
    # policy2 = load_policy("double_integrator_actor_critic_trace_lstm", policy_type='lstm')
    policy3 = load_policy("double_integrator_actor_critic_trace_gaussian", policy_type='gaussian')

    # inference_sweep(policy1, seed=10, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=500, 
    #                 noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 3}, show=False)
    # inference_sweep(policy2, seed=10, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=500, 
    #                 noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 3}, show=True)
    # inference_sweep(policy3, seed=10, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=500, 
    #                 noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 0}, show=True)

    inference(policy3, noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 0})

    # One step actor critic

    # train(load=False, seed=45, file_name='double_integrator_actor_critic', num_episodes=500, 
    #       max_steps=500, x_epsilon=0.5, vx_epsilon=1, show=False, algorithm="one_step", policy_type='simple')
    # train(load=True, seed=50, file_name='double_integrator_actor_critic', num_episodes=500, max_steps=500, x_epsilon=0.1, vx_epsilon=0.05, show=True)

    # Eligibility Trace

    # train(load=False, seed=45, file_name='double_integrator_actor_critic_trace', num_episodes=500, 
    #       max_steps=500, x_epsilon=0.5, vx_epsilon=1, show=True, algorithm="eligibility_trace", 
    #       random_action_bias=2)
    # train(load=True, seed=50, file_name='double_integrator_actor_critic_trace', num_episodes=500, 
    #     max_steps=500, x_epsilon=0.1, vx_epsilon=0.05, show=True, algorithm="eligibility_trace", 
    #     random_action_bias=4)
    
    # Gaussian output

    # train(load=False, seed=15, file_name='double_integrator_actor_critic_trace_gaussian', num_episodes=500, 
    #       max_steps=1000, x_epsilon=0.5, vx_epsilon=0.5, show=True, algorithm="one_step", 
    #       random_action_bias=0, policy_type='gaussian')
