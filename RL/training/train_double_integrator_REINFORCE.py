import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from RL.REINFORCE import REINFORCE
from RL.Environments import DoubleIntegrator1D
from RL.PolicyNetwork import DoubleIntegratorPolicy
from RL.ValueNetwork import SimpleValuePolicy
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from RL.training.common_double_integrator import *

def training(load, seed, num_episodes=1000, max_steps=200, x_epsilon=0.5, vx_epsilon=0.1, show=False):
    random.seed(seed)
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=1e3,
        x_bound=[-10, 10], 
        v_bound=[-5, 5], 
        v_penalty=0.1, 
        time_penalty=0.1, 
        action_penalty=0.5, 
        action_change_panelty=0.5,
        action_smooth=0.7, 
        x_epsilon=x_epsilon, 
        vx_epsilon=vx_epsilon, 
        debug=False)

    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])

    if load:
        policy.load_state_dict(torch.load('RL/training/models/double_integrator_REINFORCE.pth'))

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Create the REINFORCE agent
    reinforce = REINFORCE(env, policy, optimizer, num_episodes=num_episodes, max_steps=max_steps, gamma=0.99)

    # Train the agent
    reinforce.train()

    # Save the policy
    torch.save(policy.state_dict(), 'RL/training/models/double_integrator_REINFORCE.pth')

    if show:
        plot_returns(reinforce.get_returns_list())
        visualize_policy(policy)
        plt.show()

def train_baseline(load, seed, num_episodes=1000, max_steps=200, x_epsilon=0.5, vx_epsilon=0.1, show=False):
    random.seed(seed)
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=1e3,
        x_bound=[-10, 10], 
        v_bound=[-5, 5], 
        v_penalty=0.1, 
        time_penalty=0.1, 
        action_penalty=0.5, 
        action_change_panelty=0.5,
        action_smooth=0.7, 
        x_epsilon=x_epsilon, 
        vx_epsilon=vx_epsilon, 
        debug=False)
    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])

    if load:
        policy.load_state_dict(torch.load('RL/training/models/double_integrator_REINFORCE_baseline.pth'))

    # Create optimizer
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Value network
    value_net = SimpleValuePolicy(state_dim=2, hidden_dims=[16, 64])

    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Create the REINFORCE agent
    reinforce = REINFORCE(env, policy, policy_optimizer, value_func=value_net, value_optimizer=value_optimizer, num_episodes=num_episodes, max_steps=max_steps, gamma=0.99)

    # Train the agent
    reinforce.train()

    # Save the policy
    torch.save(policy.state_dict(), 'RL/training/models/double_integrator_REINFORCE_baseline.pth')

    if show:
        plot_returns(reinforce.get_returns_list())
        visualize_policy(policy)
        plt.show()

def load_policy(file_name):
    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
    policy.eval()
    return policy

if __name__=='__main__':

    ###No Baseline###
    # policy = load_policy("double_integrator_REINFORCE")
    # inference_sweep(policy, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=200)
    
    # inference(policy)

    # training(load=False, seed=20, num_episodes=3000, max_steps=200, x_epsilon=0.5, vx_epsilon=1)
    # training(load=True, seed=21, num_episodes=1000, max_steps=200, x_epsilon=0.2, vx_epsilon=0.5)
    # training(load=True, seed=22, num_episodes=1000, max_steps=200, x_epsilon=0.1, vx_epsilon=0.1)
    # training(load=True, seed=23, num_episodes=1000, max_steps=200, x_epsilon=0.1, vx_epsilon=0.05)

    ### Baseline ###
    # policy = load_policy("double_integrator_REINFORCE_baseline")
    # inference_sweep(policy, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=200)
    
    # inference(policy)

    # train_baseline(load=False, seed=10, num_episodes=3000, max_steps=200, x_epsilon=0.5, vx_epsilon=1, show=False)
    # train_baseline(load=True, seed=21, num_episodes=1000, max_steps=200, x_epsilon=0.2, vx_epsilon=0.5, show=False)
    # train_baseline(load=True, seed=22, num_episodes=1000, max_steps=200, x_epsilon=0.1, vx_epsilon=0.1, show=False)
    # train_baseline(load=True, seed=13, num_episodes=1000, max_steps=200, x_epsilon=0.1, vx_epsilon=0.05, show=True)