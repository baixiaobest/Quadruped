import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from RL.REINFORCE import REINFORCE
from RL.Environments import DoubleIntegrator1D
from RL.PolicyNetwork import DoubleIntegratorPolicy
import random
from matplotlib import pyplot as plt

def plot_returns(returns_list):
    window = 10
    windowed_returns = [sum(returns_list[i:i+window])/window for i in range(len(returns_list)-window)]
    plt.plot(range(len(windowed_returns)), windowed_returns)
    # plt.plot(range(len(returns_list)), returns_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Windowed Returns over Episodes, window={window}')
    plt.show()

if __name__=='__main__':
    random.seed(0)
    # Create the environment
    env = DoubleIntegrator1D(delta_t=0.05, target_x=0, x_bound=[-10, 10], v_bound=[-5, 5], x_epsilon=0.5, vx_epsilon=0.2)

    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Create the REINFORCE agent
    reinforce = REINFORCE(env, policy, optimizer, num_episodes=2000, max_steps=100, gamma=0.99)

    # Train the agent
    reinforce.train()

    # Save the policy
    torch.save(policy.state_dict(), 'RL/training/models/double_integrator_REINFORCE.pth')

    plot_returns(reinforce.get_returns_list())
