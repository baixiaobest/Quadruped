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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_returns(returns_list):
    window = 10
    windowed_returns = [sum(returns_list[i:i+window])/window for i in range(len(returns_list)-window)]
    windowed_variance = [sum([(r - windowed_returns[i])**2 for r in returns_list[i:i+window]])/window for i in range(len(returns_list)-window)]
    
    plt.subplot(2, 1, 1)
    plt.plot(range(len(windowed_returns)), windowed_returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Windowed Returns over Episodes, window={window}')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(windowed_variance)), windowed_variance)
    plt.xlabel('Episode')
    plt.ylabel('Variance')
    plt.title(f'Windowed Variance over Episodes, window={window}')

def training():
    random.seed(20) # this would affect the scenarios at which the agent is trained
    LOAD = True
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, target_x=0, x_bound=[-10, 10], v_bound=[-5, 5], v_penalty=0.1, time_penalty=0.1, x_epsilon=0.5, vx_epsilon=0.2, debug=False)

    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])

    if LOAD:
        policy.load_state_dict(torch.load('RL/training/models/double_integrator_REINFORCE.pth'))

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Create the REINFORCE agent
    reinforce = REINFORCE(env, policy, optimizer, num_episodes=3000, max_steps=200, gamma=0.99)

    # Train the agent
    reinforce.train()

    # Save the policy
    torch.save(policy.state_dict(), 'RL/training/models/double_integrator_REINFORCE.pth')

    plot_returns(reinforce.get_returns_list())
    visualize_policy(policy)
    plt.show()

def inference():
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, target_x=0, x_bound=[-10, 10], v_bound=[-5, 5], x_epsilon=0.5, vx_epsilon=0.2, debug=False)

    # Create the policy network
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=40, hidden_dims=[16, 64], action_range=[-1, 1])
    policy.load_state_dict(torch.load('RL/training/models/double_integrator_REINFORCE.pth'))
    policy.eval()
    
    state = env.reset()
    rewards = []
    state_list = []
    action_list = []

    with torch.no_grad():
        for step in range(10000):
            state_list.append(state)

            state_t = torch.tensor(state, dtype=torch.float32)
            actions_prob = policy.forward(state_t)
            dist = torch.distributions.Categorical(actions_prob)

            action_idx = dist.sample()

            next_state, reward, done = env.step(policy.get_action(action_idx).item())

            rewards.append(reward)

            state = next_state
            action_list.append(policy.get_action(action_idx).item())

            if done:
                print("done")
                break

    print(f"Total reward: {sum(rewards)}")
    plot_inference(state_list, action_list)
    visualize_policy(policy)
    plt.show()

def plot_inference(state_list, action_list):
    x = [s[0] for s in state_list]
    vx = [s[1] for s in state_list]

    plt.subplot(2, 1, 1)
    plt.plot(range(len(x)), x, label='x')
    plt.plot(range(len(vx)), vx, label='vx')
    plt.xlabel('Step')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(action_list)), action_list, label='action')
    plt.xlabel('Step')
    plt.ylabel('Action')

    plt.legend()
    plt.title('Inference')
    plt.legend()

def visualize_policy(policy, x_range=(-10, 10), vx_range=(-5, 5), resolution=50):
    """
    Sweeps the state space (x and vx) and evaluates the policy to produce a 3D meshgrid plot.
    
    Args:
        policy (torch.nn.Module): Neural network policy that takes a 2D state [x, vx] and outputs an action.
        x_range (tuple): The minimum and maximum values of position x.
        vx_range (tuple): The minimum and maximum values of velocity vx.
        resolution (int): Number of points along each axis in the grid.
    """
    # Create grid for x and vx
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    vx_vals = np.linspace(vx_range[0], vx_range[1], resolution)
    X, VX = np.meshgrid(x_vals, vx_vals)
    A = np.zeros_like(X)

    # Evaluate policy for each (x, vx)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], VX[i, j]])
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                actions = policy(state_tensor)
                action = torch.argmax(actions)
                A[i, j] = action.item()
    
    # Plotting the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, VX, A, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Velocity (v_x)')
    ax.set_zlabel('Action')
    ax.set_title('Policy Visualization')
    fig.colorbar(surf, shrink=0.5, aspect=5)

if __name__=='__main__':
    inference()
    # training()