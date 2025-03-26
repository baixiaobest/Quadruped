import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from matplotlib import pyplot as plt
import numpy as np
from RL.PolicyNetwork import DoubleIntegratorPolicy
from RL.Environments import DoubleIntegrator1D

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

def plot_inference(state_list, action_list):
    x = [s[0] for s in state_list]
    vx = [s[1] for s in state_list]

    plt.subplot(2, 1, 1)
    plt.title('Inference')
    plt.plot(range(len(x)), x, label='x')
    plt.plot(range(len(vx)), vx, label='vx')
    plt.xlabel('Step')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(action_list)), action_list, label='action')
    plt.xlabel('Step')
    plt.ylabel('Action')
    plt.legend()

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

def inference(policy, noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 0}):
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=1e2,
        x_bound=[-10, 10], 
        v_bound=[-5, 5], 
        v_penalty=0.1, 
        time_penalty=0.1, 
        action_penalty=0.5, 
        action_change_panelty=0.5,
        action_smooth=0.7, 
        x_epsilon=0.1, 
        vx_epsilon=0.1,
        noise=noise,
        bias=bias
    )
    
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
            action_list.append(env.get_action())

            if done:
                print("done")
                break

    print(f"Total reward: {sum(rewards)}")
    plot_inference(state_list, action_list)
    visualize_policy(policy, resolution=100)
    plt.show()

def inference_sweep(policy, seed=0, x_range=(-5, 5), v_range=(-3, 3), grid_resolution=100, 
                    max_steps=100, noise={'x': 0, 'vx': 0, 'action': 0}, 
                    bias={'x': 0, 'vx': 0, 'action': 0}, show=True):
    """
    Sweeps the initial state space and evaluates the policy.
    
    Args:
        x_range (tuple): Range (min, max) for initial position.
        v_range (tuple): Range (min, max) for initial velocity.
        grid_resolution (int): Number of initial states per dimension.
        max_steps (int): Maximum steps to run each episode.
    """
    torch.manual_seed(seed)
    target_x = 0
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=1e2,
        x_bound=[-10, 10], 
        v_bound=[-5, 5], 
        v_penalty=0.1, 
        time_penalty=0.1, 
        action_penalty=0.5, 
        action_change_panelty=0.5,
        action_smooth=0.7, 
        x_epsilon=0.1, 
        vx_epsilon=0.1,
        noise=noise,
        bias=bias
    )
    
    success_count = 0
    failure_count = 0

    # Generate grid of initial states
    x_vals = np.linspace(x_range[0], x_range[1], grid_resolution)
    v_vals = np.linspace(v_range[0], v_range[1], grid_resolution)

    RMS_error_list = []
    step_list = []
    return_list = []
    
    for x0 in x_vals:
        for v0 in v_vals:
            # Reset the environment and override the initial state
            env.set_state(x0, v0)
            state = env.get_state()
            quadratic_error_avg = 0

            # Run the episode for up to max_steps
            step = 0
            return_v = 0
            for step in range(max_steps):
                tracking_error = (state[0] - target_x)**2
                quadratic_error_avg = 1/(step + 1) * (tracking_error - quadratic_error_avg) + quadratic_error_avg

                state_tensor = torch.tensor(state, dtype=torch.float32)
                actions_prob = policy.forward(state_tensor)
                dist = torch.distributions.Categorical(actions_prob)
                action_idx = dist.sample()
                action = policy.get_action(action_idx).item()
                
                state, reward, done = env.step(action)

                return_v += reward

                if done:
                    break
            
            step_list.append(step)
            return_list.append(return_v)
            # Query the environment to determine if the goal was reached
            if env.goal_reached():
                success_count += 1
            else:
                failure_count += 1

            RMS_error_list.append(np.sqrt(quadratic_error_avg))

    RMS_avg = np.average(np.array(RMS_error_list))
    return_avg = np.average(np.array(return_list))
    num_steps_avg = np.average(np.array(step_list))

    print("Success count: ", success_count)
    print("Failure count: ", failure_count)
    print(f"Average RMS error: {RMS_avg}")
    print(f"Return Average: {return_avg}")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.hist(step_list, 50)
    plt.xlabel("steps")
    plt.ylabel("number of runs")
    plt.axvline(num_steps_avg, color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean: {num_steps_avg:.2f}')

    plt.subplot(2, 2, 2)
    plt.hist(return_list, 50)
    plt.xlabel("return")
    plt.ylabel("number of runs")
    plt.axvline(return_avg, color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean: {return_avg:.2f}')

    plt.subplot(2, 2, 3)
    plt.hist(RMS_error_list, 50)
    plt.xlabel("RMS")
    plt.ylabel("number of runs")
    plt.axvline(RMS_avg, color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean: {RMS_avg:.2f}')
    
    plt.subplot(2, 2, 4)
    plt.pie([success_count, failure_count], 
            labels=None,
            explode=(0.1, 0),
            colors=['green', 'red'],
            autopct='%1.1f%%',
            shadow=True)
    plt.legend(['Success', 'Failure'], loc="center left", bbox_to_anchor=(1, 0.5))

    if show:
        plt.show()