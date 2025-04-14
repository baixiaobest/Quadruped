import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from RL.ActorCritic import ActorCriticOneStep, ActorCriticEligibilityTrace
from RL.Environments import DoubleIntegrator1D
from RL.PolicyNetwork import *
from RL.ValueNetwork import SimpleValueFunction, SimpleQFunction
from RL.TD3 import TD3
from RL.PPO import PPO
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from RL.training.common_double_integrator import *
from RL.Logger import Logger
from RL.LoggerUI import LoggerUI

def train(load, seed, file_name, 
          num_episodes=1000, max_steps=200, 
          x_init_bound=[-5, 5], v_init_bound=[-1, 1],
          x_epsilon=0.5, vx_epsilon=0.1, 
          show=False, algorithm_name="one_step", random_action_bias=0, policy_type='simple',
          set_policy_std=None, debug=False):
    
    random.seed(seed)
    torch.manual_seed(seed)
    # Create the environment
    env = DoubleIntegrator1D(
        delta_t=0.05, 
        target_x=0, 
        goal_reward=100,
        out_of_bound_penalty=10,
        x_bound=[-10, 10], 
        x_init_bound=x_init_bound,
        v_bound=[-5, 5], 
        v_init_bound=v_init_bound,
        action_range=[-5, 5],
        v_penalty=0.1, 
        time_penalty=0.5, 
        action_penalty=0, 
        action_change_panelty=0,
        action_smooth=0.7, 
        x_epsilon=x_epsilon, 
        vx_epsilon=vx_epsilon,
        random_bias={'x': 0, 'vx': 0, 'action': random_action_bias},
        debug=debug)
    
    state_dim = 2
    action_dim = 1

    # Create the policy network
    
    policy = create_policy(policy_type)
    policy.train()

    if load:
        policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
        if set_policy_std:
            policy.set_std(set_policy_std)

    # Create optimizer
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Value network
    value_net = SimpleValueFunction(state_dim=2, hidden_dims=[32, 32])

    if load:
        value_net.load_state_dict(torch.load(f'RL/training/value_models/{file_name}.pth'))

    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    algorithm = None

    logger = Logger()

    if algorithm_name == "one_step":
        algorithm = ActorCriticOneStep(
            env, 
            policy, 
            policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99)
    
    elif algorithm_name == "eligibility_trace":
        algorithm = ActorCriticEligibilityTrace(
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
    elif algorithm_name == 'PPO':
        algorithm = PPO(
            env=env, 
            policy=policy, 
            policy_optimizer=policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            total_num_steps=num_episodes, 
            max_steps_per_episode=max_steps, 
            gamma=0.99, 
            lambda_decay=0.95, 
            entropy_coef=0,
            n_step_per_update=2048,
            batch_size=64, 
            n_epoch=10, 
            epsilon=0.2,
            value_func_epsilon=None,
            kl_threshold=0.1,
            logger=logger)
    elif algorithm_name == 'td3':

        Q1 = SimpleQFunction(state_dim, action_dim, hidden_dims=[64, 64])
        Q1_optimizer = torch.optim.Adam(Q1.parameters(), lr=1e-3)
        Q2 = SimpleQFunction(state_dim, action_dim, hidden_dims=[64, 64])
        Q2_optimizer = torch.optim.Adam(Q2.parameters(), lr=1e-3)

        algorithm = TD3(
            env, 
            policy, 
            policy_optimizer, 
            Q1, 
            Q1_optimizer, 
            Q2, 
            Q2_optimizer, 
            logger,
            n_epoch=num_episodes, 
            max_steps_per_episode=max_steps, 
            update_after=1000, 
            update_every=50, 
            eval_every=5, 
            eval_episode=5, 
            batch_size=100, 
            replay_buffer_size=1e6, 
            policy_delay=2, 
            gamma=0.99, 
            polyak=0.995, 
            action_noise=0.2, 
            target_noise=0.1, 
            noise_clip=0.2)
    else:
        print("Invalid algorithm")
        return

    # Train the agent
    algorithm.train()

    # Save the policy
    torch.save(policy.state_dict(), f'RL/training/models/{file_name}.pth')
    torch.save(value_net.state_dict(), f'RL/training/value_models/{file_name}.pth')

    if show:
        # plot_returns(algorithm.get_returns_list())
        visualize_policy(policy)
        if not (algorithm_name == "one_step" or algorithm_name == "eligibility_trace"):
            logger.save_to_file(f'RL/training/log/{file_name}.pkl')
            ui = LoggerUI(logger)
            ui.run()
        else:
            plt.show()

def create_policy(policy_type='simple'):
    policy = DoubleIntegratorPolicy(state_dim=2, action_dim=100, hidden_dims=[16, 64])
    if policy_type == 'lstm':
        policy = DoubleIntegratorPolicyLSTM(state_dim=2, action_dim=100, hidden_dims=[16, 64])
    elif policy_type == 'gaussian':
        policy = GaussianPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64], std_init=0.4)
    elif policy_type == 'gaussian_decay':
        policy = GaussianPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64], std_init=0.2, temperature_decay=0.9999)
    elif policy_type == 'gaussian_advanced':
        policy = GaussianStateDependentPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64])
    elif policy_type == 'deterministic':
        policy = DeterministicContinuousPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64])
    
    return policy

def load_policy(file_name, policy_type='simple'):
    policy = create_policy(policy_type)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
    policy.eval()
    return policy

def plot_log(file_name):
    logger = Logger()
    logger.load_from_file(f'RL/training/log/{file_name}.pkl')
    ui = LoggerUI(logger)
    ui.run()

if __name__ == '__main__':
    # policy1 = load_policy("double_integrator_actor_critic_trace", policy_type='simple')
    # policy2 = load_policy("double_integrator_actor_critic_trace_lstm", policy_type='lstm')
    # policy3 = load_policy("double_integrator_ppo_gaussian", policy_type='gaussian')
    policy4 = load_policy("double_integrator_td3", policy_type='deterministic')

    # inference_sweep(policy4, seed=10, x_range=(-5, 5), v_range=(-1, 1), grid_resolution=20, max_steps=500, 
    #                 noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 0}, show=True)

    # inference_double_integrator(policy4, noise={'x': 0, 'vx': 0, 'action': 0}, bias={'x': 0, 'vx': 0, 'action': 0})

    # One step actor critic

    # train(load=False, seed=45, file_name='double_integrator_actor_critic', num_episodes=500, 
    #       max_steps=500, x_epsilon=0.5, vx_epsilon=1, show=False, algorithm_name="one_step", policy_type='simple')
    # train(load=True, seed=50, file_name='double_integrator_actor_critic', num_episodes=500, max_steps=500, x_epsilon=0.1, vx_epsilon=0.05, show=True)

    # Eligibility Trace

    # train(load=False, seed=45, file_name='double_integrator_actor_critic_trace', num_episodes=500, 
    #       max_steps=500, x_epsilon=0.5, vx_epsilon=1, show=True, algorithm_name="eligibility_trace", 
    #       random_action_bias=2)
    # train(load=True, seed=50, file_name='double_integrator_actor_critic_trace', num_episodes=500, 
    #     max_steps=500, x_epsilon=0.1, vx_epsilon=0.05, show=True, algorithm_name="eligibility_trace", 
    #     random_action_bias=4)
    
    # Gaussian output

    # train(load=False, seed=10, file_name='double_integrator_actor_critic_gaussian', num_episodes=500, 
    #       max_steps=500, x_epsilon=0.5, vx_epsilon=0.2, show=True, algorithm_name="one_step", 
    #       random_action_bias=0, policy_type='gaussian', debug=True)
    
    # train(load=True, seed=26, file_name='double_integrator_actor_critic_gaussian', num_episodes=100, 
    #       max_steps=500, x_epsilon=0.1, vx_epsilon=0.1, show=True, algorithm_name="one_step", 
    #       random_action_bias=0, policy_type='gaussian_decay')

    # PPO training
    # train(load=False, seed=54, file_name='double_integrator_ppo_gaussian', num_episodes=100_000, 
    #       max_steps=200, x_init_bound=[-1, 1], v_init_bound=[-0.5, 0.5], x_epsilon=0.1, vx_epsilon=0.1, show=True, algorithm_name="PPO", 
    #       random_action_bias=0, policy_type='gaussian')

    # train(load=True, seed=874, file_name='double_integrator_ppo_gaussian', num_episodes=200, 
    #       max_steps=200, x_init_bound=[-5, 5], v_init_bound=[-1, 1], x_epsilon=0.1, vx_epsilon=0.1, 
    #       show=True, algorithm_name="PPO", random_action_bias=0, policy_type='gaussian', 
    #       set_policy_std=0.4, debug=False)

    # TD3 training
    train(load=False, seed=842, file_name='double_integrator_td3', num_episodes=30_000,
          max_steps=200, x_init_bound=[-5, 5], v_init_bound=[-1, 1], x_epsilon=0.1, vx_epsilon=0.1, 
          show=True, algorithm_name="td3", policy_type='deterministic', debug=False)

    # plot_log("double_integrator_td3")