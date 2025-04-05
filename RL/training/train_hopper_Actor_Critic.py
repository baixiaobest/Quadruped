import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 1. Set display backend variables (MUST come before any gymnasium/mujoco import)
os.environ["DISPLAY"] = ":0"           # Forces X11 display
os.environ["XDG_SESSION_TYPE"] = "x11" # Explicitly use X11 session
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Qt platform plugin for X11

import random
from RL.ActorCritic import ActorCriticEligibilityTrace, ActorCriticOneStep
from RL.ValueNetwork import SimpleValuePolicy
from RL.PolicyNetwork import GaussianPolicy
from RL.training.common_double_integrator import *
from RL.PPO import PPO
from RL.Logger import Logger
from RL.LoggerUI import LoggerUI
import gymnasium as gym
import torch

def train(load, seed, file_name, start_policy_name=None, num_episodes=100, max_steps=1000, 
          algorithm_name='one_step', policy_type='gaussian', set_policy_std=0, show=False, render=False):
    random.seed(seed)
    torch.manual_seed(seed)

    render_mode = 'human' if render else None
    env = create_hopper(render_mode=render_mode)
    logger = Logger()

    # Policy network and optimizer
    policy = create_policy(policy_type)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    if load:
        if start_policy_name:
            policy.load_state_dict(torch.load(f'RL/training/models/{start_policy_name}.pth'))
        else:
            policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    # Value network
    value_net = SimpleValuePolicy(state_dim=11, hidden_dims=[64, 64])
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    if set_policy_std > 0 and isinstance(policy, GaussianPolicy):
        policy.set_std(set_policy_std)

    def improve_callback(R):
        if R < 150:
            return
        print(f"saving with return {R}")
        torch.save(policy.state_dict(), f'RL/training/models/{file_name}_R_{R:.0f}.pth')

    if algorithm_name == 'eligibility_trace':
        algorithm = ActorCriticEligibilityTrace(
            env, 
            policy, 
            policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99,
            lambda_policy=0.9, 
            lambda_value=0.9,
            policy_trace_max=1,
            value_trace_max=1)
        
    elif algorithm_name == 'one_step':
        algorithm = ActorCriticOneStep(
            env, 
            policy, 
            policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99,
            print_info=False)
        
    elif algorithm_name == 'PPO':
        algorithm = PPO(
            env=env, 
            policy=policy, 
            policy_optimizer=policy_optimizer, 
            value_func=value_net, 
            value_optimizer=value_optimizer, 
            num_episodes=num_episodes, 
            max_steps=max_steps, 
            gamma=0.99, 
            lambda_decay=1.0, 
            n_step=10,
            # batch_size=10, 
            n_epoch=5, 
            epsilon=0.2,
            improve_callback=improve_callback,
            logger=logger)
    else:
        print("Invalid algorithm")
        return

    algorithm.train()
    env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'RL/training/models/{file_name}.pth')

    if show:
        # visualize_policy(policy)
        logger.save_to_file(f'RL/training/log/{file_name}.pkl')
        ui = LoggerUI(logger)
        ui.run()


def create_policy(policy_type):
    if policy_type == 'gaussian':
        policy = GaussianPolicy(state_dim=11, action_dim=3, hidden_dims=[64, 64], 
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1)
    elif policy_type == 'gaussian_decay':
        policy = GaussianPolicy(state_dim=11, action_dim=3, hidden_dims=[64, 64], 
                                std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1-1e-7)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policy

def create_hopper(render_mode=True):
    env = gym.make("Hopper-v5", 
                   render_mode=render_mode, 
                   healthy_angle_range=(-1*np.pi, 1*np.pi),
                   healthy_state_range=(-100, float("inf")),
                   healthy_z_range=(0.5, float("inf")),
                   termination_penalty=0,
                   jump_reward_weight=0.2)
    
    return env


def inference_hopper(file_name, render=True, policy_type='gaussian'):
    render_mode = 'human' if render else None
    env = create_hopper(render_mode=render_mode)

    # Policy network and optimizer
    policy = create_policy(policy_type)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, deterministic=False)

    env.close()

def plot_log(file_name):
    logger = Logger()
    logger.load_from_file(f'RL/training/log/{file_name}.pkl')
    ui = LoggerUI(logger)
    ui.run()

if __name__=='__main__':
     train(load=False, seed=45148, file_name="hopper_actor_critic_gaussian", algorithm_name="PPO", 
          policy_type='gaussian', num_episodes=500, max_steps=1000, set_policy_std=0.4, show=True, render=True)
     
    #  plot_log(file_name="hopper_actor_critic_gaussian")
    
    # train(load=True, seed=45148, file_name="hopper_actor_critic_gaussian", 
    #       start_policy_name="hopper_actor_critic_gaussian_R_224", algorithm_name="PPO", 
    #       num_episodes=500, max_steps=1000, set_policy_std=0.2, show=True, render=True)

    # inference_hopper(file_name="hopper_actor_critic_gaussian", policy_type='gaussian', render=True)