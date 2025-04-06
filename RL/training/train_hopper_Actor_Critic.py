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
          algorithm_name='one_step', policy_type='gaussian', set_policy_std=0, entropy_coef=0.01,
          show=False, render=False):
    random.seed(seed)
    torch.manual_seed(seed)

    render_mode = 'human' if render else None
    env = create_hopper(render_mode=render_mode)
    logger = Logger()

    # Policy network and optimizer
    policy = create_policy(policy_type)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Value network
    value_net = SimpleValuePolicy(state_dim=11, hidden_dims=[64, 64])
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    if load:
        if start_policy_name:
            policy.load_state_dict(torch.load(f'RL/training/models/{start_policy_name}.pth'))
            value_net.load_state_dict(torch.load(f'RL/training/value_models/{start_policy_name}.pth'))
        else:
            policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
            value_net.load_state_dict(torch.load(f'RL/training/value_models/{file_name}.pth'))

    if set_policy_std and set_policy_std > 0 and isinstance(policy, GaussianPolicy):
        policy.set_std(set_policy_std)

    def improve_callback(R):
        # if R < 200:
        #     return
        # print(f"saving with return {R}")
        # torch.save(policy.state_dict(), f'RL/training/models/{file_name}_R_{R:.0f}.pth')
        # torch.save(value_net.state_dict(), f'RL/training/value_models/{file_name}_R_{R:.0f}.pth')
        # logger.save_to_file(f'RL/training/log/{file_name}_R_{R:.0f}.pkl')
        pass

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
            lambda_decay=0.99, 
            entropy_coef=entropy_coef,
            n_step=2048,
            batch_size=64, 
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

    # Save the value network
    torch.save(value_net.state_dict(), f'RL/training/value_models/{file_name}.pth')

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
                   healthy_angle_range=(-0.2, 0.2),
                   healthy_state_range=(-100, float("inf")),
                   healthy_z_range=(0.4, float("inf")),
                   jump_reward_weight=0.4,
                   ctrl_cost_weight=1e-3)
    
    return env

def create_hopper_easy(render_mode=True):
    env = gym.make("Hopper-v5", 
                   max_episode_steps=10000,
                   render_mode=render_mode, 
                   healthy_angle_range=(-1, 1),
                   healthy_state_range=(-100, float("inf")),
                   healthy_z_range=(0.3, float("inf")),
                   ctrl_cost_weight=1e-3)
    return env


def inference_hopper(file_name, render=True, policy_type='gaussian'):
    render_mode = 'human' if render else None
    env = create_hopper_easy(render_mode=render_mode)

    # Policy network and optimizer
    policy = create_policy(policy_type)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

def plot_log(file_name):
    logger = Logger()
    logger.load_from_file(f'RL/training/log/{file_name}.pkl')
    ui = LoggerUI(logger)
    ui.run()

def run_sb_PPO():
    from stable_baselines3 import PPO

    # Create environment for training (without rendering)
    env = gym.make("Hopper-v5", render_mode="human")
    
    # Initialize and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model
    model.save("RL/training/models/ppo_hopper")
    print("Model saved.")

    # Close the training environment
    env.close()

def inference_sb_PPO():
    from stable_baselines3 import PPO

    # Now load the model and visualize
    loaded_model = PPO.load("RL/training/models/ppo_hopper")
    
    render_env = gym.make("Hopper-v5", render_mode="human")
    obs, _ = render_env.reset()  # Fix 1: Unpack tuple

    for _ in range(1000):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)  # Fix 2: Gym v0.26+ step output
        done = terminated or truncated

        if done:
            obs, _ = render_env.reset()  # Reset and unpack again

    render_env.close()

if __name__=='__main__':
    #  train(load=False, seed=64665, file_name="hopper_ppo_gaussian", algorithm_name="PPO", 
    #       policy_type='gaussian', num_episodes=5000, max_steps=1000, set_policy_std=0.4, 
    #       entropy_coef=0.02, show=True, render=True)
     
    #  plot_log(file_name="hopper_ppo_gaussian")
    
    # train(load=True, seed=27354, file_name="hopper_ppo_gaussian", 
    #       start_policy_name="hopper_ppo_gaussian_best2", algorithm_name="PPO", 
    #       num_episodes=10000, max_steps=1000, set_policy_std=None, entropy_coef=0.01, show=True, render=True)

    inference_hopper(file_name="hopper_ppo_gaussian", policy_type='gaussian', render=True)

    # run_sb_PPO()
    # inference_sb_PPO()
