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
from RL.ValueNetwork import SimpleValueFunction
from RL.PolicyNetwork import GaussianPolicy
from RL.training.common_double_integrator import *
from RL.PPO import PPO
from RL.Logger import Logger
from RL.LoggerUI import LoggerUI
import gymnasium as gym
import torch

def train(load, seed, file_name, start_policy_name=None, num_steps=100, max_steps_per_episode=1000, 
          env_name="hopper", policy_type='gaussian', set_policy_std=0, entropy_coef=0.01,
          show=False, render=False):
    random.seed(seed)
    torch.manual_seed(seed)

    render_mode = 'human' if render else None
    logger = Logger()

    visualize_env = None
    if env_name == "hopper":
        env = create_hopper(render_mode=render_mode)
        visualize_env = create_hopper(render_mode='human')

        # Policy network and optimizer
        policy = create_hopper_policy(policy_type)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Value network
        value_net = SimpleValueFunction(state_dim=11, hidden_dims=[64, 64])
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    elif env_name == "half_cheetah":
        env = create_half_cheetah(render_mode=render_mode)
        visualize_env = create_half_cheetah(render_mode='human')

        # Policy network and optimizer
        policy = create_half_cheetah_policy()
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        # Value network
        value_net = SimpleValueFunction(state_dim=17, hidden_dims=[64, 64])
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    elif env_name == "walker":
        env = create_walker(render_mode=render_mode)
        visualize_env = create_walker(render_mode='human')

        # Policy network and optimizer
        policy = create_walker_policy()
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=4e-4)
        # Value network
        value_net = SimpleValueFunction(state_dim=17, hidden_dims=[64, 64])
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=4e-4)

    elif env_name == "ant":
        env = create_ant(render_mode=render_mode)
        visualize_env = create_ant(render_mode='human')

        # Policy network and optimizer
        policy = create_ant_policy()
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # Value network
        value_net = SimpleValueFunction(state_dim=105, hidden_dims=[128, 64])
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    if load:
        if start_policy_name:
            policy.load_state_dict(torch.load(f'models/{start_policy_name}.pth'))
            value_net.load_state_dict(torch.load(f'value_models/{start_policy_name}.pth'))
            logger.load_from_file(f'log/{start_policy_name}.pkl')
            logger.set_episode_offset(logger.get_max_episode() + 1)
            logger.set_update_round_offset(logger.get_max_update_round() + 1)
        else:
            policy.load_state_dict(torch.load(f'models/{file_name}.pth'))
            value_net.load_state_dict(torch.load(f'value_models/{file_name}.pth'))
            logger.load_from_file(f'log/{file_name}.pkl')
            logger.set_episode_offset(logger.get_max_episode() + 1)
            logger.set_update_round_offset(logger.get_max_update_round() + 1)

    if set_policy_std and set_policy_std > 0 and isinstance(policy, GaussianPolicy):
        policy.set_std(set_policy_std)

    algorithm = PPO(
        env=env, 
        policy=policy, 
        policy_optimizer=policy_optimizer, 
        value_func=value_net, 
        value_optimizer=value_optimizer, 
        total_num_steps=num_steps, 
        max_steps_per_episode=max_steps_per_episode, 
        gamma=0.99, 
        lambda_decay=0.95, 
        entropy_coef=entropy_coef,
        n_step_per_update=4096,
        batch_size=128, 
        n_epoch=10, 
        epsilon=0.2,
        value_func_epsilon=None,
        kl_threshold=0.1,
        visualize_every=10,
        visualize_env=visualize_env,
        tensorboard_log_dir=f'log/{file_name}',
        verbose_logging=True)

    algorithm.train()
    env.close()
    if visualize_env:
        visualize_env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'models/{file_name}.pth')

    # Save the value network
    torch.save(value_net.state_dict(), f'value_models/{file_name}.pth')

    if show:
        # visualize_policy(policy)
        logger.save_to_file(f'log/{file_name}.pkl')
        ui = LoggerUI(logger)
        ui.run()

############################################
### Hopper
############################################

def create_hopper_policy(policy_type):
    if policy_type == 'gaussian':
        policy = GaussianPolicy(state_dim=11, action_dim=3, hidden_dims=[64, 64], 
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1)
    elif policy_type == 'gaussian_decay':
        policy = GaussianPolicy(state_dim=11, action_dim=3, hidden_dims=[64, 64], 
                                std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1-1e-7)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policy

def create_hopper(render_mode=None):
    env = gym.make("Hopper-v5", 
                   render_mode=render_mode, 
                   healthy_angle_range=(-0.2, 0.2),
                   healthy_state_range=(-100, float("inf")),
                   healthy_z_range=(0.4, float("inf")),
                   jump_reward_weight=0.1,
                   ctrl_cost_weight=1e-3)
    
    return env

def create_hopper_easy(render_mode=None):
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
    policy = create_hopper_policy(policy_type)
    policy.load_state_dict(torch.load(f'models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

############################################
### Half Cheetah
############################################

def create_half_cheetah(render_mode=None):
    env = gym.make("HalfCheetah-v5", 
                   forward_reward_weight=1.0,
                   ctrl_cost_weight=0.5,
                   render_mode=render_mode)
    return env

def create_half_cheetah_policy():
    policy = GaussianPolicy(state_dim=17, action_dim=6, hidden_dims=[64, 64],
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1)
    
    return policy

def inference_half_cheetah(file_name, render=True):
    render_mode = 'human' if render else None
    env = create_half_cheetah(render_mode=render_mode)

    # Policy network and optimizer
    policy = create_half_cheetah_policy()
    policy.load_state_dict(torch.load(f'models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

############################################
### Walker 2d
############################################
def create_walker_policy():
    policy = GaussianPolicy(state_dim=17, action_dim=6, hidden_dims=[64, 64], 
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1)
    
    return policy

def create_walker(render_mode=None):
    env = gym.make("Walker2d-v5", 
                   render_mode=render_mode, 
                   forward_reward_weight=1.0,
                   thigh_movement_reward_weight=0.01,
                   healthy_angle_range=(-1, 1),
                   healthy_z_range=(0.8, 3.0),
                   ctrl_cost_weight=0.05)
    
    return env

def inference_walker(file_name, render=True):
    render_mode = 'human' if render else None
    env = create_walker(render_mode=render_mode)

    # Policy network and optimizer
    policy = create_walker_policy()
    policy.load_state_dict(torch.load(f'models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

############################################
### Ant
############################################
def create_ant_policy():
    policy = GaussianPolicy(state_dim=105, action_dim=8, hidden_dims=[128, 64], 
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1)
    
    return policy

def create_ant(render_mode=None):
    env = gym.make("Ant-v5", 
                   render_mode=render_mode, 
                   forward_reward_weight=1.0,
                   contact_cost_weight=5e-4,
                   healthy_z_range=(0.8, 3.0),
                   ctrl_cost_weight=0.2)
    
    return env

def inference_ant(file_name, render=True):
    render_mode = 'human' if render else None
    env = create_ant(render_mode=render_mode)

    # Policy network and optimizer
    policy = create_ant_policy()
    policy.load_state_dict(torch.load(f'models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()


############################################
### Stable baselines3
############################################

def run_sb_PPO(env_name="walker", load=False):
    from stable_baselines3 import PPO

    # Create environment for training (without rendering)

    if env_name == "hopper":
        env = gym.make("Hopper-v5", render_mode="human")

    elif env_name == "walker":
        env = gym.make("Walker2d-v5", 
                   render_mode='human', 
                   forward_reward_weight=1.0,
                   thigh_movement_reward_weight=0.05,
                   healthy_z_range=(0.8, 3.0),
                   ctrl_cost_weight=0.05)
        
    elif env_name == "ant":
        env = gym.make("Ant-v5", 
                   render_mode='human', 
                   forward_reward_weight=1.0,
                   contact_cost_weight=5e-4,
                   healthy_z_range=(0.8, 3.0),
                   ctrl_cost_weight=0.05)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Initialize and train the model
    if load:
        model = PPO.load(f"models/ppo_{env_name}", env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model
    # model.save("models/ppo_hopper")
    model.save(f"models/stable_baseline_ppo_{env_name}")
    print("Model saved.")

    # Close the training environment
    env.close()

def inference_sb_PPO(env_name="walker"):
    from stable_baselines3 import PPO

    # Now load the model and visualize
    loaded_model = PPO.load(f"models/ppo_{env_name}")
    
    if env_name == "hopper":
        render_env = gym.make("Hopper-v5", render_mode="human")
    elif env_name == "walker":
        render_env = gym.make("Walker2d-v5", render_mode="human")
    elif env_name == "ant":
        render_env = gym.make("Ant-v5", render_mode="human")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    obs, _ = render_env.reset()  # Fix 1: Unpack tuple

    for _ in range(10000):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)  # Fix 2: Gym v0.26+ step output
        done = terminated or truncated

        if done:
            obs, _ = render_env.reset()  # Reset and unpack again

    render_env.close()

if __name__=='__main__':
    #  train(load=False, seed=64665, file_name="hopper_ppo_gaussian", 
    #       env_name="hopper", policy_type='gaussian', num_steps=1_000_000, max_steps_per_episode=500, set_policy_std=0.4, 
    #       entropy_coef=0.02, show=True, render=False)
     
    #  plot_log(file_name="hopper_ppo_gaussian")
    
    # train(load=True, seed=27354, file_name="hopper_ppo_gaussian", env_name="hopper",
    #       start_policy_name="hopper_ppo_gaussian_best2", 
    #       num_steps=5000, max_steps_per_episode=500, set_policy_std=None, entropy_coef=0.01, show=True, render=True)

    # inference_hopper(file_name="hopper_ppo_gaussian_best", policy_type='gaussian', render=True)

    # Half Cheetah

    train(load=False, seed=6335, file_name="half_cheetah_ppo_gaussian", 
          env_name="half_cheetah", num_steps=1_000_000, max_steps_per_episode=200,
          set_policy_std=None, entropy_coef=0.02, show=True, render=False)
    
    # inference_half_cheetah(file_name="half_cheetah_ppo_gaussian")

    # plot_log(file_name="half_cheetah_ppo_gaussian_5M_samples")

    # Walker 2d
    # train(load=False, seed=23795, file_name="walker_ppo_gaussian",
    #       env_name="walker", policy_type='gaussian', num_steps=2_000_000, max_steps_per_episode=500, set_policy_std=0.4, 
    #       entropy_coef=0.02, show=True, render=True)

    # train(load=True, seed=843684, file_name="walker_ppo_gaussian",
    #       env_name="walker", policy_type='gaussian', num_steps=2_000_000, max_steps_per_episode=500, set_policy_std=0.2, 
    #       start_policy_name="walker_ppo_gaussian", entropy_coef=0.02, show=True, render=True)
     
    # inference_walker(file_name="walker_ppo_gaussian")

    # plot_log(file_name="walker_ppo_gaussian")

    # run_sb_PPO(env_name="walker", load=True)
    # inference_sb_PPO(env_name="walker")

    # Ant
    # train(load=False, seed=22795, file_name="ant_ppo_gaussian",
    #       env_name="ant", policy_type='gaussian', num_steps=500_000, max_steps_per_episode=500, 
    #       set_policy_std=0.4, entropy_coef=0.0, show=True, render=True)

    # inference_ant(file_name="ant_ppo_gaussian")

    # plot_log(file_name="ant_ppo_gaussian")

    # run_sb_PPO(env_name="ant", load=False)
    inference_sb_PPO()
