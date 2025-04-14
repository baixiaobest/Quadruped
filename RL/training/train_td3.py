import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 1. Set display backend variables (MUST come before any gymnasium/mujoco import)
os.environ["DISPLAY"] = ":0"           # Forces X11 display
os.environ["XDG_SESSION_TYPE"] = "x11" # Explicitly use X11 session
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Qt platform plugin for X11

from RL.TD3 import TD3
from RL.Logger import Logger
from RL.LoggerUI import LoggerUI
from RL.PolicyNetwork import DeterministicContinuousPolicy
from RL.ValueNetwork import SimpleQFunction
from RL.training.common_double_integrator import *
import gymnasium as gym
import torch
import random

def train(load, seed, file_name, algorithm_name="td3", start_policy_name=None, num_epoch=1000, 
          env_name="inverted_pendulum", show=True, render=True):
    
    random.seed(seed)
    torch.manual_seed(seed)

    render_mode = 'human' if render else None

    state_dim = 0
    action_dim = 0
    if env_name=="inverted_pendulum":
        env = gym.make("InvertedPendulum-v5", render_mode=render_mode)
        state_dim = 4
        action_dim = 1

    logger = Logger()

    algorithm = None
    if algorithm_name == "td3":
        policy = DeterministicContinuousPolicy(state_dim, action_dim, hidden_dims=[64, 64])
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
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
            n_epoch=num_epoch, 
            max_steps_per_episode=500, 
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
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    if load:
        if start_policy_name:
            policy.load_state_dict(torch.load(f'RL/training/models/{start_policy_name}.pth'))
            Q1.load_state_dict(torch.load(f'RL/training/value_models/{start_policy_name}_Q1.pth'))
            Q2.load_state_dict(torch.load(f'RL/training/value_models/{start_policy_name}_Q2.pth'))
        else:
            policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
            Q1.load_state_dict(torch.load(f'RL/training/value_models/{file_name}_Q1.pth'))
            Q2.load_state_dict(torch.load(f'RL/training/value_models/{file_name}_Q2.pth'))

    algorithm.train()
    env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'RL/training/models/{file_name}.pth')

    # Save the Q value network
    torch.save(Q1.state_dict(), f'RL/training/value_models/{file_name}_Q1.pth')
    torch.save(Q2.state_dict(), f'RL/training/value_models/{file_name}_Q2.pth')

    if show:
        logger.save_to_file(f'RL/training/log/{file_name}.pkl')
        ui = LoggerUI(logger)
        ui.run()
    
    logger.save_to_file(f'RL/training/log/{file_name}.pkl')

def inference_inverted_pendulum(file_name, render=True):
    render_mode = 'human' if render else None
    env = env = gym.make("InvertedPendulum-v5", render_mode=render_mode)
    state_dim = 4
    action_dim = 1
    # Policy network and optimizer
    policy = DeterministicContinuousPolicy(state_dim, action_dim, hidden_dims=[64, 64])
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

if __name__=="__main__":
    # train(load=False, seed=0, file_name="td3_inverted_pendulum", algorithm_name="td3", start_policy_name=None, 
    #       num_epoch=26_000, show=True, render=True)
    
    # plot_log(file_name="td3_inverted_pendulum")

    inference_inverted_pendulum(file_name="td3_inverted_pendulum", render=True)
