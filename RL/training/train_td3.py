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

def train(load, load_log, seed, file_name, algorithm_name="td3", start_policy_name=None, num_epoch=1000, 
          env_name="inverted_pendulum", max_steps_per_episode=500, show=True, verbose_logging=False):
    
    random.seed(seed)
    torch.manual_seed(seed)

    state_dim = 0
    action_dim = 0
    visualize_env = None
    if env_name=="inverted_pendulum":
        training_env = gym.make("InvertedPendulum-v5", render_mode=None)
        visualize_env = gym.make("InvertedPendulum-v5", render_mode='human')
        state_dim = 4
        action_dim = 1

    elif env_name=="half_cheetah":
        training_env = gym.make("HalfCheetah-v5", render_mode=None)
        visualize_env = gym.make("HalfCheetah-v5", render_mode='human')
        state_dim = 17
        action_dim = 6
    elif env_name=="walker":
        training_env = gym.make("Walker2d-v5", render_mode=None)
        visualize_env = gym.make("Walker2d-v5", render_mode='human')
        state_dim = 17
        action_dim = 6
    elif env_name=="hopper":
        training_env = gym.make("Hopper-v5", 
                       render_mode=None,
                       jump_reward_weight=0.2)
        visualize_env = gym.make("Hopper-v5", render_mode='human')
        state_dim = 11
        action_dim = 3
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    logger = Logger()
    init_policy = 'uniform'
    if load:
        init_policy = 'current'

    def get_eval_callback():
        max_val = 180
        def eval_callback(epoch, R, policy, Q1, Q2):
            nonlocal max_val
            if epoch > 0 and R > max_val + 10:
                print("saving model")
                max_val = R
                # Save the policy
                torch.save(policy.state_dict(), f'RL/training/models/{file_name}_R{R:.0f}.pth')

                # Save the Q value network
                torch.save(Q1.state_dict(), f'RL/training/value_models/{file_name}_R{R:.0f}_Q1.pth')
                torch.save(Q2.state_dict(), f'RL/training/value_models/{file_name}_R{R:.0f}_Q2.pth')
        return eval_callback

    algorithm = None
    if algorithm_name == "td3":
        policy = create_policy(state_dim, action_dim)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
        Q1 = SimpleQFunction(state_dim, action_dim, hidden_dims=[64, 64])
        Q1_optimizer = torch.optim.Adam(Q1.parameters(), lr=5e-4)
        Q2 = SimpleQFunction(state_dim, action_dim, hidden_dims=[64, 64])
        Q2_optimizer = torch.optim.Adam(Q2.parameters(), lr=5e-4)

        algorithm = TD3(
            training_env, 
            policy, 
            policy_optimizer, 
            Q1, 
            Q1_optimizer, 
            Q2, 
            Q2_optimizer, 
            logger,
            n_epoch=num_epoch, 
            max_steps_per_episode=max_steps_per_episode, 
            init_buffer_size=50_000, 
            init_policy=init_policy,
            update_every=50, 
            eval_every=50, 
            eval_episode=5, 
            batch_size=300, 
            replay_buffer_size=1e6, 
            policy_delay=2, 
            gamma=0.99, 
            polyak=0.995, 
            action_noise_config={
                'type': 'OU',
                'theta': 0.15,
                'sigma': 0.2,
                'dt': 2e-3,
                'noise_clip': 0.3,
            }, 
            # action_noise_config={
            #     'type': 'gaussian',
            #     'sigma': 0.2,
            #     'noise_clip': 0.4,
            # }, 
            target_noise=0.2, 
            eval_callback=get_eval_callback(),
            true_q_estimate_every=2000,
            true_q_estimate_episode=50,
            verbose_logging=verbose_logging,
            visualize_env=visualize_env,
            visualize_every=2000)
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    

    if load:
        if start_policy_name:
            policy.load_state_dict(torch.load(f'RL/training/models/{start_policy_name}.pth'))
            Q1.load_state_dict(torch.load(f'RL/training/value_models/{start_policy_name}_Q1.pth'))
            Q2.load_state_dict(torch.load(f'RL/training/value_models/{start_policy_name}_Q2.pth'))
            if load_log:
                logger.load_from_file(f'RL/training/log/{start_policy_name}.pkl')
                logger.set_episode_offset(logger.get_max_episode() + 1)
                logger.set_update_round_offset(logger.get_max_update_round() + 1)
        else:
            policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))
            Q1.load_state_dict(torch.load(f'RL/training/value_models/{file_name}_Q1.pth'))
            Q2.load_state_dict(torch.load(f'RL/training/value_models/{file_name}_Q2.pth'))
            if load_log:
                logger.load_from_file(f'RL/training/log/{file_name}.pkl')
                logger.set_episode_offset(logger.get_max_episode() + 1)
                logger.set_update_round_offset(logger.get_max_update_round() + 1)

    algorithm.train()
    training_env.close()
    visualize_env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'RL/training/models/{file_name}.pth')

    # Save the Q value network
    torch.save(Q1.state_dict(), f'RL/training/value_models/{file_name}_Q1.pth')
    torch.save(Q2.state_dict(), f'RL/training/value_models/{file_name}_Q2.pth')

    logger.save_to_file(f'RL/training/log/{file_name}.pkl')
    
    if show:
        ui = LoggerUI(logger)
        ui.run()

def create_policy(state_dim, action_dim):
    policy = DeterministicContinuousPolicy(state_dim, action_dim, hidden_dims=[64, 64])
    return policy

def inference_inverted_pendulum(file_name, render=True):
    render_mode = 'human' if render else None
    env = env = gym.make("InvertedPendulum-v5", render_mode=render_mode)
    state_dim = 4
    action_dim = 1
    # Policy network and optimizer
    policy = create_policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

def inference_half_cheetah(file_name, render=True):
    render_mode = 'human' if render else None
    env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    state_dim = 17
    action_dim = 6
    # Policy network and optimizer
    policy = create_policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

def inference_walker(file_name, render=True):
    render_mode = 'human' if render else None
    env = gym.make("Walker2d-v5", render_mode=render_mode)
    state_dim = 17
    action_dim = 6
    # Policy network and optimizer
    policy = create_policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

def inference_hopper(file_name, render=True):
    render_mode = 'human' if render else None
    env = gym.make("Hopper-v5", render_mode=render_mode)
    state_dim = 11
    action_dim = 3
    # Policy network and optimizer
    policy = create_policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(f'RL/training/models/{file_name}.pth'))

    inference(policy, env, max_step=10000, continue_on_terminate=True, deterministic=True)

    env.close()

if __name__=="__main__":

    # Inverted pendulum

    # train(load=False, load_log=False, seed=7846, file_name="td3_inverted_pendulum", algorithm_name="td3", start_policy_name=None, 
    #       env_name="inverted_pendulum", num_epoch=30_000, max_steps_per_episode=200, show=True)
    
    # plot_log(file_name="td3_inverted_pendulum")

    # inference_inverted_pendulum(file_name="td3_inverted_pendulum", render=True)

    # Half cheetah

    # train(load=False, seed=546221, file_name="td3_half_cheetah", algorithm_name="td3", start_policy_name=None, 
    #       env_name="half_cheetah", num_epoch=100_000, max_steps_per_episode=200, show=True)
    
    # inference_half_cheetah(file_name="td3_half_cheetah", render=True)

    # plot_log(file_name="td3_half_cheetah")

    # Hopper

    train(load=True, load_log=False, seed=53587, file_name="td3_hopper", algorithm_name="td3", 
          start_policy_name="td3_hopper_R232", env_name="hopper", num_epoch=20_000, max_steps_per_episode=500, 
          show=True, verbose_logging=False)
    
    # inference_hopper(file_name="td3_hopper", render=True)

    # plot_log(file_name="td3_hopper")

