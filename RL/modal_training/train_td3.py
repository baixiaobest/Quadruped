import modal
import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from RL.modal_training.image_definitions import IMAGES

############### Training Setting ########################

env_name = "half_cheetah"
PREFIX_NAME = f"td3_custom_{env_name}"

lr_multiplier = 1
lr_base = -4
lr = lr_multiplier * 10 ** lr_base

LR_SUFFIX = f"lr_{lr_multiplier}e{lr_base}"

num_epoch = 600_000

EPOCH_SUFFIX = f"{num_epoch}_epochs"

file_name = f'{PREFIX_NAME}_{LR_SUFFIX}_{EPOCH_SUFFIX}'
##########################################################

image = IMAGES["default"].add_local_python_source("RL")

App = modal.App(PREFIX_NAME, image=image)

model_volume = modal.Volume.from_name("models")
value_model_volume = modal.Volume.from_name("value_models")
log_volume = modal.Volume.from_name("log")

@App.function(
    timeout=12*60*60,
    volumes={"/models": model_volume, "/value_models": value_model_volume, "/log": log_volume},
)
def train():
    import gymnasium as gym
    import mujoco
    import torch 
    import random
    from RL.TD3 import TD3
    from datetime import datetime
    from RL.PolicyNetwork import DeterministicContinuousPolicy
    from RL.ValueNetwork import SimpleQFunction

    seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    torch.manual_seed(seed)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if env_name=="inverted_pendulum":
        training_env = gym.make("InvertedPendulum-v5", render_mode=None)
        eval_env = gym.make("InvertedPendulum-v5", render_mode=None)
        state_dim = 4
        action_dim = 1
    elif env_name=="half_cheetah":
        training_env = gym.make("HalfCheetah-v5", render_mode=None)
        eval_env = gym.make("HalfCheetah-v5", render_mode=None)
        state_dim = 17
        action_dim = 6
    elif env_name=="walker":
        training_env = gym.make("Walker2d-v5", render_mode=None)
        eval_env = gym.make("Walker2d-v5", render_mode=None)
        state_dim = 17
        action_dim = 6
    elif env_name=="hopper":
        training_env = gym.make("Hopper-v5", 
                       render_mode=None,
                       jump_reward_weight=0.2)
        eval_env = gym.make("Hopper-v5",
                       render_mode=None,
                       jump_reward_weight=0.2)
        state_dim = 11
        action_dim = 3
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    policy = DeterministicContinuousPolicy(state_dim, action_dim, hidden_dims=[128, 64])
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    Q1 = SimpleQFunction(state_dim, action_dim, hidden_dims=[128, 64])
    Q1_optimizer = torch.optim.Adam(Q1.parameters(), lr=lr)
    Q2 = SimpleQFunction(state_dim, action_dim, hidden_dims=[128, 64])
    Q2_optimizer = torch.optim.Adam(Q2.parameters(), lr=lr)

    algorithm = TD3(
        training_env, 
        eval_env,
        state_dim,
        action_dim,
        policy, 
        policy_optimizer, 
        Q1, 
        Q1_optimizer, 
        Q2, 
        Q2_optimizer, 
        tensorboard_log_dir=f'/log/tensorboard/{file_name}_{current_time}',
        log_every=1000,
        n_epoch=num_epoch, 
        max_steps_per_episode=1000, 
        init_buffer_size=50_000, 
        init_policy="uniform",
        rollout_steps=100,
        update_per_rollout=20,
        eval_every=1000, 
        eval_episode=1, 
        batch_size=256, 
        replay_buffer_size=1e6, 
        policy_delay=2, 
        gamma=0.99, 
        polyak=0.995, 
        # action_noise_config={
        #     'type': 'OU',
        #     'theta': 0.15,
        #     'sigma': 0.2,
        #     'dt': 2e-3,
        #     'noise_clip': 0.5,
        # }, 
        action_noise_config={
            'type': 'gaussian',
            'sigma': 0.2,
            'noise_clip': 0.5,
        }, 
        target_noise=0.2, 
        verbose_logging=False)
    
    algorithm.train()
    
    training_env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'/models/{file_name}_{current_time}.pth')

    # Save the Q value network
    torch.save(Q1.state_dict(), f'/value_models/Q1_{file_name}_{current_time}.pth')
    torch.save(Q2.state_dict(), f'/value_models/Q2_{file_name}_{current_time}.pth')

@App.local_entrypoint()
def main():
    train.remote()
