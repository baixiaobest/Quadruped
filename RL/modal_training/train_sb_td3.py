import modal
import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
from RL.modal_training.image_definitions import IMAGES


############### Training Setting ########################
PREFIX_NAME = "sb_td3_half_cheetah"
env_name = "HalfCheetah-v5"

lr_multiplier = 1
lr_base = -4
lr = lr_multiplier * 10 ** lr_base

total_num_steps = 3_000_000

save_file_name = f'{PREFIX_NAME}_lr_{lr_multiplier}e{lr_base}_{total_num_steps}_steps'
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
    from stable_baselines3 import TD3

    seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    torch.manual_seed(seed)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create environment with Gymnasium
    env = gym.make(env_name, render_mode=None)
    
    # Initialize and train model
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=int(1e6),
        learning_starts=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(100, "step"),
        gradient_steps=20,
        action_noise=None,
        policy_kwargs=dict(net_arch=[128, 64]),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"/log/tensorboard/{save_file_name}_{current_time}",
    )
    model.learn(total_timesteps=total_num_steps, log_interval=10)

    model.save(f'/models/{save_file_name}_{current_time}.zip')
    # Save the model

   
    env.close()

@App.local_entrypoint()
def main():
    train.remote()
