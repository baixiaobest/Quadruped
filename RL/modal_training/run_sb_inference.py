import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gymnasium as gym
from stable_baselines3 import TD3
from RL.PolicyNetwork import DeterministicContinuousPolicy
import torch
from RL.training.common_double_integrator import inference

def sb_load_and_run_inference(model_path, env_name, max_steps=1000):
    # Now load the model and visualize
    loaded_model = TD3.load(model_path)
    
    if env_name == "hopper":
        render_env = gym.make("Hopper-v5", render_mode="human")
    elif env_name == "half_cheetah":
        render_env = gym.make("HalfCheetah-v5", render_mode="human")
    elif env_name == "walker":
        render_env = gym.make("Walker2d-v5", render_mode="human")
    elif env_name == "ant":
        render_env = gym.make("Ant-v5", render_mode="human")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    obs, _ = render_env.reset()  # Fix 1: Unpack tuple

    rewards = []

    for _ in range(max_steps):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)  # Fix 2: Gym v0.26+ step output
        done = terminated or truncated
        rewards.append(reward)

        if done:
            obs, _ = render_env.reset()  # Reset and unpack again
            break

    print(f"Total reward: {sum(rewards)}")

    render_env.close()

def custom_load_and_run_cheetah_inference(model_path, env_name, max_steps=1000):
    env = gym.make("HalfCheetah-v5", render_mode='human')
    state_dim = 17
    action_dim = 6
    # Policy network and optimizer
    policy = DeterministicContinuousPolicy(state_dim, action_dim, hidden_dims=[128, 64])
    policy.load_state_dict(torch.load(model_path))

    rewards, _, _ = inference(policy, env, max_step=max_steps, continue_on_terminate=True, deterministic=True)

    print(f"Total reward: {sum(rewards)}")

if __name__ == "__main__":
    # model_path = "models/modal_training/sb_td3_half_cheetah_lr_1e-4_3000000_steps_2025-04-24_06-10-53"  # Replace with your model path
    # env_name = "half_cheetah"  # Replace with your environment name
    # sb_load_and_run_inference(model_path, env_name)

    custom_load_and_run_cheetah_inference("models/modal_training/td3_custom_half_cheetah_lr_1e-4_600000_epochs_2025-05-05_08-21-49.pth", "HalfCheetah-v5")

# Example usage:
# load_and_run_inference("/models/your_model_2023-09-30_12-34-56.zip", "Pendulum-v1")