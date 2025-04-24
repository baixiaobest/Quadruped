import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gymnasium as gym
from stable_baselines3 import TD3

def load_and_run_inference(model_path, env_name):
    # Now load the model and visualize
    loaded_model = TD3.load(model_path)
    
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

if __name__ == "__main__":
    # Example usage
    model_path = "models/modal_training/sb_td3_hopper_3000000_steps_2025-04-23_14-18-41"  # Replace with your model path
    env_name = "hopper"  # Replace with your environment name
    load_and_run_inference(model_path, env_name)

# Example usage:
# load_and_run_inference("/models/your_model_2023-09-30_12-34-56.zip", "Pendulum-v1")