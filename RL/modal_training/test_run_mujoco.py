import modal
import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RL.modal_training.image_definitions import IMAGES

image = IMAGES["default"].add_local_python_source("RL")

App = modal.App("ppo_walker", image=image)

model_volume = modal.Volume.from_name("models")
value_model_volume = modal.Volume.from_name("value_models")
log_volume = modal.Volume.from_name("log")

@App.function(
    timeout=12*60*60,
    volumes={"/models": model_volume, "/value_models": value_model_volume, "/log": log_volume},
)

def train():
    import gymnasium as gym
    from RL.PPO import PPO
    import mujoco
    import torch 
    import random
    from RL.Logger import Logger
    from RL.ValueNetwork import SimpleValueFunction
    from RL.PolicyNetwork import GaussianPolicy

    ############### Training Setting ########################
    seed = 2934
    random.seed(seed)
    torch.manual_seed(seed)

    lr = 1e-3
    total_num_steps = 10_000_000
    save_file_name = f'ppo_walker_{total_num_steps}_steps'
    env_name = "Walker2d-v5"

    ##########################################################

    # Create environment with Gymnasium
    env = gym.make(env_name, render_mode=None)
    
    policy = GaussianPolicy(state_dim=17, action_dim=6, hidden_dims=[128, 64], 
                            std_init=0.2, std_min=1e-4, std_max=0.6, temperature_decay=1) 
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Value network
    value_net = SimpleValueFunction(state_dim=17, hidden_dims=[128, 64])
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    logger = Logger()
    
    # Initialize and train model
    ppo = PPO(
        env=env, 
        policy=policy, 
        policy_optimizer=policy_optimizer, 
        value_func=value_net, 
        value_optimizer=value_optimizer, 
        total_num_steps=total_num_steps, 
        max_steps_per_episode=200, 
        gamma=0.99, 
        lambda_decay=0.95, 
        entropy_coef=0.02,
        n_step_per_update=4096,
        batch_size=128, 
        n_epoch=10, 
        epsilon=0.2,
        value_func_epsilon=None,
        kl_threshold=0.1,
        logger=logger,
        visualize_every=10,
        visualize_env=None)
    
    ppo.train()
    env.close()

    # Save the policy
    torch.save(policy.state_dict(), f'/models/{save_file_name}.pth')

    # Save the value network
    torch.save(value_net.state_dict(), f'/value_models/val_{save_file_name}.pth')

    logger.save_to_file(f'/log/{save_file_name}.pkl')

@App.local_entrypoint()
def main():
    train.remote()
