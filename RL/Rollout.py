import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

class SimpleRollout:
    def __init__(self, env) -> None:
        self.env = env

    def rollout(self, num_steps, policy, max_steps_per_episode=1000):
        state, _ = self.env.reset()

        transitions = []
        steps_in_episode = 0

        for steps in range(num_steps):
            state_t = torch.tensor(state, dtype=torch.float32)
            action = policy(state_t).detach().numpy()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            steps_in_episode += 1

            state = next_state

            if done or steps_in_episode >= max_steps_per_episode:
                state, _ = self.env.reset()
                steps_in_episode = 0

        return transitions
    
    def eval_rollout(self, n_episode, policy, max_steps_per_episode=1000, gamma=0.99):
        episode_returns = []
        episode_length = []

        for episode in range(n_episode):
            state, _ = self.env.reset()
            rewards = []
            for step in range(max_steps_per_episode):
                state_t = torch.tensor(state, dtype=torch.float32)
                action = policy(state_t).detach().numpy()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                rewards.append(reward)

                state = next_state

                if done:
                    break

            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
            episode_returns.append(G)
            episode_length.append(len(rewards))

        return episode_returns, episode_length