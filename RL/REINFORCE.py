import torch

class REINFORCE:
    def __init__(self, env, policy, optimizer, num_episodes=1000, max_steps=100, gamma=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        for episode in range(self.num_episodes):
            self.env.reset()
            state = self.env.get_state()
            rewards = []
            log_probs = []

            for step in range(self.max_steps):
                state_t = torch.tensor(state, dtype=torch.float32)

                actions_prob = self.policy.forward(state_t)
                dist = torch.distributions.Categorical(actions_prob)

                action_idx = dist.sample()
                
                action_log_prob = dist.log_prob(action_idx)

                next_state, reward, done = self.env.step(self.policy.get_action(action_idx).item())

                log_probs.append(action_log_prob)
                rewards.append(reward)

                state = next_state
                
                if done:
                    break
                
            if step == self.max_steps:
                print(f"max step reached at episode {episode}")

            returns = []
            G = 0

            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)

            returns.reverse()

            # Update step
            self.optimizer.zero_grad()

            loss = torch.tensor(0, dtype=torch.float32)
            for log_prob, r in zip(log_probs, returns):
                loss += -log_prob*r

            loss.backward()
            self.optimizer.step()

            self.return_list.append(returns[0])

            print(f"episode {episode} return: {returns[0]}")

    def get_returns_list(self):
        return self.return_list
