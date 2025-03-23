import torch

class REINFORCE:
    def __init__(self, env, policy, policy_optimizer, value_func=None, value_optimizer=None, num_episodes=1000, max_steps=100, gamma=0.99):
        self.env = env
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_func = value_func
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.return_list = []
    
    def train(self):
        if self.value_func is None:
            return self.train_no_baseline()
        else:
            return self.train_baseline()

    def train_no_baseline(self):
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
            self.policy_optimizer.zero_grad()

            loss = torch.tensor(0, dtype=torch.float32)
            discount_0_to_t = 1.0
            for log_prob, r in zip(log_probs, returns):
                loss += -log_prob*r*discount_0_to_t
                discount_0_to_t *= self.gamma

            loss.backward()
            self.policy_optimizer.step()

            self.return_list.append(returns[0])

            print(f"episode {episode} return: {returns[0]}")

    def train_baseline(self):
        self.policy.train()
        self.value_func.train()
        self.return_list = []

        for episode in range(self.num_episodes):
            self.env.reset()
            state = self.env.get_state()
            rewards = []
            log_probs = []
            values=[]

            for step in range(self.max_steps):
                state_t = torch.tensor(state, dtype=torch.float32)

                actions_prob = self.policy.forward(state_t)
                dist = torch.distributions.Categorical(actions_prob)

                action_idx = dist.sample()
                
                action_log_prob = dist.log_prob(action_idx)

                next_state, reward, done = self.env.step(self.policy.get_action(action_idx).item())

                log_probs.append(action_log_prob)
                rewards.append(reward)
                values.append(self.value_func(state_t).squeeze())

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
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            policy_loss = torch.tensor(0, dtype=torch.float32)
            discount_0_to_t = 1.0
            for log_prob, G, val in zip(log_probs, returns, values):
                policy_loss = policy_loss - log_prob * (G - val.item()) * discount_0_to_t
                discount_0_to_t *= self.gamma
            
            value_loss = torch.tensor(0, dtype=torch.float32)
            for G, val in zip(returns, values):
                value_loss += (G - val).pow(2)

            policy_loss.backward()
            self.policy_optimizer.step()

            value_loss.backward()
            self.value_optimizer.step()

            self.return_list.append(returns[0])

            print(f"episode {episode} return: {returns[0]}")

    def get_returns_list(self):
        return self.return_list
