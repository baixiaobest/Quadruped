import torch

class ActorCriticOneStep:
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, num_episodes=1000, max_steps=100, gamma=0.99):
        self.env = env
        self.policy = policy
        self.value_func = value_func
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.return_list = []
    
    def train(self):
        self.policy.train()
        self.return_list = []

        for episode in range(self.num_episodes):
            self.env.reset()
            state = torch.tensor(self.env.get_state(), dtype=torch.float32)
            discount = 1
            rewards = []

            for step in range(self.max_steps):
                actions_prob = self.policy.forward(state)
                dist = torch.distributions.Categorical(actions_prob)

                action_idx = dist.sample()
                
                action_log_prob = dist.log_prob(action_idx)

                next_state, reward, done = self.env.step(self.policy.get_action(action_idx).item())
                next_state = torch.tensor(next_state, dtype=torch.float32)

                rewards.append(reward)

                td_target = reward + self.gamma * self.value_func.forward(next_state).detach()
                td_error = td_target - self.value_func.forward(state)

                 # Update step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                policy_loss = -action_log_prob * td_error.detach() * discount
                value_loss = td_error.pow(2)

                policy_loss.backward()
                self.policy_optimizer.step()
                
                value_loss.backward()
                self.value_optimizer.step()

                if done:
                    break

                state = next_state
                discount *= self.gamma
                
            if step == self.max_steps:
                print(f"max step reached at episode {episode}")

            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
            self.return_list.append(G)

            print(f"episode {episode} return: {G}")

    def get_returns_list(self):
        return self.return_list


class ActorCriticEligibilityTrace:
    def __init__(self):
        pass