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


class ActorCriticEligibilityTrace(ActorCriticOneStep):
    def __init__(self, env, policy, policy_optimizer, value_func, value_optimizer, 
                 num_episodes=1000, max_steps=100, gamma=0.99, lambda_policy=0.9, lambda_value=0.9):
        self.env = env
        self.policy = policy
        self.value_func = value_func
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.lambda_policy = lambda_policy
        self.lambda_value = lambda_value

        self.return_list = []

    def train(self):
        self.policy.train()
        self.return_list = []
        torch.autograd.set_detect_anomaly(True)

        for episode in range(self.num_episodes):
            self.env.reset()
            state = torch.tensor(self.env.get_state(), dtype=torch.float32)
            discount = 1
            rewards = []

            policy_trace = None
            value_trace = None

            for step in range(self.max_steps):
                actions_prob = self.policy.forward(state.detach())
                dist = torch.distributions.Categorical(actions_prob)

                action_idx = dist.sample()
                
                action_log_prob = dist.log_prob(action_idx)

                next_state, reward, done = self.env.step(self.policy.get_action(action_idx).item())
                next_state = torch.tensor(next_state, dtype=torch.float32)

                rewards.append(reward)

                # Policy forward pass and gradient computation
                td_error = reward + self.gamma * self.value_func.forward(next_state).detach() - self.value_func.forward(state).detach() * (1-done)

                action_log_grad = torch.autograd.grad(action_log_prob, self.policy.parameters())
                threshold1 = 20
                if _grad_norm(action_log_grad) > threshold1:
                    action_log_grad = _grad_const_mul(_normalize_grad(action_log_grad), threshold1)

                if policy_trace is None:
                    policy_trace = _grad_const_mul(action_log_grad, discount)
                else:
                    policy_trace = _grad_add(_grad_const_mul(policy_trace, self.gamma * self.lambda_value), _grad_const_mul(action_log_grad, discount))
                
                policy_grad = _grad_const_mul(policy_trace, -td_error)

                self.policy_optimizer.zero_grad()
                _set_grad(self.policy.parameters(), policy_grad)
                self.policy_optimizer.step()

                # print(f"action_log: {norm(action_log_grad):.3f}")
                # print(f"trace: {norm(policy_trace):.3f}")
                # print(f"grad: {norm(policy_grad):.3f}")
                # print(f"td error: {td_error.item():.3f}\n")

                # if norm(policy_trace).item() > 10:
                #     print("---")

                # # Value forward pass and gradient computation
                curr_value_grad = torch.autograd.grad(self.value_func.forward(state), self.value_func.parameters())
                threshold2=1
                if _grad_norm(curr_value_grad) > threshold2:
                    curr_value_grad = _grad_const_mul(_normalize_grad(curr_value_grad), threshold2) 

                if value_trace is None:
                    value_trace = curr_value_grad
                else:
                    value_trace = _grad_add(_grad_const_mul(value_trace, self.gamma * self.lambda_value), curr_value_grad)
                
                value_grad = _grad_const_mul(value_trace, -td_error)

                self.policy_optimizer.zero_grad()
                _set_grad(self.value_func.parameters(), value_grad)
                self.value_optimizer.step()

                # print(f"value grad: {norm(curr_value_grad):.3f}")
                # print(f"trace: {norm(value_trace):.3f}")
                # print(f"grad: {norm(value_grad):.3f}")
                # print(f"td error: {td_error.item():.3f}\n")

                # if norm(value_trace).item() > 10:
                #     print("---")

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
    
def _normalize_grad(grad):
    n = _grad_norm(grad)
    return tuple(g / (n + 1e-5) for g in grad)

def _grad_norm(grad):
    flattened_tensors = [t.flatten() for t in grad]
    combined = torch.cat(flattened_tensors) 
    return torch.norm(combined)

def _set_grad(parameters, grad):
    # Assign computed gradients to policy parameters
    for param, grad in zip(parameters, grad):
        param.grad = grad  # Manually set gradients
    
def _grad_const_mul(grad, scalar):
    return tuple(scalar * t for t in grad)

def _grad_add(grad1, grad2):
    return tuple(a + b for a, b in zip(grad1, grad2))
