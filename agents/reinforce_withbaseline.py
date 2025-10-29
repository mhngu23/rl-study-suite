import torch
import torch.optim as optim
from utils.network import PolicyNetwork, ValueNetwork

# REINFORCE Agent using Monte Carlo policy gradient
class REINFORCEAgent_withBaseline:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)  # Policy network
        self.value = ValueNetwork(state_dim)                 # Value network
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # Optimizer
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)  # Value network optimizer
        self.gamma = gamma  # Discount factor
        self.trajectory = []  # Stores (log_prob, reward) tuples for one episode
        self.name = "reinforce_with_baseline"  # Agent name for identification

    # Select an action using the current policy
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension: (1, state_dim)
        probs = self.policy(state)                     # Get action probabilities from Policy network
        dist = torch.distributions.Categorical(probs)  # Create categorical distribution
        action = dist.sample()                         # Sample action
        log_prob = dist.log_prob(action)               # Get log-probability for policy gradient
        return action.item(), log_prob

    # Store the log-probability and reward at each step
    def store(self, state, log_prob, reward):
        self.trajectory.append((state, log_prob, reward))

    def finish_episode(self):
        R = 0
        returns = []
        for (_, _, reward) in reversed(self.trajectory):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss, value_loss = 0, 0
        for (state, log_prob, _), R in zip(self.trajectory, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.value(state_tensor).squeeze(0)
            advantage = R - value.detach()  # detach to prevent backprop into critic. 
            # Here we use the baseline (value) to reduce variance.
            # It tells us how much better (or worse) the actual return was compared to the expected return.
            # positive avantage -> increase action probability; negative advantage -> decrease action probability
            # this is because if the action led to a better outcome than expected, we want to reinforce it, and vice versa.
            policy_loss += -log_prob * advantage # # REINFORCE update with baseline use advantage instead of actual return
            value_loss += (value - R) ** 2

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.trajectory = []
        return policy_loss.item(), value_loss.item()
