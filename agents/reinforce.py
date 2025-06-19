import torch
import torch.optim as optim
from utils.network import PolicyNetwork

# REINFORCE Agent using Monte Carlo policy gradient
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)  # Policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # Optimizer
        self.gamma = gamma  # Discount factor
        self.trajectory = []  # Stores (log_prob, reward) tuples for one episode
        self.name = "reinforce"  # Agent name for identification

    # Select an action using the current policy
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension: (1, state_dim)
        probs = self.policy(state)                     # Get action probabilities
        dist = torch.distributions.Categorical(probs)  # Create categorical distribution
        action = dist.sample()                         # Sample action
        log_prob = dist.log_prob(action)               # Get log-probability for policy gradient
        return action.item(), log_prob

    # Store the log-probability and reward at each step
    def store(self, log_prob, reward):
        self.trajectory.append((log_prob, reward))

    # Perform a policy update using the REINFORCE rule
    def finish_episode(self):
        R = 0
        returns = []
        
        # Compute return-to-go (discounted sum of future rewards)
        for _, reward in reversed(self.trajectory):
            R = reward + self.gamma * R
            returns.insert(0, R)  # insert at beginning to maintain correct order

        # Convert to tensor and normalize returns for stability
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the loss: negative log-probabilities weighted by returns
        loss = 0
        for (log_prob, _), R in zip(self.trajectory, returns):
            # log_prob is ∇θ log πθ(a_t | s_t): how our policy’s parameters θ affect
            # the log‑probability of the taken action a_t in state s_t.
            #
            # R is the return‑to‑go G_t = ∑_{k=t} γ^{k−t} r_k, i.e. the total future reward
            # from this timestep onward.
            #
            # Multiplying them gives ∇θ log πθ(a_t|s_t) * G_t, which is exactly the
            # Monte‑Carlo estimate of the gradient of the expected return w.r.t. θ.
            #
            # Taking the negative here (–log_prob * R) turns our gradient **ascent**
            # on J(θ) into a **descent** on this loss L(θ) = –∑ log_prob * R,
            # so we can use standard optimizer.step() to improve the policy.
            #
            # The intuition is that we want to increase the probability of actions
            # that lead to higher returns (ascent the expected return and decent the gradient), which is achieved by minimizing this loss.
            loss += -log_prob * R  # Policy gradient objective. This is the gradient of the expected return. 

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear trajectory for the next episode
        self.trajectory = []

        return loss.item()  # Return loss for logging/debugging
