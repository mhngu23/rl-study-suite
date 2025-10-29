import torch.nn as nn

# Policy Network maps states to action probabilities
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # First hidden layer
            nn.ReLU(),                         # Activation
            nn.Linear(hidden_dim, action_dim), # Output layer
            nn.Softmax(dim=-1)                 # Convert to action probabilities
        )

    def forward(self, x):
        return self.net(x)

# Value Network estimates the value of a given state
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

# Q-Network estimates the Q-values for each action in a given state
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)