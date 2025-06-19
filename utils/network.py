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