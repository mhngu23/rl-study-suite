# agents/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from utils.network import QNetwork

    
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        # Define two Q-networks: online and target
        # The first estimates Q-values for action selection
        # The second is used to compute target Q-values for stability
        # The target network is periodically updated with the online network's weights
        # The reason for two networks is to reduce correlations between the target and predicted Q-values, which helps stabilize training.
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.update_count = 0 # For when to update target network

        self.action_dim = action_dim

        # Define optimizer and other parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon_start # Exploration probability
        self.epsilon_end = epsilon_end # Minimum exploration probability
        self.epsilon_decay = epsilon_decay # Decay rate for exploration probability

        self.memory = deque(maxlen=10000)
        self.batch_size = 16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.name = "dqn"

    def act(self, state):
        """This is the main action selection method for the DQN agent.
         It uses an epsilon-greedy strategy to balance exploration and exploitation.
         With probability epsilon, it selects a random action (exploration).
         Otherwise, it selects the action with the highest predicted Q-value (exploitation).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax(dim=1).item()


    def store(self, transition):
        self.memory.append(transition)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )
    
    def update(self):
        if len(self.memory) < self.batch_size: # If the memory is not large enough, skip update
            return None

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Update Q-network by first computing the loss between predicted Q-values and target Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

            # Calculate target Q-values accounting for terminal states
            # The formula for target Q-values is:
            # target = r + γ * max_a' Q_target(s', a') if not done
            # target = r if done
            targets = rewards + self.gamma * next_q_values * (1 - dones) 


        loss = nn.MSELoss()(q_values, targets)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update every few steps
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()