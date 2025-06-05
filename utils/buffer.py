import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        """
        Initialize the replay buffer.

        Args:
            obs_dim (int): Dimension of the observation/state space.
            act_dim (int): Dimension of the action space.
            size (int): Maximum number of transitions to store in the buffer.
        """
        # Buffer to store observations (states), shape: [size, obs_dim]
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)

        # Buffer to store next observations (next states), same shape as obs_buf
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)

        # Buffer to store actions, shape: [size, act_dim]
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)

        # Buffer to store rewards, shape: [size]
        self.rew_buf = np.zeros(size, dtype=np.float32)

        # Buffer to store done flags (1 if episode ended), shape: [size]
        self.done_buf = np.zeros(size, dtype=np.float32)

        # Pointer to the next index to write a new transition (overwrites oldest when full)
        self.ptr = 0

        # Maximum number of transitions the buffer can store
        self.max_size = size

        # Current number of transitions in the buffer (≤ max_size)
        self.size = 0
    
    def store(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': self.obs_buf[idx],
            'act': self.act_buf[idx],
            'rew': self.rew_buf[idx],
            'done': self.done_buf[idx]
        }