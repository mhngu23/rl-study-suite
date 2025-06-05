import gym

def make_env(env_id="CartPole-v1", seed=42):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env