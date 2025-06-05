import gym
from envs.wrappers import make_env

def rollout(env, policy=None, seed=42, max_steps=200):
    obs = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample() if policy is None else policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":
    env = make_env(env_id="CartPole-v1")
    reward = rollout(env)
    print(f"Total reward: {reward}")