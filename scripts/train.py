import argparse
import numpy as np
from envs.wrappers import make_env
from agents.reinforce import REINFORCEAgent
from agents.reinforce_withbaseline import REINFORCEAgent_withBaseline
from agents.dqn import DQNAgent
import csv

# Callable policy wrapper to plug into rollout logic
def policy(env, agent, obs):
    if agent.name == "reinforce":
        action, log_prob = agent.act(obs)
        agent.store(log_prob, None)  # Store log-prob only; reward added later
        return action
    elif agent.name == "reinforce_with_baseline":
        action, log_prob = agent.act(obs)
        agent.store(obs, log_prob, None)  # Store state and log-prob; reward added later
        return action
    elif agent.name == "dqn":
        return agent.act(obs)

    
def train(agent_name="reinforce", env_id="CartPole-v1", episodes=500, save_path="returns.csv"):
    env = make_env(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Instantiate agent
    if agent_name == "reinforce":
        agent = REINFORCEAgent(state_dim, action_dim)
    elif agent_name == "dqn":
        agent = DQNAgent(state_dim, action_dim)
    elif agent_name == "reinforce_with_baseline":
        agent = REINFORCEAgent_withBaseline(state_dim, action_dim)
    else:
        raise NotImplementedError(f"Agent '{agent_name}' is not implemented yet.")

    max_steps = 200
    returns = []

    for ep in range(episodes):
        obs = env.reset(seed=ep)[0]
        total_reward = 0
        agent.trajectory = []  # Clear trajectory before episode

        for t in range(max_steps):
            action = policy(env, agent, obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # For DQN
            if agent_name == "dqn":
                # Store transition and update
                agent.store((obs, action, reward, next_obs, float(done)))
                # Value-based methods update at each step
                loss = agent.update()
            else:
            # For policy gradient Monte Carlo methods
                if agent.name == "reinforce":
                    agent.trajectory[t] = (agent.trajectory[t][0], reward)  # Update the reward
                elif agent.name == "reinforce_with_baseline":
                    agent.trajectory[t] = (agent.trajectory[t][0], agent.trajectory[t][1], reward)
            
            total_reward += reward

            obs = next_obs
            if done:
                break
        
        returns.append(total_reward)

        if agent.name == "reinforce":
            # Policy gradient, Monte Carlo update at episode end
            loss = agent.finish_episode()
            if ep % 10 == 0:
                print(f"[{agent.name.upper()}] Episode {ep} | Return: {total_reward:.2f} | Loss: {loss:.4f}")
        elif agent.name == "reinforce_with_baseline":
            # Policy gradient, Monte Carlo update at episode end
            policy_loss, value_loss = agent.finish_episode()
            if ep % 10 == 0:
                print(f"[{agent.name.upper()}] Episode {ep} | Return: {total_reward:.2f} | Value Loss: {value_loss:.4f} | Policy Loss: {policy_loss:.4f}")
        elif agent.name == "dqn":
            # Value-based, updates done per step already did before
            if ep % 10 == 0:
                print(f"[DQN] Episode {ep} | Return: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save returns to CSV
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        for ep_num, r in enumerate(returns):
            writer.writerow([ep_num, r])

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument("--agent", type=str, default="reinforce", help="Agent name (e.g., reinforce, ppo, dqn)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment ID")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--save_path", type=str, default="returns.csv", help="Path to save episode returns CSV file")
    args = parser.parse_args()
    train(agent_name=args.agent, env_id=args.env, episodes=args.episodes, save_path=args.save_path)
