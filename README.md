# 🎓 rl-study-suite

A personal collection of foundational reinforcement learning algorithms, implemented from scratch for study and experimentation.  
This project is inspired by [Deepmind](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) and is structured for revisiting core RL concepts over a 3-month period with ~4 hours/week commitment.

---

## 📚 Purpose

This repository serves as a lightweight codebase for revisiting RL concepts, understanding implementation details, and comparing algorithm performance in a clean, modular way.

---

## 🧭 Roadmap

Each week is based on a 4-hour time budget.

| Week | Topic                        | Goal |
|------|------------------------------|------|
| 1    | Setup + Rollouts             | Project structure, rollout collection |
| 2    | REINFORCE                    | Basic policy gradient with return-to-go |
| 3    | Baseline + Advantage         | Add value network for variance reduction |
| 4    | Logging + Comparison         | Visualize and compare learning stability |
| 5    | DQN                          | Value-based method with replay + target net |
| 6    | Double & Dueling DQN         | Improve stability and value estimation |
| 7    | Actor-Critic                 | Implement A2C with online value bootstrapping |
| 8    | Refactor + Summary           | Code cleanup, config support, benchmark PG vs DQN |
| 9    | DDPG                         | Off-policy continuous control with deterministic actor |
| 10   | TD3                          | Double critics and policy smoothing |
| 11   | PPO                          | Clipped surrogate loss, stable policy updates |
| 12   | Final Comparison             | Evaluate all methods and summarize findings |

---

## 🗂️ Directory Structure


rl-study-suite/
- README.md
- requirements.txt
- agents/ # Core RL algorithms
    - reinforce.py
- envs/ # Env wrappers and preprocessors
    - wrappers.py
- utils/ # Shared tools
    - buffer.py
- scripts/ # Training scripts
    - train.py
- experiments/ # Logs, metrics, model checkpoints

---


##  🗓️  Implemented Algorithms

| Category           | Algorithm       | Status       |
|--------------------|------------------|---------------|
| Policy Gradient    | REINFORCE        | ✅ Done |
|                    | Baseline PG      | ✅ Done |
| Value-Based        | DQN              | ✅ Done |
|                    | Double/Dueling   | ⏳ Planned |
| Actor-Critic       | A2C              | ⏳ Planned |
|                    | PPO              | ⏳ Planned |
| Off-Policy Methods | DDPG             | ⏳ Planned |
|                    | TD3              | ⏳ Planned |

---

## 🚀 Getting Started

```bash
# Clone and set up environment
git clone https://github.com/yourusername/rl-study-suite.git
cd rl-study-suite
conda create -n rl-study python=3.10 -y
conda activate rl-study
pip install -r requirements.txt
```

## ▶️ Running Training
```bash
PYTHONPATH=. python scripts/train.py --agent reinforce --env CartPole-v1 --episodes 300 --save_path results/reinforce_cartpole_returns.csv

PYTHONPATH=. python scripts/train.py --agent reinforce_with_baseline --env CartPole-v1 --episodes 300 --save_path results/reinforce_with_baseline_cartpole_returns.csv

PYTHONPATH=. python scripts/train.py --agent dqn --env CartPole-v1 --episodes 300 --save_path results/dqn.csv

```
- agent   : Agent name (e.g., reinforce, ppo, etc.)

- env     : Gym environment ID (e.g., CartPole-v1)

- episodes: Number of training episodes (default: 200)