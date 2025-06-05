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
    - dqn.py
    - ddpg.py
    - td3.py
    - ppo.py
- envs/ # Env wrappers and preprocessors
    - wrappers.py
- utils/ # Shared tools
    - logger.py
    - buffer.py
    - networks.py
    - config.py
- scripts/ # Training scripts
    - train_pg.py
    - train_dqn.py
    - train_ddpg.py
    - train_ppo.py
- experiments/ # Logs, metrics, model checkpoints

---

##  🗓️  Implemented Algorithms

| Category           | Algorithm       | Status       |
|--------------------|------------------|---------------|
| Policy Gradient    | REINFORCE        | ⏳ Planned |
|                    | Baseline PG      | ⏳ Planned |
| Value-Based        | DQN              | ⏳ Planned |
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
