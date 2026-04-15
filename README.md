# Decentralized Control System Reduction to Single-Agent Reinforcement Learning

[Add a 1–2 sentence project summary here.]

## Overview

This repository contains the code and supplementary materials for an honors thesis on reducing a decentralized, partially observable control problem to a single-agent reinforcement learning formulation. The thesis compares Central Q-Learning (CQL) and Independent Q-Learning (IQL) on a cat-and-mouse maze benchmark under different observability and reward settings.

## Research Goal

The project investigates whether a decentralized system with partial observability can be solved effectively by reducing it to a single-agent reinforcement learning problem, and how that compares with independent learning in the same environment.

## Repository Contents

This repository currently contains:

- `Central Q Learning/` — materials for the central-learning experiments
- `Independent Q Learning/` — materials for the independent-learning experiments
- `Decentralized_Control_System_Reduction_to_Single_Agent_Reinforcement.pdf` — the full thesis manuscript

[Add any additional folders or files here if the repository is expanded later.]

## Method Summary

The thesis frames the problem using reinforcement learning and multi-agent reinforcement learning concepts, then studies two reduction approaches:

- **Central Q-Learning (CQL):** a joint-action learner that selects actions from the combined action space
- **Independent Q-Learning (IQL):** each agent learns a local policy using only its own observations and rewards

The experiments are based on a cat-and-mouse maze system with partial observability and a shared reward structure.

## Experiments

The thesis reports four experiments:

1. CQL with partial observation
2. CQL with full observation
3. IQL with the default reward system
4. IQL with an updated reward system

[Add links to notebooks, scripts, or result files for each experiment here.]

## Key Findings

The thesis reports that CQL converged more stably in the small benchmark, while IQL was more affected by partial observability and non-stationarity. The updated reward design improved IQL performance, but convergence was still not guaranteed.

## How to Reproduce

### Requirements

- Python [version]
- Jupyter Notebook [version]
- gymnasium
- [Add any other dependencies here]

### Setup

```bash
pip install -r requirements.txt
```

[Add the correct environment setup steps here if a requirements file does not exist.]

### Run

[Add the exact notebook or script name here.]

```bash
python [main_script.py]
```

or

```bash
jupyter notebook
```

## Repository Structure

```text
.
├── Central Q Learning/
├── Independent Q Learning/
└── Decentralized_Control_System_Reduction_to_Single_Agent_Reinforcement.pdf
```

[Update this tree if there are more files or subfolders.]


## License
All rights reserved
