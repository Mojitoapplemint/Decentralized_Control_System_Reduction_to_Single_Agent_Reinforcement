# Decentralized Control System Reduction to Single-Agent Reinforcement Learning

## Overview
One widely studied approach to decentralized partially observable systems is based on Reinforcement Learning (RL). As the RL algorithm also gets generalized as a Multi-agent RL, training the model to converge to the solution becomes more unstable due to inherent limitations of the system, such as Non-stationarity. Hence, a way to reduce a multi-agent system into a single-agent system using ordinary single-agent RL has been proposed. This paper aims to apply two reduction algorithms to a decentralized system where multiple agents collaborate to find the solution and analyze the performance by comparing the two algorithms via implementation and simulation. 

## Research Goal

The project investigates whether a decentralized system with partial observability can be solved effectively by reducing it to a single-agent reinforcement learning problem, and how that compares with independent learning in the same environment.

## Method Summary

The thesis frames the problem using reinforcement learning and multi-agent reinforcement learning concepts, then studies two reduction approaches:

- **Central Q-Learning (CQL):** a joint-action learner that selects actions from the combined action space
- **Independent Q-Learning (IQL):** each agent learns a local policy using only its own observations and rewards

The experiments are based on a cat-and-mouse maze system with partial observability and a shared reward structure.

## Experiments

The thesis reports four experiments:

1. CQL with partial observation()
2. CQL with full observation
3. IQL with the default reward system
4. IQL with an updated reward system

[Add links to notebooks, scripts, or result files for each experiment here.]

## Key Findings

The thesis reports that CQL converged more stably in the small benchmark, while IQL was more affected by partial observability and non-stationarity. The updated reward design improved IQL performance, but convergence was still not guaranteed.


## Requirements

- python==3.12.10
- pandas==2.2.2
- numpy==1.26.4
- matplotlib==3.9.0
- gymnasium==0.29.1

# License
All rights reserved
