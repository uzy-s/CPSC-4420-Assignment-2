
# Taxi Q Learning

CPSC 4420 Assignment 2 using Gymnasium Taxi-v4.

The program trains a tabular Q-learning agent and compares learning rates,
discount factors, and epsilon decay settings. It also runs the action noise
experiment for extra credit.

## Setup

Use Python 3.12 or newer. Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the experiments

```powershell
.\.venv\Scripts\python.exe q_learning.py
```

This runs all experiments, including extra credit, and saves the output in
`results/`.

The default settings are 10,000 training episodes per seed, seeds 0, 1, and 2,
alpha = 0.5, and gamma = 0.99. Epsilon starts at 1.0 and is multiplied by
0.9995 each episode until it reaches 0.01.

The program compares:

- Alpha: 0.1, 0.5, 0.9
- Gamma: 0.50, 0.90, 0.99
- Epsilon decay: 0.999, 0.9995, 0.9999
- Action noise: 0, 0.10, 0.25

Each policy is evaluated for 300 episodes per training seed without exploration.
The best configuration is selected using separate validation episodes.

## Results

The `results/` folder contains:

- Six plots showing training and experiment results
- `summary.json` with settings and final metrics
- `validation.csv` and `evaluation.csv` with numerical results
- `training_rewards.npz` with training rewards
- `best_q.npy` with the selected agent's Q-table
- `trained_agent.mp4` showing one episode from that agent

The completed default run achieved 100% final evaluation success for both the
baseline and selected configuration. Mean rewards were 7.9733 and 7.9800,
respectively. Action noise reduced rewards even though deliveries still finished.

To see all command-line options:

```powershell
.\.venv\Scripts\python.exe q_learning.py --help
```
