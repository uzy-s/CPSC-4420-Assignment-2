"""Tabular Q-learning and controlled experiments for Gymnasium Taxi-v4."""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    episodes: int = 10000

    def __post_init__(self):
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon <= 1:
            raise ValueError("require 0 <= epsilon_min <= epsilon <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not isinstance(self.episodes, int) or self.episodes < 1:
            raise ValueError("episodes must be a positive integer")

    def exploration(self, episode):
        return max(self.epsilon_min, self.epsilon * self.epsilon_decay**episode)


class ActionNoise(gym.ActionWrapper):
    """Replace an action with a different action with probability p."""

    def __init__(self, env, probability):
        super().__init__(env)
        if not 0 <= probability <= 1:
            raise ValueError("noise probability must be in [0, 1]")
        self.probability = probability

    def action(self, action):
        if self.probability and self.np_random.random() < self.probability:
            replacement = int(self.np_random.integers(self.action_space.n - 1))
            return replacement + (replacement >= action)
        return action


def make_env(noise=0.0, render_mode=None):
    return ActionNoise(gym.make("Taxi-v4", render_mode=render_mode), noise)


def select_action(values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(values)))
    return int(np.argmax(values))


def update_q(q, state, action, reward, next_state, terminated, alpha, gamma):
    # A time-limit truncation still bootstraps; a true terminal state does not.
    target = reward if terminated else reward + gamma * np.max(q[next_state])
    q[state, action] += alpha * (target - q[state, action])


def train(config, seed, noise=0.0):
    env = make_env(noise)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = np.zeros(config.episodes)
    try:
        for episode in range(config.episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            epsilon = config.exploration(episode)
            while True:
                action = select_action(q[state], epsilon, rng)
                next_state, reward, terminated, truncated, _ = env.step(action)
                update_q(q, state, action, reward, next_state, terminated,
                         config.alpha, config.gamma)
                rewards[episode] += reward
                state = next_state
                if terminated or truncated:
                    break
    finally:
        env.close()
    return q, rewards


def evaluate(q, episodes=300, seed=100000, noise=0.0):
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive")
    env = make_env(noise)
    records = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total, steps = 0.0, 0
            while True:
                state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
                total += reward
                steps += 1
                if terminated or truncated:
                    records.append((total, steps, int(terminated)))
                    break
    finally:
        env.close()
    return np.asarray(records)


def metrics(records):
    successes = records[:, 2].astype(bool)
    return {
        "episodes": len(records),
        "mean_reward": float(records[:, 0].mean()),
        "reward_std": float(records[:, 0].std()),
        "success_rate": float(successes.mean()),
        "mean_steps": float(records[:, 1].mean()),
        "mean_success_steps": float(records[successes, 1].mean()) if successes.any() else None,
    }


def moving_average(values, window):
    if window < 1 or window > len(values):
        raise ValueError("window must be between 1 and the number of rewards")
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning(curves, path, title, window=100):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, runs in curves.items():
        width = min(window, len(runs[0]))
        smooth = np.asarray([moving_average(run, width) for run in runs])
        x = np.arange(width, width + smooth.shape[1])
        mean, std = smooth.mean(axis=0), smooth.std(axis=0)
        line, = ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Training episode", ylabel="Reward (100-episode moving mean)", title=title)
    if width != 100:
        ax.set_ylabel(f"Reward ({width}-episode moving mean)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def record_video(q, path, seed=200000):
    import imageio.v2 as imageio

    env = make_env(render_mode="rgb_array")
    try:
        state, _ = env.reset(seed=seed)
        with imageio.get_writer(str(path), fps=6, macro_block_size=1) as writer:
            writer.append_data(env.render())
            while True:
                state, _, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
                writer.append_data(env.render())
                if terminated or truncated:
                    break
    finally:
        env.close()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(baseline, seeds, eval_episodes, output, include_noise=True, video=True):
    if eval_episodes < 30:
        raise ValueError("use at least 30 evaluation episodes")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache, history, rows, evaluations = {}, {}, [], []

    def get_runs(config, noise=0.0):
        key = (config, noise)
        if key not in cache:
            runs = []
            for seed in seeds:
                print(f"Training: alpha={config.alpha}, gamma={config.gamma}, "
                      f"decay={config.epsilon_decay}, noise={noise}, seed={seed}", flush=True)
                q, rewards = train(config, seed, noise)
                validation = metrics(evaluate(q, eval_episodes, seed=100000, noise=noise))
                run_id = f"run_{len(rows):02d}"
                history[run_id] = rewards
                rows.append({"run": run_id, "seed": seed, "noise": noise,
                             **asdict(config),
                             **{f"validation_{k}": v for k, v in validation.items()}})
                runs.append((q, rewards, validation))
            cache[key] = runs
        return cache[key]

    def assess(label, runs, noise=0.0):
        all_records = []
        for seed, (q, _, _) in zip(seeds, runs):
            records = evaluate(q, eval_episodes, seed=200000, noise=noise)
            all_records.append(records)
            for episode, (reward, steps, success) in enumerate(records):
                evaluations.append({"policy": label, "training_seed": seed,
                                    "noise": noise, "episode": episode + 1,
                                    "reward": reward, "steps": int(steps),
                                    "success": int(success)})
        return metrics(np.concatenate(all_records))

    baseline_runs = get_runs(baseline)
    plot_learning({"Baseline": [r[1] for r in baseline_runs]},
                  output / "baseline.png", "Baseline training")
    for parameter, values in [("alpha", (0.1, 0.5, 0.9)),
                              ("gamma", (0.5, 0.9, 0.99)),
                              ("epsilon_decay", (0.999, 0.9995, 0.9999))]:
        curves = {}
        for value in values:
            config = replace(baseline, **{parameter: value})
            curves[f"{parameter}={value}"] = [r[1] for r in get_runs(config)]
        plot_learning(curves, output / f"{parameter}.png", f"Effect of {parameter}")

    # Select configurations on validation episodes, keeping final evaluation separate.
    best_config, _ = max(cache, key=lambda key: (
        np.mean([r[2]["mean_reward"] for r in cache[key]]),
        np.mean([r[2]["success_rate"] for r in cache[key]])))
    best_runs = get_runs(best_config)
    summary = {
        "environment": "Taxi-v4", "training_seeds": seeds,
        "validation_seed_start": 100000, "test_seed_start": 200000,
        "evaluation_episodes_per_seed": eval_episodes,
        "baseline_config": asdict(baseline), "best_config": asdict(best_config),
        "selection": "Highest mean validation reward across training seeds; success rate breaks ties; "
                     "remaining ties keep the first configuration. No joint grid search.",
        "baseline": assess("baseline", baseline_runs),
        "best": assess("best", best_runs),
        "saved_policy_training_seed": seeds[0],
        "saved_policy": metrics(evaluate(best_runs[0][0], eval_episodes, seed=200000)),
    }
    np.save(output / "best_q.npy", best_runs[0][0])

    if include_noise:
        noise_results, curves = [], {}
        for noise in (0.0, 0.1, 0.25):
            runs = get_runs(best_config, noise)
            curves[f"p={noise}"] = [r[1] for r in runs]
            noise_results.append({"noise": noise,
                                  "retrained": assess("noise_retrained", runs, noise),
                                  "original_policy": assess("original_under_noise", best_runs, noise)})
        summary["noise"] = noise_results
        plot_learning(curves, output / "noise_learning.png", "Training with action noise")
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label in [("retrained", "Trained at each noise level"),
                           ("original_policy", "Original p=0 policies")]:
            ax.plot([r["noise"] for r in noise_results],
                    [r[key]["mean_reward"] for r in noise_results], "o-", label=label)
        ax.set(xlabel="Action replacement probability", ylabel="Mean evaluation reward",
               title="Policy robustness (exploration disabled)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "noise_evaluation.png", dpi=160)
        plt.close(fig)

    np.savez_compressed(output / "training_rewards.npz", **history)
    write_csv(output / "validation.csv", rows)
    write_csv(output / "evaluation.csv", evaluations)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if video:
        record_video(best_runs[0][0], output / "trained_agent.mp4")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    try:
        config = Config(args.alpha, args.gamma, args.epsilon, args.epsilon_min,
                        args.epsilon_decay, args.episodes)
        run_experiments(config, args.seeds, args.eval_episodes, args.output,
                        not args.skip_noise, not args.skip_video)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
