# visualization.py

import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import DDPGAgent  # replace with your agent class
# in utils.py you should have something like:
# def load_history(path): return np.load(path)
# def plot_rewards(rewards): ...

def load_agent(
    model_path: str,
    env_name: str,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    tau: float = 0.005,
    gamma: float = 0.99
):
    """
    Create env and agent, load weights from model_path.
    Returns (agent, env).
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma
    )
    agent.load_checkpoint(model_path)
    print(f"[visualization] Loaded model from {model_path}")
    return agent, env

def run_episode(agent, env, render: bool = True):
    """
    Runs one episode, returns total reward.
    If render=True, calls env.render().
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, noise=False)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()
    return total_reward

def visualize_episodes(agent, env, episodes: int = 1, render: bool = True):
    """
    Runs `episodes` episodes, prints & returns a list of rewards.
    Closes env (render window) when done.
    """
    rewards = []
    for i in range(episodes):
        r = run_episode(agent, env, render)
        print(f"Episode {i+1} → total reward: {r:.2f}")
        rewards.append(r)
    if render:
        env.close()
    return rewards

def load_history(path: str):
    """
    Load a NumPy .npy or CSV of training metrics.
    """
    if path.endswith((".npy", ".npz")):
        return np.load(path, allow_pickle=True)
    else:
        import pandas as pd
        df = pd.read_csv(path)
        # assume a column named "reward"
        return df["reward"].values

def plot_rewards_curve(rewards: np.ndarray):
    """
    Plot rewards over episodes.
    """
    plt.figure()
    plt.plot(rewards, marker="o")
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
