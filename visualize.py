import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_episode_path(csv_path='csv2.csv', episode_number=1):
    """
    Reads csv2 (columns: episode, step, observation, reward),
    filters for one episode, parses the JSON observation,
    and plots the (x,y) path in the 50×50 grid with y-axis inverted.
    """
    # 1. Load CSV
    df = pd.read_csv(csv_path)

    # 2. Filter for the requested episode
    df_ep = df[df['episode'] == episode_number]
    if df_ep.empty:
        print(f"No data for episode {episode_number}")
        return

    # 3. Parse observation JSON into lists and extract x,y
    obs_lists = df_ep['observation'].apply(json.loads)
    xs = obs_lists.apply(lambda v: v[0])
    ys = obs_lists.apply(lambda v: v[1])

    # 4. Plot trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, '-o', markersize=4, linewidth=1, label=f'Ep {episode_number} path')
    ax.scatter(xs.iloc[0], ys.iloc[0], color='green', s=80, label='start')
    ax.scatter(xs.iloc[-1], ys.iloc[-1], color='red', s=80, label='end')

    # 5. Draw obstacles
    obstacles = [
        (12, 12, 25, 15),
        (25, 25, 27, 37)
    ]
    for x1, y1, x2, y2 in obstacles:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='black', facecolor='gray', alpha=0.6)
        ax.add_patch(rect)

    ax.set_ylim(50, 0)
    ax.set_xlim(0, 50)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_title(f'Episode {episode_number} Trajectory')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', 'box')
    # 8. Show
    plt.show()


if __name__ == '__main__':
    plot_episode_path('/content/drive/MyDrive/Projects/rl/DDPG-Pytorch/14_18_jun19/csv2.csv', episode_number=700)
