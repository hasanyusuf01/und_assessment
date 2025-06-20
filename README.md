
![image](https://github.com/user-attachments/assets/f248cffb-44e3-4ac3-a0ee-a76fd4cd34c9)


The Training System serves as the central orchestration layer for the DDPG reinforcement learning pipeline. It manages the complete training lifecycle from initialization through episode execution, experience collection, network updates, and performance evaluation. The system is implemented primarily through the Trainer class which coordinates interactions between the DDPG agent, environment, replay buffer, and logging subsystems.

![image](https://github.com/user-attachments/assets/7ff95c8e-2f63-4310-b5fb-c91d4d08f0f5)


# Point Particle Environment
## Purpose and Scope
The Point Particle Environment provides a continuous 2D navigation simulation for reinforcement learning training. This environment implements a point particle agent that must navigate from a starting position to a goal while avoiding rectangular obstacles and boundary walls. The environment features sophisticated reward shaping, collision detection, and visual rendering capabilities.

This document covers the PointParticleEnv class implementation, including state/action spaces, physics simulation, reward computation, and rendering. For information about how this environment integrates with the training system, see Training System. For details about the DDPG agent that interacts with this environment, see DDPG Agent.

## Environment Configuration

The environment supports extensive configuration through initialization parameters:

| Parameter             | Type    | Default        | Description                           |
|-----------------------|---------|----------------|---------------------------------------|
| `size`                | tuple   | (50, 50)       | Environment dimensions                |
| `goal`                | array   | [45.0, 45.0]   | Goal position coordinates             |
| `exploring_starts`    | bool    | False          | Random vs fixed start position        |
| `obs`                 | bool    | False          | Include obstacles in state            |
| `action_range`        | tuple   | (0, 180)       | Action space bounds                   |
| `max_episode_steps`   | int     | 500            | Episode time limit                    |
| `obstacle_penalty`    | float   | 50.0           | Collision penalty                     |
| `wall_penalty`        | float   | 100.0          | Wall collision penalty                |
| `goal_reward`         | float   | 500.0          | Goal achievement bonus                |


![image](https://github.com/user-attachments/assets/ab49e8c9-a897-4fe5-a20b-45e86dd8126c)

## Obstacle Configuration

Predefined rectangular obstacles: `[(12, 12, 25, 15), (25, 25, 27, 37)]`  
Format: `(x1, y1, x2, y2)` representing rectangle corners  
Can be modified by editing the `obstacles` list in initialization.

## Environment Architecture
The PointParticleEnv class implements the OpenAI Gym interface and provides a complete 2D navigation simulation with the following key components:

| Component            | Purpose                                          | Key Methods                          |
|----------------------|--------------------------------------------------|--------------------------------------|
| State Management      | Track agent position heading                 | `reset()`, `step()`                  |
| Physics Simulation    | Move agent and detect collisions                 | `_simulate_step()`, collision detection |
| Reward System         | Multi-component reward calculation               | `compute_reward()`, distance functions |
| Rendering             | Visual display with Pygame                       | `render()`, `close()`                |

## State and Action Spaces
### State Space
The environment state consists of the agent's position and orientation:

![image](https://github.com/user-attachments/assets/9613744a-bb76-4fb5-848d-816fbaa57341)

The state space is configured during initialization based on the obs parameter:

- Basic state (obs=False): 3D vector [x, y, heading]
- Extended state (obs=True): Includes flattened obstacle coordinates
Observation Space Bounds:

- Position: [0, 0] to [size_x, size_y] (default: [50, 50])
- Heading: 0 to 360 degrees
- Obstacles: Flattened coordinates when obs=True

## Action Space
The action space is continuous and one-dimensional, representing the desired heading angle:
![image](https://github.com/user-attachments/assets/825e4326-1fd7-468a-b596-437be56dca5a)
The action is smoothed using exponential averaging to prevent abrupt heading changes, promoting smoother trajectories.

## Reward System
The environment implements a sophisticated multi-component reward system designed to guide the agent toward efficient goal-reaching behavior:

### Components

| Component            | Formula                          | Purpose                   | Default Weight |
|---------------------|----------------------------------|---------------------------|-----------------|
| Potential Shaping   | `k1 * (d_old - d_new)`          | Progress toward goal      | `k1 = 1.0`      |
| Heading Alignment    | `k2 * cos(heading_error)`       | Face toward goal          | `k2 = 0.2`      |
| Smoothness          | `-k3 * abs(heading_change)`     | Smooth movement           | `k3 = 0.05`     |
| Obstacle Proximity   | `-k4 * exp(-d_min/d0)`         | Avoid obstacles           | `k4 = 10.0`     |

### Special Rewards

- **Goal reached:** `+goal_reward` (default: 500.0)
- **Obstacle collision:** `-obstacle_penalty` (default: 50.0)
- **Wall collision:** `-wall_penalty` (default: 100.0)

![image](https://github.com/user-attachments/assets/d8e8af62-12ac-4b37-8b17-a0c40f26c95b)




## Results Analysis

The Episode Logging System captures high-level performance metrics for each complete training episode in the DDPG reinforcement learning system. This system records episode-level aggregated data including total rewards, completion status, and termination reasons to enable training progress analysis and performance evaluation.
#### Episode-Level Logging (csv1.csv)

- Episode rewards and completion status
- Goal achievement tracking
- Episode length statistics

#### Step-Level Logging (csv2.csv)

- Detailed state observations per step
- Immediate reward values
- Complete trajectory information

The episode logs reveal distinct phases in the learning progression:

- **Exploration**: Initial phase where the model is trying to understand the environment, leading to negative rewards and a low success rate.
- **Breakthrough**: A transitional phase where the model starts to show improvement, with variable rewards and a moderate success rate.
- **Convergence**: Final phase where the model has learned effectively, achieving high rewards and a 100% success rate.


| Phase        | Episode Range | Avg Reward | Success Rate | Avg Steps |
|--------------|---------------|------------|--------------|-----------|
| Exploration  | 1-83          | -25,000    | 0%           | 300       |
| Breakthrough | 84-86         | Variable   | 67%          | 150       |
| Convergence  | 87+           | +300       | 100%         | 50        |

![image](https://github.com/user-attachments/assets/5df7d0d6-d98d-4e8e-b671-1ac4c37797d3)

---

# Configuration System
The configuration system consists of JSON files located in the DDPG-Pytorch/configs/ directory. Each configuration file defines a complete set of hyperparameters for training a DDPG agent on a specific environment.
### JSON Configuration Schema

Each configuration file follows a consistent schema with the following parameters:

| Parameter          | Type    | Purpose                                           | Used By                     |
|--------------------|---------|---------------------------------------------------|-----------------------------|
| `env_name`         | String  | Environment identifier                            | Trainer initialization      |
| `seed`             | Integer | Random seed for reproducibility                   | Environment and agent seeding|
| `discount`         | Float   | Reward discount factor (γ)                        | DDPGAgent training          |
| `tau`              | Float   | Soft update parameter for target networks         | DDPGAgent network updates    |
| `time_steps`       | Integer | Total training steps                              | Training loop termination    |
| `start_time_step`  | Integer | Steps before training begins                      | Exploration phase control    |
| `expl_noise`       | Float   | Exploration noise standard deviation              | Action selection noise       |
| `batch_size`       | Integer | Training batch size                               | ReplayBuffer sampling        |
| `evaluate_frequency`| Integer | Evaluation interval in steps                      | Policy evaluation scheduling  |

---

### Development Dependencies

The system requires several Python packages across different functional areas:

| Category                | Dependencies              | Usage                          |
|------------------------|---------------------------|--------------------------------|
| Deep Learning          | `torch`, `torchvision`    | Neural network implementation   |
| Scientific Computing    | `numpy`, `scipy`         | Mathematical operations         |
| Visualization          | `matplotlib`, `pygame`    | Plotting and environment rendering |
 Data Management        | `pandas`                  | Log file processing            |
| Experiment Tracking    | `wandb`                   | Remote experiment monitoring    |
| Environment            | `gym`                     | Reinforcement learning interface |



## Training Loop Data Flow

![image](https://github.com/user-attachments/assets/802fb04c-2879-41bd-8f93-b7043779e58d)


