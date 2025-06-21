# point_particle_env.py
import  gym
from typing import Optional
import numpy as np
# import gym
from gym import spaces
from gym.utils import seeding
import pygame
from pygame import gfxdraw

class PointParticleEnv(gym.Env):
    """
    Continuous 2D navigation with interior obstacles and boundary walls.
    - State: [x, y, heading] (plus flattened obstacles if obs=True)
    - Action: heading angle in degrees [action_min, action_max]
    - Rewards: potential shaping, heading alignment, smoothness, proximity to obstacles.
    - Episode ends (terminated) on goal; truncated on time-limit.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 size=(50, 50),
                 goal=None,
                 exploring_starts: bool=False,
                 obs: bool=False,
                 action_range=(0, 180),
                 max_episode_steps: int = 500,
                 obstacle_penalty: float = 50.0,
                 wall_penalty: float = 100.0,
                 goal_reward: float = 500.0,
                 render_mode='human'):
        super().__init__()
        # —— seeding for reproducibility
        self.seed()

        # —— env parameters
        self.size = np.array(size, dtype=np.float32)
        self.speed = 5.0
        self.goal = np.array([45.0, 45.0], dtype=np.float32) \
                    if goal is None else np.array(goal, dtype=np.float32)
        self.exploring_starts = exploring_starts
        self.obs = obs
        self.action_min, self.action_max = action_range

        # —— penalties & rewards
        self.obstacle_penalty = obstacle_penalty
        self.wall_penalty     = wall_penalty
        self.goal_reward      = goal_reward
        self.k1, self.k2       = 1.0, 0.2
        self.k3, self.k4       = 0.05, 10.0
        self.d0, self.k5       = 1.0, 0.01
        self.goal_thresh       = 5.0





        # —— time‐limit bookkeeping
        self.max_episode_steps = max_episode_steps
        self.current_step      = 0

        # —— action & observation spaces
        self.action_space = spaces.Box(
            low  = np.array([self.action_min], dtype=np.float32),
            high = np.array([self.action_max], dtype=np.float32),
            dtype=np.float32
        )

        # define some interior rectangular obstacles (x1,y1,x2,y2)
        self.obstacles = [
            (12, 12, 25, 15),
            (25, 25, 27, 37)
        ]

        # observation space: [x, y, heading] + obstacles if obs=True
        if self.obs:
            obs_low  = np.zeros(4 * len(self.obstacles), dtype=np.float32)
            obs_high = np.concatenate((self.size, [360])).astype(np.float32)
            self.observation_space = spaces.Box(
                low  = np.concatenate(([0, 0, 0], obs_low)),
                high = np.concatenate((obs_high, obs_low + self.size[0])),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low  = np.array([0, 0, 0], dtype=np.float32),
                high = np.array([*self.size, 360], dtype=np.float32),
                dtype=np.float32
            )

        # internal
        self.render_mode = render_mode
        self.state  = None
        self.screen = None



    def point_to_rect_distance(self, pt, rect):
        x, y = pt
        x1, y1, x2, y2 = rect
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return np.hypot(dx, dy)

    def compute_reward(self, old_state, new_state, prev_heading):
        pos_old, theta_old = old_state[:2], old_state[2]
        pos_new, theta_new = new_state[:2], new_state[2]

        # potential shaping
        d_old = np.linalg.norm(pos_old - self.goal)
        d_new = np.linalg.norm(pos_new - self.goal)
        r1 = self.k1 * (d_old - d_new)

        # heading alignment
        vec = self.goal - pos_new
        theta_goal = np.degrees(np.arctan2(vec[1], vec[0])) % 360
        # ensure minimal wrap: diff in [-180,180]
        diff = ((theta_new - theta_goal + 180) % 360) - 180
        r2 = self.k2 * np.cos(np.radians(diff))

        # smoothness
        diff2 = ((theta_new - prev_heading + 180) % 360) - 180
        r3 = -self.k3 * abs(diff2)

        # proximity to obstacles
        d_min = min(
            self.point_to_rect_distance(pos_new, obs)
            for obs in self.obstacles
        )
        r4 = -self.k4 * np.exp(-d_min / self.d0)
        reward = r1 + r2 + r3 + r4
        if d_new < self.goal_thresh:
            reward += self.goal_reward
        return reward




    def seed(self, seed: Optional[int]=None):
        """Gym-style seeding."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: Optional[int]=None, options=None):
        """Reset environment; return (obs, info)."""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0

        if self.exploring_starts:
            x       = self.np_random.uniform(0, self.size[0])
            y       = self.np_random.uniform(0, self.size[1])
            heading = self.np_random.uniform(self.action_min, self.action_max)
        else:
            x, y, heading = 10.0, 10.0, 0.0

        if self.obs:
            flat_obs   = np.array(self.obstacles).flatten().astype(np.float32)
            self.state  = np.concatenate(([x, y, heading], flat_obs))
        else:
            self.state  = np.array([x, y, heading], dtype=np.float32)

        return self.state, {}

    def is_collision(self, x: float, y: float) -> bool:
        """Check interior obstacles only."""
        for x1, y1, x2, y2 in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False


    def _line_intersects_rectangle(self, x0, y0, x1, y1, rect):
            xmin, ymin, xmax, ymax = rect
            # Handle zero-movement case
            if x0 == x1 and y0 == y1:
                return (xmin <= x0 <= xmax) and (ymin <= y0 <= ymax)

            dx, dy = x1 - x0, y1 - y0
            p = [-dx, dx, -dy, dy]
            q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
            t0, t1 = 0.0, 1.0

            for i in range(4):
                if abs(p[i]) < 1e-10:  # Parallel to edge
                    if q[i] < 0: return False
                else:
                    t = q[i] / p[i]
                    if p[i] < 0:
                        if t > t1: return False
                        if t > t0: t0 = t
                    else:
                        if t < t0: return False
                        if t < t1: t1 = t

            return t0 <= t1 and t0 <= 1 and t1 >= 0

    def step(self, action):
        """Apply action, return (next_state, reward, terminated, truncated, info)."""
        self.current_step += 1
        result = self._simulate_step(self.state, action)
        new_state, base_reward, terminated, hit_obstacle, hit_wall, info = result

        # reward = base_reward
        prev_heading = self.state[2]
        shaped = self.compute_reward(self.state, new_state, prev_heading)
        reward = shaped
        if hit_obstacle:
            reward -= self.obstacle_penalty
        if hit_wall:
            reward -= self.wall_penalty
        if terminated:
            reward += self.goal_reward

        # truncation on time-limit
        truncated = (self.current_step >= self.max_episode_steps)
        # update state

        self.state = new_state

        return new_state, reward, terminated, truncated,info

    def _simulate_step(self, state, action):
        """Compute next_state, base_reward, terminated (no penalties here)."""
        pos = state[:2]
        current_heading = state[2]
        alpha = 0.3
        # Convert action to relative heading change
        delta_heading = float(action[0])
        new_heading = ((1-alpha)*current_heading + alpha*delta_heading) % 360  # Keep in [0,360)
        dx = self.speed * np.cos(np.radians(new_heading))
        dy = self.speed * np.sin(np.radians(new_heading))
        new_x, new_y = pos[0] + dx, pos[1] + dy

        # Continuous obstacle collision check
        hit_obstacle = any(
            self._line_intersects_rectangle(pos[0], pos[1], new_x, new_y, obs)
            for obs in self.obstacles
        )

        # Wall collision (endpoint only - walls are convex)
        hit_wall = not (0 <= new_x <= self.size[0] and 0 <= new_y <= self.size[1])

        # Roll back position on collision
        if hit_obstacle or hit_wall:

            new_x, new_y = pos[0], pos[1]

        # clip to exact bounds
        new_x = np.clip(new_x, 0, self.size[0])
        new_y = np.clip(new_y, 0, self.size[1])

        if self.obs:
            obs_data  = state[3:]
            new_state = np.concatenate(([new_x, new_y, new_heading], obs_data))
        else:
            new_state = np.array([new_x, new_y, new_heading], dtype=np.float32)

        # base reward = negative distance to goal
        distance   = np.linalg.norm(new_state[:2] - self.goal)
        base_reward = -distance

        # termination when close to goal
        terminated = (distance < 5.0)



        return new_state, base_reward, terminated, hit_obstacle, hit_wall, {}

    def render(self, mode= None):
        """Draw agent, goal, obstacles, and walls."""
        if mode== None:
          mode = self.render_mode

        screen_size = 600
        scale = screen_size / max(self.size)

        if self.screen is None:
            pygame.init()
            if mode == 'human':
                # Create the display surface for human mode
                self.screen = pygame.display.set_mode((screen_size, screen_size))
            else:
                # Create a regular surface for other modes (like rgb_array)
                self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))  # background

        # draw boundary walls (as gray frame)
        wall_thick = 5
        pygame.draw.rect(surf, (100,100,100), (0, 0, screen_size, wall_thick))  # top
        pygame.draw.rect(surf, (100,100,100), (0, 0, wall_thick, screen_size))  # left
        pygame.draw.rect(surf, (100,100,100), (0, screen_size-wall_thick, screen_size, wall_thick))  # bottom
        pygame.draw.rect(surf, (100,100,100), (screen_size-wall_thick, 0, wall_thick, screen_size))  # right

        # draw goal
        gx = int(self.goal[0] * scale)
        gy = screen_size - int(self.goal[1] * scale)
        gfxdraw.filled_circle(surf, gx, gy, int(scale*2), (40,199,172))

        # draw obstacles
        for x1, y1, x2, y2 in self.obstacles:
            rx = int(x1*scale)
            ry = screen_size - int(y2*scale)
            w  = int((x2-x1)*scale)
            h  = int((y2-y1)*scale)
            pygame.draw.rect(surf, (128,128,128), (rx, ry, w, h))

        # draw agent
        ax = int(self.state[0]*scale)
        ay = screen_size - int(self.state[1]*scale)
        gfxdraw.filled_circle(surf, ax, ay, int(scale*1), (228,63,90))
        # heading line
        end_x = ax + int(10*np.cos(np.radians(self.state[2])))
        end_y = ay - int(10*np.sin(np.radians(self.state[2])))
        pygame.draw.line(surf, (255,255,255), (ax,ay), (end_x,end_y), 2)

        canvas = pygame.transform.flip(surf, False, True)
        self.screen.blit(canvas, (0,0))

        if mode=='human':
            pygame.display.flip()
            return None
        elif mode=='rgb_array':
            arr = pygame.surfarray.pixels3d(self.screen)
            return np.transpose(arr, (1,0,2)).astype(np.uint8)
        else:
            raise ValueError(f"Render mode {mode} not supported")

    def close(self):
        """Shutdown pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
