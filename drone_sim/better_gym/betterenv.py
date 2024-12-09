import gym
from gym import spaces
from drone_sim.sim import Drone
from drone_sim.sim import PositionTracker, IMU
from drone_sim.sim.parameters import *
from drone_sim.viz import Graphics
from drone_sim.viz import Body
import matplotlib as plt

import numpy as np

class SimpleDroneEnv(gym.Env):
    metadata = {'render.modes': ["human"]}

    def __init__(self, goal_position=[0, 5, 5], max_episode_steps=1000) -> None:
        super(SimpleDroneEnv, self).__init__()

        # Initialise the Drone class, attach body and sensors, and make the Graphics object
        self.drone = Drone(0, 0, 2, enable_death=False)
        self.body = Body()
        self.body.attach_to(self.drone)

        self.ui = Graphics()
        self.ui.add_actor(self.drone)

        # Goal and initial positions
        self.goal_position = np.array(goal_position)
        self.init_position = np.array([0, 0, 2])
        
        # Gym environment parameters
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(9,),  # Full state vector 
            dtype=np.float32
        )
        self.reward_range = (0, 100)
        self.max_episode_steps = max_episode_steps
        self._current_step = 0

    def step(self, action):
        # Validate action
        #assert self.action_space.contains(action), f"{action} does not exist in the action space"
        if not np.all((action>=0) & (action <=1)):
            print(f'action not between range: {action}')
            print(f'highest value={np.max(action)} lowest value={np.min(action)}')
            raise ValueError('action must be between 0 and 1')

        # Scale action to motor velocities
        motor_velocities = action * NULL_ROT
        self.drone.step(motor_velocities)

        # Compute observation (full state vector)
        observation = np.array([
            self.drone.x, self.drone.y, self.drone.z,  # Position
            self.drone.vx, self.drone.vy, self.drone.vz,  # Velocity
            self.drone.phi, self.drone.theta, self.drone.psi  # Orientation
        ])

        # Distance calculations
        current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        dist_to_go = np.linalg.norm(current_pos - self.goal_position)
        total_dist = np.linalg.norm(self.init_position - self.goal_position)

        # Reward calculation
        reward = 1 - (dist_to_go / total_dist)

        # Termination conditions
        self._current_step += 1
        done = False
        
        # Angle limits
        if (abs(self.drone.phi) > np.radians(60.0) or 
            abs(self.drone.theta) > np.radians(60.0)):
            done = True
            reward -= 5

        # Altitude limit
        elif self.drone.z < 0:
            done = True
            reward -= 5

        # Goal reached
        elif dist_to_go < 0.01:
            done = True
            reward += 10

        # Maximum steps reached
        elif self._current_step >= self.max_episode_steps:
            done = True

        return observation, reward, done, {}

    def reset(self):
        # Reset drone and tracking
        self.drone.__reset__()
        self._current_step = 0

        # Return initial observation
        return np.array([
            self.drone.x, self.drone.y, self.drone.z,  # Position
            self.drone.vx, self.drone.vy, self.drone.vz,  # Velocity
            self.drone.phi, self.drone.theta, self.drone.psi  # Orientation
        ])

    def render(self):
        # Render the simulation
        self.ui.update()

    def close(self):
        # Close the visualization
        plt.close()