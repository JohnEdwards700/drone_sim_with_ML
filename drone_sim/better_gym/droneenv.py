import torch
import numpy as np
from drone_sim.sim import Drone
from drone_sim.sim import PositionTracker, IMU
from drone_sim.sim.parameters import *
from drone_sim.viz import Graphics
from drone_sim.viz import Body
from typing import Optional, Tuple

class SimpleDroneEnv:
    def __init__(self, goal_position=[0, 5, 5], device=Optional[torch.device]) -> None:
        # Initialise the Drone class, attach body and sensors, and make the Graphics object
        # Reset will be manually called in the step function
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.drone = Drone(0, 0, 2, enable_death=False)
        self.body = Body()

        self.body.attach_to(self.drone)

        self.ui = Graphics()
        self.ui.add_actor(self.drone)

        self.goal_position = goal_position
        self.init_position = [0, 0, 2]
        
        # Define action and observation space bounds
        self.action_space_low = torch.zeros(4)
        self.action_space_high = torch.ones(4)
        
        self.observation_space_low = torch.tensor(float('-inf') * np.ones((3, 3)))
        self.observation_space_high = torch.tensor(float('inf') * np.ones((3, 3)))

        # Some rotation values
        self.NULL_ROT = NULL_ROT

    def step(self, action):
        # Ensure action is within bounds
        action = torch.clamp(action, self.action_space_low, self.action_space_high)

        # Convert action to numpy if it's a torch tensor
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        # The action is of the type (w1, w2, w3, w4)
        self.drone.step(action * self.NULL_ROT)

        # Function has to return the observations
        observation = torch.tensor([
            [self.drone.x, self.drone.y, self.drone.z],
            [self.drone.acceleration[0][0], self.drone.acceleration[1][0], self.drone.acceleration[2][0]],
            [self.drone.p, self.drone.q, self.drone.r]
        ], dtype=torch.float32)

        # Reward is calculated as the ratio of 1 - (dist(present, goal)/dist(start, goal))
        dist_to_go = self.dist([self.drone.x, self.drone.y, self.drone.z], [self.init_position[0], self.init_position[1], self.init_position[2]])
        total_dist = self.dist([self.goal_position[0], self.goal_position[1], self.goal_position[2]], [self.init_position[0], self.init_position[1], self.init_position[2]])

        reward = 1 - dist_to_go/total_dist

        # Termination condition
        if abs(self.drone.phi) > np.radians(60.0) or abs(self.drone.theta) > np.radians(60):
            done = True
            self.drone.__reset__()
            reward -= 5

        # Condition 2: If the z altitude goes negative, we reset the simulation
        elif self.drone.z < 0:
            done = True
            self.drone.__reset__()
            reward -= 5

        elif dist_to_go < 0.01:
            done = True
            reward += 10
        
        else:
            done = False

        return observation, torch.tensor(reward, dtype=torch.float32), done, {}

    def reset(self):
        # Function to reset the simulation
        self.drone.__reset__()
        observation = torch.tensor([
            [self.drone.x, self.drone.y, self.drone.z],
            [self.drone.acceleration[0][0], self.drone.acceleration[1][0], self.drone.acceleration[2][0]],
            [self.drone.p, self.drone.q, self.drone.r]
        ], dtype=torch.float32)

        return observation

    def render(self):
        # Function to render the simulation
        self.ui.update()

    # Helper functions
    def dist(self, x1, x2):
        x, y, z = x1
        X, Y, Z = x2

        return np.sqrt((x-X)**2 + (y-Y)**2 + (z-Z)**2)