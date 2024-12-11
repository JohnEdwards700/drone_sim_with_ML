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

# Parameters for dron env:
#initial position is a list of (x, y, z) for possible start locations
#goal positions is a list of (x, y , z) for possible goal locations
#obstacle locations is a list of (x, y, z, radius)
#max episode steps is maximum steps it can take per training session


    def __init__(self,
                 init_positions:list[(float, float, float)] = None,
                 goal_positions:list[(float, float, float)] = None,
                 obstacle_positions:list[(float, float, float)] = None,
                 max_episode_steps=1000
                 ) -> None:
    # def __init__(self,
    #              init_positions = None,
    #              goal_positions= None,
    #              obstacle_positions = None,
    #              max_episode_steps=1000
    #              ) -> None:
        super(SimpleDroneEnv, self).__init__()
        
        self.init_positions = init_positions or [
            (0.0, 0.0, 2.0),
            (1.0, 1.0, 2.0),
            (-1.0, -1.0, 2.0)
        ]
        
        self.goal_positions = goal_positions or [
            (0.0, 5.0, 5.0),
            (5.0, 5.0, 5.0),
            (-5.0, 5.0, 5.0)
        ]
        
        self.obstacle_positions = obstacle_positions or [
            (2.5, 2.5, 3.0, 0.5),   # (x, y, z, radius)
            (-2.5, 2.5, 3.0, 0.5),
            (0.0, 4.0, 4.0, 0.5)
        ]
        
        
        
        
        self.init_positions = [tuple(map(float, pos)) for pos in self.init_positions]
        self.goal_positions = [tuple(map(float, pos)) for pos in self.goal_positions]
        self.obstacle_positions = [tuple(map(float, obs)) for obs in self.obstacle_positions]
        # Initialise the Drone class, attach body and sensors, and make the Graphics object
        self.drone = Drone(0.0, 0.0, 2.0, enable_death=False)
        self.body = Body()
        self.body.attach_to(self.drone)

        self.ui = Graphics()
        self.ui.add_actor(self.drone)

        # Goal and initial positions
        # self.goal_position = np.array(goal_positions)
        # self.init_position = np.array([0, 0, 2])
        self.select_random_positions()
        
        #When positions aren't provided this is the initial positions
        #each line checks if there are any initials
        #when there aren't any then chose one of the default positions
    
        # Gym environment parameters
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        obsticleDimension = 9 + 2 # drone state 9 + goal state 3
        obsticleDimension+= len(self.obstacle_positions)*3 #obstacle positions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            # shape=(9,),  # Full state vector 
            shape=(4,),
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
        
        current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        current_goal = np.array(self.current_goal_pos)
        obstacles = np.array([ obstacle[:3] for obstacle in self.obstacle_positions]).flatten()

        # Compute observation (full state vector)
        observation = np.concatenate([
            current_pos,
            # [self.drone.x, self.drone.y, self.drone.z],  # Position
            [self.drone.vx, self.drone.vy, self.drone.vz],  # Velocity
            [self.drone.phi, self.drone.theta, self.drone.psi],  # Orientation
            current_goal,
            obstacles
            
        ])

        # Distance calculations
        # current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        dist_to_go = np.linalg.norm(current_pos - np.array(self.goal_positions))
        total_dist = np.linalg.norm(np.array(self.init_positions) - np.array(self.goal_positions))

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
        self.select_random_positions()
        # Reset drone and tracking
        self.drone.__reset__()
        self._current_step = 0
        
        current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        
        return np.concatenate([
            current_pos,  # Position
            [self.drone.vx, self.drone.vy, self.drone.vz],  # Velocity
            [self.drone.phi, self.drone.theta, self.drone.psi],  # Orientation
            self.current_goal_pos,  # Goal position
            # Add obstacle positions to observation
            np.array([obs[:3] for obs in self.obstacles]).flatten()
        ])

        # Return initial observation
        # return np.array([
        #     self.drone.x, self.drone.y, self.drone.z,  # Position
        #     self.drone.vx, self.drone.vy, self.drone.vz,  # Velocity
        #     self.drone.phi, self.drone.theta, self.drone.psi  # Orientation
        # ])
    
    def select_random_positions(self):
        #this choses random initials based on what you give it
        self.current_initial_pos = tuple(self.init_positions[
            np.random.randint(len(self.init_positions))
        ])
        self.current_goal_pos = tuple(self.goal_positions[
            np.random.randint(len(self.goal_positions))
        ])
        
        #   # Verify the type and unpack safely
        # if not isinstance(self.current_initial_pos, (tuple, list)):
        #     raise ValueError(f"Initial position must be a tuple/list, got {type(self.current_initial_pos)}")
        
        # # Unpack with error handling
        # try:
        #     x, y, z = self.current_initial_pos
        #     self.drone.x = x
        #     self.drone.y = y
        #     self.drone.z = z
        # except ValueError:
        #     raise ValueError(f"Initial position must be (x,y,z), got {self.current_initial_pos}")
        
        # # Similarly for goal position
    #     self.current_goal_pos = self.goal_positions[
    #         np.random.randint(len(self.goal_positions))
    # ]
        # Reset drone to the selected initial position
        # print(self.drone.x, self.drone.y, self.drone.z)
        # type(self.drone.x, self.drone.y, self.drone.z)
        self.drone.x, self.drone.y, self.drone.z = self.current_initial_pos
        
    #check obstacle collision takes a position of the dron
    #if the drone is within the readius of the object then they have collided
    # return True if it collides and False if not at each step
    def check_obstacle_collision(self, droneposition):
        for obstacleX, obstacleY, obstaclez, ObstacleRad in self.obstacle_positions:
            distance_to_obstacle = np.linalg.norm(
                np.array(droneposition) - np.array([obstacleX, obstacleY, obstaclez])
            )
            if distance_to_obstacle < ObstacleRad:
                return True
        return False
        
    def render(self):
        # Render the simulation
        self.ui.update()

    def close(self):
        # Close the visualization
        plt.close()