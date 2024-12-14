# import gymnasium as gym
# from gymnasium import spaces
import gym as gym
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


    def __init__(self, seed=None
                #  init_positions:list[(float, float, float)] = None,
                #  goal_positions:list[(float, float, float)] = None,
                #  obstacle_positions:list[(float, float, float, float)] = None,
                #  max_episode_steps=1000
                 ) -> None:
                                # def __init__(self,
                                #              init_positions = None,
                                #              goal_positions= None,
                                #              obstacle_positions = None,
                                #              max_episode_steps=1000
                                #              ) -> None:
        super(SimpleDroneEnv, self).__init__()
        
        # if seed is not None:
        #     np.random.seed(seed)
        
        # self.init_positions = init_positions or [
        #     (0.0, 0.0, 2.0),
        #     (1.0, 1.0, 2.0),
        #     (-1.0, -1.0, 2.0)
        # ]
        
        # self.goal_positions = goal_positions or [
        #     (0.0, 5.0, 5.0),
        #     (5.0, 5.0, 5.0),
        #     (-5.0, 5.0, 5.0)
        # ]
        
        # self.obstacle_positions = obstacle_positions or [
        #     (2.5, 2.5, 3.0, 0.5),   # (x, y, z, radius)
        #     (-2.5, 2.5, 3.0, 0.5),
        #     (0.0, 4.0, 4.0, 0.5)
        # ]
        
        
        self.init_positions = [
            (0.0, 0.0, 2.0),
            (1.0, 1.0, 2.0),
            (-1.0, -1.0, 2.0)
        ]
        
        self.goal_positions = [
            (0.0, 5.0, 5.0),
            (5.0, 5.0, 5.0),
            (-5.0, 5.0, 5.0)
        ]
        
        self.obstacle_positions = [
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
        # COME BACK AND FIX THIS
        #self.ui.set_goals_obstacles(self.goal_positions, self.obstacle_positions)
        self.select_random_positions()
        # Goal and initial positions
        # self.goal_position = np.array(goal_positions)
        # self.init_position = np.array([0, 0, 2])
        
        
        #When positions aren't provided this is the initial positions
        #each line checks if there are any initials
        #when there aren't any then chose one of the default positions
    
        # Gym environment parameters
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        obsticleDimension = 9 + 3 + 3 # drone state 9 + goal state 3 + vector for distance
        obsticleDimension+= len(self.obstacle_positions)*3 #obstacle positions
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            # shape=(9,),  # Full state vector 
            shape=(obsticleDimension,),
            dtype=np.float32
        )
        self.reward_range = (0, 100)
        self.max_episode_steps = 100
        # self.max_episode_steps = max_episode_steps
        self._current_step = 0

    def step(self, action):
        # Validate action
        #assert self.action_space.contains(action), f"{action} does not exist in the action space"
        # if not np.all((action>=-1) & (action <=1)):
        #     print(f'action not between range: {action}')
        #     print(f'highest value={np.max(action)} lowest value={np.min(action)}')
        #     raise ValueError('action must be between 0 and 1')

        # Scale action to motor velocities
        #action = [w1, w2, w3, w4] each are velocities of the drone
        motor_velocities = action * 1000
        print(f'Action: {action}, Motor velocities {motor_velocities}')
        self.drone.step(motor_velocities)
        self.drone.x += np.random.uniform(0, 0.6)
        self.drone.y +=np.random.uniform(0, 0.6)
        current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        current_goal = np.array(self.current_goal_pos)
        obstacles = np.array([ obstacle[:3] for obstacle in self.obstacle_positions]).flatten()
        directiongoal = current_goal - current_pos

        # Compute observation (full state vector)
        observation = np.concatenate([
            current_pos,
            # [self.drone.x, self.drone.y, self.drone.z],  # Position
            [self.drone.vx, self.drone.vy, self.drone.vz],  # Velocity
            [self.drone.phi, self.drone.theta, self.drone.psi],  # Orientation
            current_goal,
            directiongoal,
            obstacles,
            
        ])
        print(f"""
              Drone Position (x,y,z): {self.drone.x, self.drone.y, self.drone.z}
              Drone Velocity (vx, vy, vz): {self.drone.vx,self.drone.vy,self.drone.vz}
              Drone Phi, Theta, psi: {self.drone.phi, self.drone.theta, self.drone.psi}
              """)
        dist_to_goal = np.linalg.norm(current_pos - current_goal)

        reward = 0
        done = False
        #Hover
        # distx = 0.9*(abs(1-self.drone.z))
        # disty = 0.1 *((0-self.drone.x)**2)
        # distz = 0.1 * ((0 - self.drone.y)**2)
        # reward += distz - distx -disty
        #Move
        dist_x = (0 - self.drone.x) ** 2
        dist_y = (0 - self.drone.y) ** 2
        if 1.4 - self.drone.z < 0:
            dist_z = (1.2 - self.drone.z) ** 2 - np.abs(1.4 - self.drone.z)/4
        else:
            dist_z = (1.2 - self.drone.z) ** 2

        reward = -(0.9 * dist_z + 0.05 * dist_x + 0.05 * dist_y)
        
        if (abs(self.drone.phi) > np.radians(60.0) or 
        abs(self.drone.theta) > np.radians(60.0)):
            done = True
        elif dist_to_goal < 0.1:
            done = True
            reward += 100
        
        
        
        return observation, reward, done, {}

    def reset(self):
        self.select_random_positions()
        # Reset drone and tracking
        x, y, z= self.current_initial_pos
        self.drone.__reset__()
        self._current_step = 0
        self.drone.x = 0
        self.drone.y = 0
        self.drone.z = 2
        current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        obstacles_pos = np.array([obs[:3] for obs in self.obstacle_positions]).flatten()
        directiongoal = np.array(self.current_goal_pos) - current_pos
        observation = np.concatenate([
            current_pos,  # Position
            [self.drone.vx, self.drone.vy, self.drone.vz],  # Velocity
            [self.drone.phi, self.drone.theta, self.drone.psi],  # Orientation
            self.current_goal_pos,  # Goal position
            directiongoal,
            obstacles_pos, # obstacle positions to observation
            
            
        ])
        
        return observation

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
            
            #This choses random position
            # self.drone.x, self.drone.y, self.drone.z = self.current_initial_pos
        
    def check_obstacle_collision(self, droneposition):
        #check obstacle collision takes a position of the dron
        #if the drone is within the readius of the object then they have collided
        # return True if it collides and False if not at each step
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