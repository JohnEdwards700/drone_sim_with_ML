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
        self.max_episode_steps = 1000
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

        # Distance calculations
        # current_pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        dist_to_go = np.linalg.norm(current_pos - current_goal)
        total_dist = np.linalg.norm(np.array(self.current_initial_pos) - current_goal)

        # Reward calculation
        reward = 0
        
        distance_from_initial = np.linalg.norm(current_pos - np.array(self.current_initial_pos))
        exploration_distance = np.linalg.norm(np.array(self.current_goal_pos) - np.array(self.current_initial_pos))
        
        exploration_reward = (distance_from_initial / exploration_distance)*20*self._current_step 
        reward += exploration_reward
        
        velocity_reward = np.linalg.norm([self.drone.vx, self.drone.vy, self.drone.vz])
        reward += velocity_reward*2
        
        progress_of_distance_reward = max(0,1 - ( total_dist-dist_to_go) / total_dist)
        reward += progress_of_distance_reward*50*self._current_step #as it gets closer more and more rewards
        
        currx, curry, currz = self.current_initial_pos
        
        x_mov = abs(self.drone.x) - abs(currx)
        y_mov = abs(self.drone.y) - abs(curry)
        z_mov = abs(self.drone.z) - abs(currz)
        
        movement = np.sqrt(x_mov**2 + y_mov**2+ z_mov**2)
        if movement > self._current_step:
            reward += movement * self._current_step
        # if self.check_obstacle_collision(current_pos):
        #     reward -= 10
        
        # stability_reward = 1 - min(1,(abs(self.drone.phi) > np.radians(60.0) + abs(self.drone.theta) > np.radians(60.0)))
        
        # reward += stability_reward
        
        stability =abs(self.drone.phi) + abs(self.drone.theta)
        if stability < np.radians(15.0):
            reward +=5
        else:
            #the greater the stability penalty the lower the reward
            reward -= stability*5

        # Termination conditions
        
        # Angle limits

        
        #faster solution, so it wont be slow
        reward -= 0.1 

        # Very close to Goal
        if dist_to_go < 2:
            reward += 20
            
        done = False   
        
        #Altitude limit
        if self.drone.z < -8 or self.drone.z > 8:
            reward -= 20
        if self.drone.y < -1 or self.drone.y > 10:
            reward -=20
        if self.drone.x < -10 or self.drone.x > 10:
            reward -=20
        
        if (abs(self.drone.phi) > np.radians(60.0) or 
        abs(self.drone.theta) > np.radians(60.0)):
            reward -= 50
        elif dist_to_go < 0.01:
            reward += 100*self.current_goal_pos
            
        

        # Maximum steps reached
        self._current_step += 1

        done =(
        abs(self.drone.z) > 10 or  # More vertical space
        abs(self.drone.x) > 15 or  # More horizontal space
        abs(self.drone.y) > 15 or
        abs(self.drone.phi) > np.radians(75.0) or  # More lenient angle
        abs(self.drone.theta) > np.radians(75.0) or
        self._current_step >= self.max_episode_steps or
        dist_to_go < 0.5  
        )
        
        print(f"""
            Rewards Breakdown:
            Exploration Reward: {exploration_reward:.2f}
            Progress Reward: {progress_of_distance_reward:.2f}
            Velocity Reward: {velocity_reward:.2f}
            Total Reward: {reward:.2f}
            Current Position: {current_pos}
            Initial Position: {self.current_initial_pos}
            Goal Position: {self.current_goal_pos}
            Distance from Initial: {distance_from_initial:.2f}
            Distance to Goal: {dist_to_go:.2f}
            Movement reward: {movement:.2f}
            """)
        
        
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