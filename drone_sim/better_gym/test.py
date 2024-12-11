import gym
import numpy as np
from betterenv import SimpleDroneEnv

#Here is the test to check:
#if when the a number in the action array is greater than 1
#This is the Random Agent
#given the action, you plug that into you environment and then the agent does the task
#it will return you observation, reqrd, and the boolean value
custom_initial_pos = [(0,0,2), (3,3,3), (-3,-3,2)]
custom_goals = [(5,5,5), (-5,5,5), (0,7,6)]
custom_obstacles = [
    (2.5, 2.5, 3, 0.5),   # x, y, z, radius
    (-2.5, 2.5, 3, 0.5),
    (0, 4, 4, 0.5)
]
env = SimpleDroneEnv(
    init_positions=custom_initial_pos, 
    goal_positions=custom_goals, 
    obstacle_positions=custom_obstacles)

for t in range(1000):
    action = np.array([1.0, 0.99, 0.99, 1.0])
    print(action)

    obs, reward, done, _ = env.step(action)
    #print(obs, done)

    print(reward)

    env.render()

    if done:
        break