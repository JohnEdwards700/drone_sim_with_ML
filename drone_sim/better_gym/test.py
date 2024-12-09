import gym
import numpy as np
from betterenv import SimpleDroneEnv

#Here is the test to check:
#if when the a number in the action array is greater than 1

env = SimpleDroneEnv()

for t in range(1000):
    action = np.array([1.0, 0.99, 0.99, 1.0])
    print(action)

    obs, reward, done, _ = env.step(action)
    #print(obs, done)

    print(reward)

    env.render()

    if done:
        break