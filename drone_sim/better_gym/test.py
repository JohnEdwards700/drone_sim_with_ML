import gym
import numpy as np
from easyenv import SimpleDroneEnv

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
# env = SimpleDroneEnv(
#     # init_positions=custom_initial_pos, 
#     # goal_positions=custom_goals, 
#     # obstacle_positions=custom_obstacles
#     )

# for t in range(10000):
#     action = np.array([1.0, 0.99, 0.99, 1.0])
#     print(action)

#     obs, reward, done, _ = env.step(action)
#     #print(obs, done)

#     print(reward)

#     env.render()

#     if done:
#         break

def compute_action(drone, target_position, dt=0.1):
    current_pos = np.array([drone.x, drone.y, drone.z])
    position_error = target_position - current_pos
    Kp_xy = 1.0  # Proportional gain for x and y
    Kp_z = 1.0   # Proportional gain for z
    
    thrust = np.clip(Kp_z * position_error[2], 0, 2.0)  
    desired_roll = np.clip(Kp_xy * position_error[0], -np.pi/6, np.pi/6)
    desired_pitch = np.clip(Kp_xy * position_error[1], -np.pi/6, np.pi/6)
    
    force_x = thrust * np.sin(desired_pitch)
    force_y = thrust * np.sin(desired_roll)  
    
    drone.x += force_x * dt
    drone.y += force_y * dt
    drone.z += thrust * dt
    base_thrust = thrust
    w1 = base_thrust + desired_roll - desired_pitch
    w2 = base_thrust - desired_roll - desired_pitch
    w3 = base_thrust - desired_roll + desired_pitch
    w4 = base_thrust + desired_roll + desired_pitch
    
    return np.array([w1, w2, w3, w4])


env = SimpleDroneEnv()
target_position = np.array([3.0, 5.0, -2.0]) 

for t in range(10000):
    action = compute_action(env.drone, target_position)
    
    obs, reward, done, _ = env.step(action)
    
    print(f"Current position: [{env.drone.x}, {env.drone.y}, {env.drone.z}]")
    print(f"Target position: {target_position}")
    
    env.render()
    
    if np.linalg.norm(np.array([env.drone.x, env.drone.y, env.drone.z]) - target_position) < 0.1:
        print("Reached target!")
        break
    
    if done:
        break