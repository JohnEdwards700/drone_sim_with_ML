
#This is simpledroneenv.py

import gym
from gym import spaces
from drone_sim.sim import Drone
from drone_sim.sim import PositionTracker, IMU
from drone_sim.sim.parameters import *
from drone_sim.viz import Graphics
from drone_sim.viz import Body

import numpy as np

class SimpleDroneEnv(gym.Env):
    metadata = {'render.modes': ["human"]}

    def __init__(self, goal_position=[0, 5, 5]) -> None:
        super(SimpleDroneEnv, self).__init__()

        # Initialise the Drone class, attach body and sensors, and make the Graphics object
        # Reset will be manually called in the step function
        self.drone = Drone(0, 0, 2, enable_death=False)
        self.body = Body()

        self.body.attach_to(self.drone)

        self.ui = Graphics()
        self.ui.add_actor(self.drone)

        self.goal_position = goal_position
        self.init_position = [0, 0, 2]
        # Gym
        self.obs_low = -np.inf
        self.obs_high = -np.inf
        
        # Define action and observation space
        # We will use a continous action space for the values of the motor's rotation
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, shape=(3, 3), dtype=np.float32)
        self.reward_range = (0, 100)

        # Some rotation values
        self.NULL_ROT = NULL_ROT

    def step(self, action):
        assert self.action_space.contains(action), f"{action} doesnot exist in the action space"

        # The action is of the type (w1, w2, w3, w4)
        self.drone.step(action*NULL_ROT)


        # Function has to return the observations
        observation = np.array(
            [
                [self.drone.x, self.drone.y, self.drone.z],
                [self.drone.acceleration[0][0], self.drone.acceleration[1][0], self.drone.acceleration[2][0]],
                [self.drone.p, self.drone.q, self.drone.r]
            ]
        )

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


        return observation, reward, done, {}

    def reset(self):
        # Function to reset the simulation
        self.drone.__reset__()
        observation = np.array(
            [
                [self.drone.x, self.drone.y, self.drone.z],
                [self.drone.acceleration[0][0], self.drone.acceleration[1][0], self.drone.acceleration[2][0]],
                [self.drone.p, self.drone.q, self.drone.r]
            ]
        )

        return observation

    def render(self):
        # Function to render the simulation
        self.ui.update()

    # Helper functions
    def dist(self, x1, x2):
        x, y, z = x1
        X, Y, Z = x2

        return np.sqrt((x-X)**2 + (y-Y)**2 + (z-Z)**2)

#End of simpledroneenv


#drone.py
import numpy as np
from numpy import sin as s, cos as c, tan as t
from drone_sim.sim.parameters import *

class Drone:
    def __init__(self, x=0, y=0, z=0.5, enable_death=True):
        # Position
        self.x, self.y, self.z = x, y, z

        # Roll Pitch Yaw
        self.phi, self.theta, self.psi = 0, 0, 0

        # Linear velocities
        self.vx, self.vy, self.vz = 0, 0, 0

        # Angular Velocities
        self.p, self.q, self.r = 0, 0, 0

        self.linear_position = lambda: np.array([self.x, self.y, self.z]).reshape(3, 1)
        self.angular_position = lambda: np.array([self.phi, self.theta, self.psi]).reshape(3, 1)
        self.linear_velocity = lambda: np.array([self.vx, self.vy, self.vz]).reshape(3, 1)
        self.angular_velocity = lambda: np.array([self.p, self.q, self.r]).reshape(3, 1)

        # Omegas
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0

        # Inertia Matrix
        self.inertia = np.diag([IXX, IYY, IZZ])
        
        # Drag Matrix
        self.drag = np.diag([AX, AY, AZ])

        # Thrust Vector
        self.thrust = np.array(
            [
                [0],
                [0],
                [K*(self.w1**2 + self.w2**2 + self.w3**2 + self.w4**2)]
            ])

        # Torque Vector
        self.torque = np.array(
            [
                [L*K*(self.w1**2 - self.w3**2)],
                [L*K*(self.w2**2 - self.w4**2)],
                [B*(self.w1**2 - self.w2**2 + self.w3**2 - self.w4**2)]
            ]
        )
        # Drag Force Vector
        self.fd = -self.drag@self.linear_velocity()

        # Gravity Vector
        self.gravity = np.array([0, 0, -G]).reshape(-1, 1)

        # Transformation Matrices
        self.R_phi = np.array(
            [
                [c(self.phi), -s(self.phi), 0],
                [s(self.phi), c(self.phi), 0],
                [0, 0, 1]
            ]
        )

        self.R_theta = np.array(
            [
                [1, 0, 0],
                [0, c(self.theta), -s(self.theta)],
                [0, s(self.theta), c(self.theta)]
            ]
        )

        self.R_psi = np.array(
            [
                [c(self.psi), -s(self.psi), 0],
                [s(self.psi), c(self.psi), 0],
                [0, 0, 1]
            ]
        )

        self.R = self.R_phi @ self.R_theta @ self.R_psi

        self.W =np.array(
            [
                [1, 0, -s(self.theta)],
                [0, c(self.phi), c(self.theta)*s(self.phi)],
                [0, -s(self.phi), c(self.theta)*c(self.phi)]
            ]
        )

        self.acceleration = np.zeros((3, 1))

        self.sensors = []
        self.body = None

        # Death
        self.enable_death = enable_death

    def __reset__(self):
        """Call this function to reset the simulation. This is called in function method"""
        # Position
        self.x, self.y, self.z = 0, 0, 0.5

        # Roll Pitch Yaw
        self.phi, self.theta, self.psi = 0, 0, 0

        # Linear velocities
        self.vx, self.vy, self.vz = 0, 0, 0

        # Angular Velocities
        self.p, self.q, self.r = 0, 0, 0

        print("LOG: The Drone is dead. Reset Simulation")
    
    def step(self, velocities):
        """Function to step, i.e. set the angular velocties, to be called externally by the user"""

        self.w1, self.w2, self.w3, self.w4 = velocities[0], -velocities[1], velocities[2], -velocities[3]
        # Decide on this, whether, you need to update as soon as you step or not
        self.update()

        if self.enable_death:
            self.death()

    # All State Update functions
    def __update_transformations__(self):
        self.R_phi = np.array(
            [
                [c(self.phi), -s(self.phi), 0],
                [s(self.phi), c(self.phi), 0],
                [0, 0, 1]
            ]
        )

        self.R_theta = np.array(
            [
                [1, 0, 0],
                [0, c(self.theta), -s(self.theta)],
                [0, s(self.theta), c(self.theta)]
            ]
        )

        self.R_psi = np.array(
            [
                [c(self.psi), -s(self.psi), 0],
                [s(self.psi), c(self.psi), 0],
                [0, 0, 1]
            ]
        )

        self.R = self.R_phi @ self.R_theta @ self.R_psi
        
        self.W =np.array(
            [
                [1, 0, -s(self.theta)],
                [0, c(self.phi), c(self.theta)*s(self.phi)],
                [0, -s(self.phi), c(self.theta)*c(self.phi)]
            ]
        )

    def __update_thrust_and_torque__(self):
        self.thrust = np.array(
            [
                [0],
                [0],
                [K*(self.w1**2 + self.w2**2 + self.w3**2 + self.w4**2)]
            ])

        # Torque Vector
        self.torque = np.array(
            [
                [L*K*(self.w1**2 - self.w3**2)],
                [L*K*(self.w2**2 - self.w4**2)],
                [B*(self.w1**2 - self.w2**2 + self.w3**2 - self.w4**2)]
            ]
        )

        # Drag Force Vector
        self.fd = -self.drag @ self.linear_velocity()

    def __update_acceleration__(self):
        """Uses the omegas to update acceleration"""
        self.acceleration = self.gravity + (1/MASS)*self.R@self.thrust + (1/MASS)*self.fd

    def __update_omega_dot__(self):
        """Updates omega_dot to calculate final state vector"""
        ang_vel = self.angular_velocity()
        cross_pdt = np.cross(ang_vel.reshape(3,), (self.inertia@ang_vel).reshape(3,)).reshape(3, 1)
        MM = self.torque - cross_pdt

        w_dot = np.linalg.inv(self.inertia)@MM
        
        self.p = w_dot[0][0]
        self.q = w_dot[1][0]
        self.r = w_dot[2][0]
    
    def update(self):
        """This function is called everytime to update the state of the system"""
        # At this point, we assume that the angular velocities are set and hence we go on to update
        # simulation step. This will finally be updated as a gym environment, hence we can easily call the 
        # functions defined in the gym environment to update the velocities.
        self.__update_transformations__()
        self.__update_thrust_and_torque__()
        self.__update_acceleration__()
        self.__update_omega_dot__()

        angle = self.angular_position() + self.angular_velocity() * DT

        # Set the angles
        self.phi = self.normalise_theta(angle[0][0])
        self.theta = self.normalise_theta(angle[1][0])
        self.psi = self.normalise_theta(angle[2][0])

        vel = self.linear_velocity() + self.acceleration * DT

        # set the velocities
        self.vx = vel[0][0]
        self.vy = vel[1][0]
        self.vz = vel[2][0]

        position = self.linear_position() + self.linear_velocity() * DT

        # set the positions
        self.x = position[0][0]
        self.y = position[1][0]
        self.z = position[2][0]

    #!--- Helper functions ---!
    def normalise_theta(self, angle):
        """This is used normalise the angle within -pi to pi"""
        if angle > np.pi:
            while angle > np.pi:
                angle -= 2*np.pi
            return angle
        elif angle < -np.pi:
            while angle < np.pi:
                angle += 2*np.pi
            return angle
        return angle

    #!--- Attaching Sensors ---!
    def attach_sensor(self, sensor):
        """This is called when a sensor is added to the drone"""
        self.sensors.append(sensor)
    
    def list_sensors(self):
        """Can be used to list the sensors placed on the drone"""
        for sensor in self.sensors:
            print(sensor.__class__.__name__)

    #!--- Attach Body ---!
    def attach_body(self, body):
        """This is called to attach a body to the drone, i.e. use this to visualise the drone"""
        self.body = body

    #!--- Death Constraints ---#
    def death(self):
        """This function is used to terminate the simulation. This can be enabled or disabled in the constuctor
        If a death condition is reached, the simulation is reset."""
        
        # Condition 1: If the roll or pitch is more than 60 degrees, we reset the simulation
        if abs(self.phi) > np.radians(60.0) or abs(self.theta) > np.radians(60):
            self.__reset__()

        # Condition 2: If the z altitude goes negative, we reset the simulation
        if self.z < 0:
            self.__reset__()
            
# end of drone 


# paramter.py
"""File to Store common parameters for the drone"""

#! -- Simulation parameters --!
DT = 0.004
#! -- ********* --!

#! -- Drone parameters --!
MASS = 0.468
RADIUS_PROP = 0.1
K = 2.980e-6
B = 1.140e-7
L = 0.225
G = 9.8

NULL_ROT = 620.2943

# Inertia
IXX = 4.856e-3
IYY = 4.856e-3
IZZ = 8.801e-3

# Drag
AX = 0.25
AY = 0.25
AZ = 0.25

#! -- ********* --!

#! -- Sensor Parameters --!
# GPS
GPS_MEAN = 0
GPS_STDDEV = 0.1

# IMU
IMU_MEANS = {
    "accelx": 0,
    "accely": 0,
    "accelz": 0,

    "gyrox": 0,
    "gyroy": 0,
    "gyroz": 0
}
IMU_STDDEV = {
    "accelx": 0.1,
    "accely": 0.1,
    "accelz": 0.1,

    "gyrox": 0.1,
    "gyroy": 0.1,
    "gyroz": 0.1
}
#! -- ********* --!

#! -- Plotting Params --!
PLT_PAUSE = 1e-6

# end of parameters.py

# sensors.py
"""This file aims to simulate some of the senors for the drone"""

import numpy as np
from drone_sim.sim.parameters import *

class Sensor:
    def __init__(self):
        self.drone = None

    def attach_to(self, drone):
        raise NotImplementedError

    def sense(self):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

class PositionTracker(Sensor):
    def __init__(self, add_noise=True):
        super(PositionTracker, self).__init__()

        self.drone = None

        if self.add_noise:
            self.mean = GPS_MEAN
            self.std_dev = GPS_STDDEV
        else:
            self.mean = 0
            self.std_dev = 0

    def attach_to(self, drone):
        self.drone = drone
        self.drone.attach_sensor(self)

    def sense(self):
        self.x = self.drone.x + (self.mean + np.random.randn() * self.std_dev)
        self.y = self.drone.y + (self.mean + np.random.randn() * self.std_dev)
        self.z = self.drone.z + (self.mean + np.random.randn() * self.std_dev)

        return [self.x, self.y, self.z]

class IMU(Sensor):
    def __init__(self, add_noise=True):
        super(IMU, self).__init__()

        self.drone = None
        self.add_noise = add_noise
        if self.add_noise:
            self.initialise_random_walks()
        else:
            self.gyrox_bias = 0
            self.gyroy_bias = 0
            self.gyroz_bias = 0

    def initialise_random_walks(self):
        self.gyrox_bias = 0
        self.gyroy_bias = 0
        self.gyroz_bias = 0

    def update_bias(self):
        self.gyrox_bias += (IMU_MEANS["gyrox"] + np.random.randn() * IMU_STDDEV["gyrox"])*DT
        self.gyroy_bias += (IMU_MEANS["gyroy"] + np.random.randn() * IMU_STDDEV["gyroy"])*DT
        self.gyroz_bias += (IMU_MEANS["gyroz"] + np.random.randn() * IMU_STDDEV["gyroz"])*DT

    def attach_to(self, drone):
        self.drone = drone
        self.drone.attach_sensor(self)
        
    def sense(self):
        if self.add_noise:
            # If noise is added, gyro's drift has a random walk.
            wx = self.drone.p + self.gyrox_bias
            wy = self.drone.q + self.gyroy_bias
            wz = self.drone.r + self.gyroz_bias

            ax = self.drone.acceleration[0] + IMU_MEANS[0] + np.random.randn() * IMU_STDDEV[0]
            ay = self.drone.acceleration[1] + IMU_MEANS[1] + np.random.randn() * IMU_STDDEV[1]
            az = self.drone.acceleration[2] + IMU_MEANS[2] + np.random.randn() * IMU_STDDEV[2]

            # Update bias
            self.update_bias()

        else:
            wx = self.drone.p
            wy = self.drone.q
            wz = self.drone.r

            ax = self.drone.acceleration[0]
            ay = self.drone.acceleration[1]
            az = self.drone.acceleration[2]

        return [ax, ay, az], [wx, wy, wz]
    #end of sensors.py
    
    """This file contains drawing of the body"""
from drone_sim.sim.parameters import *
from drone_sim.sim.drone import Drone
import numpy as np

class Body:
    def __init__(self, viz_ax=None):
        self.drone = None
        self.viz_ax = viz_ax
        # Store all the endpoints in an array
        # We have the drone as with 5 important coordinates
        # Left front rotor, Right front Rotor
        #               Main Body
        # Left Rear rotor, Right read Rotor
        self.d = L/np.sqrt(2)
        self.coords = np.array(
            [
                [0, self.d, -self.d, -self.d, self.d],
                [0, self.d, self.d, -self.d, -self.d],
                [0, 0, 0, 0, 0]
            ]
        )
    
    def attach_to(self, drone):
        self.drone = drone
        self.drone.attach_body(self)

        self.trajectory = {
            "X": [self.drone.x],
            "Y": [self.drone.y],
            "Z": [self.drone.z]
        }

    def plot_body(self):
        assert self.drone is not None, "Add the body to a Drone"
        assert self.viz_ax is not None, "Don't know where to plot. Pass the axes when constructing the Object"

        # First transform all the points to the global frame
        coords = self.drone.R @ self.coords + self.drone.linear_position()

        origin = coords[:, 0]
        rf = coords[:, 1]
        lf = coords[:, 2]
        lr = coords[:, 3]
        rr = coords[:, 4]

        self.viz_ax.plot([origin[0], rf[0]], [origin[1], rf[1]], [origin[2], rf[2]], color="red")
        self.viz_ax.plot([origin[0], lf[0]], [origin[1], lf[1]], [origin[2], lf[2]], color="blue")
        self.viz_ax.plot([origin[0], lr[0]], [origin[1], lr[1]], [origin[2], lr[2]], color="black")
        self.viz_ax.plot([origin[0], rr[0]], [origin[1], rr[1]], [origin[2], rr[2]], color="green")

        self.viz_ax.scatter([origin[0]], [origin[1]], origin[2], color="yellow", s=2)

    def update_trajectory(self):
        self.trajectory["X"].append(self.drone.x)
        self.trajectory["Y"].append(self.drone.y)
        self.trajectory["Z"].append(self.drone.z)

    def plot_trajectory(self):
        self.viz_ax.plot(self.trajectory["X"], self.trajectory["Y"], self.trajectory["Z"], "gray")
#end of sensors.py

# body.py
"""This file contains drawing of the body"""
from drone_sim.sim.parameters import *
from drone_sim.sim.drone import Drone
import numpy as np

class Body:
    def __init__(self, viz_ax=None):
        self.drone = None
        self.viz_ax = viz_ax
        # Store all the endpoints in an array
        # We have the drone as with 5 important coordinates
        # Left front rotor, Right front Rotor
        #               Main Body
        # Left Rear rotor, Right read Rotor
        self.d = L/np.sqrt(2)
        self.coords = np.array(
            [
                [0, self.d, -self.d, -self.d, self.d],
                [0, self.d, self.d, -self.d, -self.d],
                [0, 0, 0, 0, 0]
            ]
        )
    
    def attach_to(self, drone):
        self.drone = drone
        self.drone.attach_body(self)

        self.trajectory = {
            "X": [self.drone.x],
            "Y": [self.drone.y],
            "Z": [self.drone.z]
        }

    def plot_body(self):
        assert self.drone is not None, "Add the body to a Drone"
        assert self.viz_ax is not None, "Don't know where to plot. Pass the axes when constructing the Object"

        # First transform all the points to the global frame
        coords = self.drone.R @ self.coords + self.drone.linear_position()

        origin = coords[:, 0]
        rf = coords[:, 1]
        lf = coords[:, 2]
        lr = coords[:, 3]
        rr = coords[:, 4]

        self.viz_ax.plot([origin[0], rf[0]], [origin[1], rf[1]], [origin[2], rf[2]], color="red")
        self.viz_ax.plot([origin[0], lf[0]], [origin[1], lf[1]], [origin[2], lf[2]], color="blue")
        self.viz_ax.plot([origin[0], lr[0]], [origin[1], lr[1]], [origin[2], lr[2]], color="black")
        self.viz_ax.plot([origin[0], rr[0]], [origin[1], rr[1]], [origin[2], rr[2]], color="green")

        self.viz_ax.scatter([origin[0]], [origin[1]], origin[2], color="yellow", s=2)

    def update_trajectory(self):
        self.trajectory["X"].append(self.drone.x)
        self.trajectory["Y"].append(self.drone.y)
        self.trajectory["Z"].append(self.drone.z)

    def plot_trajectory(self):
        self.viz_ax.plot(self.trajectory["X"], self.trajectory["Y"], self.trajectory["Z"], "gray")

# End of body.py

#visualiser.py
"""This file is for visualising the Drone while it is in flight using Matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_sim.sim.drone import Drone
from drone_sim.sim.parameters import PLT_PAUSE

class Graphics:
    """This only generates the 3D Plot of the ongoing simulation and has multi drone support"""
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(projection="3d")

        self.actors = []

    def add_actor(self, drone):
        """This add a Drone for the Graphics object to display"""
        drone.body.viz_ax = self.ax
        self.actors.append(drone)

    def plot_background(self):
        self.ax.set_xlim3d([-5.0, 15.0])
        self.ax.set_ylim3d([-5.0, 15.0])
        self.ax.set_zlim3d([-1.0, 5.0])

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title("Quadcopter Simulation")

    def plot_actors(self):
        plt.cla()
        self.plot_background()
        for actor in self.actors:
            actor.body.plot_body()
            actor.body.update_trajectory()
            actor.body.plot_trajectory()
        plt.pause(PLT_PAUSE)

    def update(self):
        self.plot_actors()
        
#end of visualiser.py

# test.py

"""Tests a multidrone simulation"""
from drone_sim.sim import Drone
from drone_sim.viz import Body
from drone_sim.viz import Graphics
from drone_sim.sim.parameters import *

from time import time

import numpy as np

def calculate_rotor_vels(f, tphi, ttheta, tpsi):
    b = np.array([f, tphi, ttheta, tpsi]).reshape(-1, 1)

    a = np.array(
        [
            [K, K, K, K],
            [L*K, 0, 0, -L*K],
            [0, L*K, -L*K, 0],
            [B, -B, B, -B]
        ]
    )
    
    x = np.linalg.solve(a, b)

    return x

drone = Drone(True)
drone.z = 2.5

drone.phi = 1
print(drone.phi)

# Make a body
body = Body()
body.attach_to(drone)


# Make Graphics object
ui = Graphics()
ui.add_actor(drone)

T = (AX * 5)/np.sin(drone.phi)
print(T)

for i in range(1050):
    x = calculate_rotor_vels(T, 0, 0, 0)
    # w1, w2, w3, w4 = x[0][0], x[1][0], x[2][0], x[3][0]

    if i < 25:

        w1, w2, w3, w4 = NULL_ROT, NULL_ROT, 1.4*NULL_ROT, 1.1*NULL_ROT
    else:
        w1, w2, w3, w4 = NULL_ROT, NULL_ROT, NULL_ROT, NULL_ROT
    # print(drone.vx)

    time_ = time()
    drone.step([w1, w2, w3, w4])
    ui.update()
    
# end of test.py