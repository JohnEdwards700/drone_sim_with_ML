"""This file is for visualising the Drone while it is in flight using Matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_sim.sim.drone import Drone
from drone_sim.sim.parameters import PLT_PAUSE

class Graphics:
    """This only generates the 3D Plot of the ongoing simulation and has multi drone support"""
    def __init__(self,
                 goal_positions,
                 obstacle_positions):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(projection="3d")

        self.actors = []
        self.goal_positions = list(goal_positions)
        self.obstacle_positions = list(obstacle_positions)

    def add_actor(self, drone):
        """This add a Drone for the Graphics object to display"""
        drone.body.viz_ax = self.ax
        self.actors.append(drone)
    
    def set_goals_obstacles(self, goal_positions, obstacle_positions):
        self.goal_positions = goal_positions
        self.obstacle_positions = obstacle_positions

    def plot_background(self):
        self.ax.set_xlim3d([-5.0, 15.0])
        self.ax.set_ylim3d([-5.0, 15.0])
        self.ax.set_zlim3d([-1.0, 5.0])

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title("Quadcopter Simulation")
        
    def plot_goals(self):
        for i, goal in enumerate(self.goal_positions):
            self.ax.scatter(
                [goal[0]],
                [goal[1]],
                [goal[2]],
                color = 'green',
                s = 100,
                linewidths=3,
                label=f'Goal {i+1}'
                
            )
        if len(self.goal_positions)>1:
            self.ax.legend()

    def plot_actors(self):
        plt.cla()
        self.plot_background()
        for actor in self.actors:
            actor.body.plot_body()
            actor.body.update_trajectory()
            actor.body.plot_trajectory()
        self.plot_goals
        plt.pause(PLT_PAUSE)

    def update(self):
        self.plot_actors()




