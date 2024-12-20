# Drone Flight with Reinforcement Learning
## Contribution
### Goal for Contribution
The goal of this project is to use Reinforcement Learning to control a quad motor drone. Not only having a autonomous model fly the drone but make it to a destination with obstacles in the way. Many options for RL are available but the most reliable and quick to train was a PPO (Proximal Policy Optimization) model. Using the autonomous model my goal is to provide somewhat smooth flight while heading to a goal position.

![Screenshot 2024-12-13 180738](https://github.com/user-attachments/assets/18e23ad8-f716-4320-beb4-3c513484b0ee)

### Here is a list of things I contributed to the library and for my project
1. Graphics: implementation for other objects to be plotted in matplot:
   * Functions for rendering Object and Goals
2. Modified environment: adapted environment training models, usable for stablebaseline3 library and pytorch
   * betterenv: first iteration where convert library for use in models, major changes for additional observation, collision, and reward functions
   * simpleenv: modification to betterenv, but reward is simplified because of problems with fault rewarding for incorrectness
4. Model: Added PPO model training/result files
5. test file: modification to test file
   
### Results for Project
[Drone_Reinforcement_Learning.pdf](https://github.com/user-attachments/files/18215687/Drone_Reinforcement_Learning.pdf)



https://github.com/user-attachments/assets/79a0f737-520e-48b7-9f17-b48c5e68f6d3



### v Below is the Default Library provided by the previous developers v

# Drone_sim [WIP]
A simple Drone dynamics simulation written in Python.
## Structure
The repository is divided into 3 main sub-modules:
1. `sim`: The files that simulate the Rigid Body Dynamics, Sensors etc. are here.
2. `viz`: The files that visualise the Drone Simulation are here.
3. `env`: A Simple GYM environment is implemented here.

## Installation Instructions
1. Clone the repository:
```bash
git clone git@github.com:SuhrudhSarathy/drone_sim.git 
```
2. Install using pip
```
cd drone_sim
pip install .
```
3. Alternatively, you can install it diretly using pip
```bash
python -m pip install git+<repository link>
```
- If the installation, doesn't work try updating the pip by running the following command
```bash
python -m pip install --upgrade pip
```
## Usage
A simple drone can be simulated using
```python
from drone_sim.sim import Drone
from drone_sim.sim.parameters import NULL_ROT
from drone_sim.viz import Body
from drone_sim.viz import Graphics
from drone_sim.sim import IMU

drone = Drone(x=0, y=0, z=2.5, enable_death=True)

body = Body()
imu = IMU()

body.attach_to(drone)
imu.attach_to(drone)

ui = Graphics()
ui.add_actor(drone)

omega = NULL_ROT

for t in range(1000):
   # Steps the simulation by setting Rotor Velocities
   drone.step([omega, omega, omega, omega])

   # Get the sensor data
   sensor_data = imu.sense()

   # Update the UI to display the simulation
   ui.update()
```
