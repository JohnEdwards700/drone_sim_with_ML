import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

import os
from betterenv import SimpleDroneEnv

logDirectory = 'tmp/'
os.makedirs(logDirectory, exist_ok=True)

def make_env(env_id, rank, seed=0):
    def __init():
        env = SimpleDroneEnv()