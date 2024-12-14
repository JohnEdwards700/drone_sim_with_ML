import os
# from betterenv import SimpleDroneEnv
from easyenv import SimpleDroneEnv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch 
# print(torch.__version__)  # Check PyTorch version
# print(torch.version.cuda)  # Check CUDA version PyTorch was built with
# print(torch.cuda.is_available()) 
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Current device: {torch.cuda.current_device()}")
# print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# from stb3_contrib import RecurrentPPO
# from stable_baselines3.common.env_checker import check_env

models_dir = f'models/{int(time.time())}'
logdir = f'logs/{int(time.time())}'


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# env = SubprocVecEnv([lambda: SimpleDroneEnv() for _ in range(4)])
env = DummyVecEnv([lambda: SimpleDroneEnv()])

# env = DummyVecEnv([lambda: SimpleDroneEnv()])

env.reset()
# check_env(env)

model = PPO.load('drone_sim/better_gym/models/1734155466/15700.zip', env = env) 


def testactions():
    for episode in range(10):
        done = False
        obs = env.reset()
        score = 0
        while True:
            randomaction = env.action_space.sample()
            print('here is the action: ',randomaction)
            obs, reward, done, info = env.step(randomaction)
            score+=reward
            print(f'episode: {episode} Score: {score}')
# testactions()
#def testmodel():
for episode in range(5):
    env = SimpleDroneEnv()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

# model = PPO('MlpPolicy', env=env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2, ent_coef=0.01, tensorboard_log=logdir, verbose=1, device='cuda')
# timestep = 100
# iters = 0
# while True:
#     iters +=1
#     print(f'\n\n\n\n\n\nThis iteration: {iters}\n\n\n\n\n')
#     model.learn(total_timesteps=timestep, reset_num_timesteps=False,tb_log_name='PPO')
#     model.save(f'{models_dir}/{timestep*iters}')