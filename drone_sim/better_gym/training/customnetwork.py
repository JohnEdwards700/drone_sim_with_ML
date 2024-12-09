import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from betterenv import SimpleDroneEnv

class ReinforcementModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReinforcementModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
            
        )
    def forward(self,x):
        return self.network(x)
    
def tainModel():
    env = SimpleDroneEnv()
    model = ReinforcementModel(input_dim=9, output_dim=4)
    optimizer = optim.Adam(model.parameters())
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        totalReward = 0 
        while not done:
            tensorState = torch.FloatTensor(state).unsqueeze(0)
            actions = ReinforcementModel(tensorState)
            action = actions.sample()
            next_state, reward, done, _ = env.step(action.numpy())

            # Update policy (simplified example)
            optimizer.zero_grad()
            loss = -torch.log(actions.gather(1, action.unsqueeze(1))).mean() * reward
            loss.backward()
            optimizer.step()

            state = next_state
            totalReward += reward

        print(f"Episode {epoch}, Total Reward: {totalReward}")

            