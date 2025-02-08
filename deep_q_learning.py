# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:01:56 2025

@author: jtm44
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action
        self.state_dim=state_dim
        self.action_dim=action_dim
        
    def forward(self, x):
        x=self.correction(x)
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        out= self.fc3(x)
        
        return self.max_action * torch.sigmoid(out) # Scaled continuous action output
    def correction(self,x):
        return x-22

# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.state_dim=state_dim
        self.action_dim=action_dim
    def correction(self,x):
        return x-22    
    def forward(self, state, action):
        state=self.correction(state)
        
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action

        x = torch.relu(self.fc1(x))
 
        x = torch.relu(self.fc2(x))

        return self.fc3(x)  # Q-value
    
def test_ddpg_model(env, actor_model):


 
    state = env.reset()
    states=np.zeros((env.lim,env.num_rooms+1))
    actions=np.zeros((env.lim,env.num_rooms))
    total_reward = 0
    done = False

    for i in tqdm(range(env.lim)):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            # Get action from the actor model
            action = actor_model(state_tensor).cpu().numpy()
        
        # Execute action in the environment
        state, reward, done, _ = env.step(action)
        
        total_reward += reward
        states[i,:]=state
        actions[i,:]=action





    env.close()
    return states,actions