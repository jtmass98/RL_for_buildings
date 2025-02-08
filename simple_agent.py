# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:15:45 2025

@author: jtm44
"""
import roomHeatEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
from deep_q_learning import Actor,Critic,test_ddpg_model
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#%%
N=10**4
env=roomHeatEnv.RoomHeatEnv(4, 15,N)
#%%
# Hyperparameters
gamma = 0.99  # Discount factor
tau = 0.05  # Soft update rate for target networks
lr_actor = 1e-4
lr_critic = 1e-4
replay_buffer_capacity = 100000
batch_size = 64

# Initialize networks
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]  # Max action value from the environment

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
target_actor = Actor(state_dim, action_dim, max_action) #the networks that don't have their weights upgraded
target_critic = Critic(state_dim, action_dim)

# make target networks with the main networks
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
# Replay buffer
replay_buffer = deque(maxlen=replay_buffer_capacity)
#%%
# Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)



# Add experiences to replay buffer
def push_to_replay_buffer(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

# Sample a batch from the replay buffer
def sample_from_replay_buffer():
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones))

# Training loop
for episode in range(1000):  # Number of episodes
    state = env.reset()
    done = False
    episode_reward = 0

    for i in tqdm(range(N)):
        # Select action using the actor (add noise for exploration)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = actor(state_tensor).detach().numpy()
        action += np.random.normal(0, 0.1, size=action_dim)  # Exploration noise
        action = np.clip(action, env.action_space.low, env.action_space.high)
     
        # Step in the environment
        next_state, reward, done, _ = env.step(action)
        push_to_replay_buffer(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train the networks if enough samples are in the buffer
        if len(replay_buffer) > batch_size:
            # Sample a batch
            states, actions, rewards, next_states, dones = sample_from_replay_buffer()
            
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Update Critic
            with torch.no_grad():
                target_actions = target_actor(next_states)
                target_q_values = target_critic(next_states, target_actions)
                y = rewards + gamma * (1 - dones) * target_q_values #1-dones tells you whether the episode ended i.e. should you carry on

            q_values = critic(states, actions)
            critic_loss = nn.MSELoss()(q_values, y) ##temporal difference learning part

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update Actor
            actor_loss = -critic(states, actor(states)).mean()  # Maximize Q-value - actor should use the critic as its policy
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update Target Networks
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    print(f"Episode {episode + 1}, Reward: {episode_reward}, Actor Loss: {actor_loss},Critic Loss: {critic_loss}")
    


#%%
env2=roomHeatEnv.RoomHeatEnv(4,4*10**4,0.01)
#%%
states,actions=test_ddpg_model(env2, actor)

#%%
print(env2.heat_transfer_coefficients,env2.external_transfer_coefficients)
colors=['red','blue','green','black']
plt.plot(states[:,-1],'purple')
for i in range(4):
    plt.plot(states.T[i,:],colors[i],label=str(i))
    plt.plot(actions.T[i,:],colors[i])
plt.legend()