# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:49:43 2025

@author: jtm44
"""

import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp


class RoomHeatEnv(gym.Env):
    def __init__(self,num_rooms,lim=10**6,period=0.01):
        super(RoomHeatEnv,self).__init__()
        self.num_rooms=4
        self.time_step=0
        self.lim=lim
        self.period=period
        self.state=np.round(np.random.randn(self.num_rooms),1)*3+20
        self.external_temp=max(0,np.random.randn()*0+15)  # External ambient temperature (constant)
        self.heat_transfer_coefficients=np.maximum(1+np.round(np.random.randn(self.num_rooms),1)*0.1,0)
        self.external_transfer_coefficients=np.maximum(0.5+np.round(np.random.randn(self.num_rooms),1)*0.02,0)
        self.room_heat_capacities=1000*np.ones((self.num_rooms,))
        self.action_space = spaces.Box(low=0, high=10, shape=(self.num_rooms,), dtype=np.float32)  # Heat source power levels (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_rooms + 1,),  # Room temperatures + external temperature
            dtype=np.float32
        )
    def external_temperature(self,t):
        return self.external_temp+5*np.sin(t*2*np.pi/(24*60*60*self.period))  # External ambient temperature (constant)
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
    def step(self,action):
        heat_source_amplitudes=action
        def heat_transfer_model(t, T):
            dTdt = np.zeros(self.num_rooms)
            for i in range(self.num_rooms):
                if i > 0:  # Transfer from the left room
                    dTdt[i] += self.heat_transfer_coefficients[i-1] * (T[i-1] - T[i]) / self.room_heat_capacities[i]
                if i < self.num_rooms - 1:  # Transfer from the right room
                    dTdt[i] += self.heat_transfer_coefficients[i] * (T[i+1] - T[i]) / self.room_heat_capacities[i]
                # Heat loss to the external environment
                dTdt[i] += self.external_transfer_coefficients[i] * (self.external_temperature(self.time_step) - T[i]) / self.room_heat_capacities[i]
                # Heat source contribution
                dTdt[i] += heat_source_amplitudes[i] / self.room_heat_capacities[i]
            return dTdt
        next_state = solve_ivp(
            heat_transfer_model,
            (self.time_step, self.time_step + 1),
            self.state
        ).y[:, -1]
        # Compute reward (example: minimize deviation from 22°C)
        reward = -np.mean((next_state[:-1] - 22)**2)**0.5  # Penalize deviations from 22°C
        
        self.time_step += 1
        self.state = next_state
     
        if self.time_step>=self.lim:
            done=True
            
        else:
            done=False
        return np.append(self.state, self.external_temperature(self.time_step)),reward,done, {}
    
    def reset(self):
        self.time_step=0
        self.state=np.round(np.random.randn(self.num_rooms),1)*3+20
        return np.append(self.state, self.external_temperature(self.time_step))




