from cgitb import reset
import gym
import numpy as np
import numpy as np 
import operator
import math
from tabulate import tabulate
import torch
from gym import spaces
from prey_pred_helpers.instantiate_env import *

import csv

import numpy as np 
from PIL import Image
import cv2


render_images = []

colors = {1:(255, 0, 0), 2:(0,255,0), 3:(0,0,255)}
N_CHANNELS = 3
HEIGHT = 200
WIDTH = 200

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    np.array((1, 0)), # Pointing right (positive X) : 0
    np.array((1,1)), # Downright (positive x, positive y)
    np.array((0, 1)), # Down (positive Y) : 2
    np.array((-1,1)), # Downleft (negative x, positive y)
    np.array((-1, 0)), # Pointing left (negative X) : 4 
    np.array((-1,-1)), # Upleft (negative x, negative y)
    np.array((0, -1)), # Up (negative Y) : 6 
    np.array((1,-1)), # Upright (positive x, negative y)
]

class PWMEnv(gym.Env):
    def __init__(self):
        self.all_envs, self.win_count, self.dead_count = init_envs(1)
        init_cond = self.all_envs[0]
        predator_loc, goal_loc, agent_loc, grid_size = init_cond[0], init_cond[1], init_cond[2], 5
        self.goal_loc = goal_loc
        self.predator_loc = predator_loc
        self.agent_loc = agent_loc 
        self.ogagent_loc = agent_loc  
        self.grid_size = grid_size
        self.step_count = 0

        self.goal_dir = 6 # north
        self.predator_dir = 5 
        self.agent_dir = 6

        self.grid = np.zeros((grid_size, grid_size))
        self.grid[self.goal_loc] = 1
        self.grid[self.predator_loc] = -1
        self.grid[self.agent_loc] = 9
        self.action_space = gym.spaces.Discrete(8)
        # self.observation_space = gym.spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=grid_size,
                                            shape=(3, 2), dtype=np.uint8)
        # print("****", self.observation_space)

    def set_grid(self, state, grid_size = 5):

        self.predator_loc = state[0]
        self.goal_loc = state[1]
        self.agent_loc = state[2] 
        self.ogagent_loc = state[2]  
        self.grid_size = grid_size
        self.step_count = 0

        self.grid = np.zeros((grid_size, grid_size))
        self.grid[self.goal_loc] = 1
        self.grid[self.predator_loc] = -1
        self.grid[self.agent_loc] = 9
        self.action_space = gym.spaces.Discrete(8)

        self.action_space = gym.spaces.Discrete(8)
        # self.observation_space = gym.spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=grid_size,
                                            shape=(3, 2), dtype=np.uint8)

    def update_entity(self, entity):
        if entity == "goal":
            entity_dir = self.goal_dir
            dxy_loc = entity_dir
            ogx = self.goal_loc[0]
            ogy = self.goal_loc[1]
        elif entity == "predator":
            entity_dir = self.predator_dir
            dxy_loc = entity_dir
            ogx = self.predator_loc[0]
            ogy = self.predator_loc[1]
        elif entity == "agent":
            entity_dir = self.agent_dir
            dxy_loc = entity_dir
            ogx = self.agent_loc[0]
            ogy = self.agent_loc[1]
        
        if type(dxy_loc) != torch.Tensor: # DON'T TOUCH THIS
            dxy_loc = DIR_TO_VEC[dxy_loc]

        x = (ogx + dxy_loc[0])
        y = (ogy + dxy_loc[1])
        new_x = x % self.grid_size
        new_y = y % self.grid_size
        if entity_dir == 1:
            if ogx == self.grid_size - 1:
                new_y = new_x
                new_x = abs(ogx-ogy)
            elif ogy == self.grid_size - 1:
                new_x = new_y
                new_y = abs(ogx-ogy)
        elif entity_dir == 3:
            if new_x == 0 or new_y == 0:
                tempx = ogx
                new_x = ogy
                new_y = tempx
        elif entity_dir == 5:
            if ogx == 0:
                new_y = new_x
                new_x = (self.grid_size-1)-abs(ogx-ogy)
            elif ogy == 0:
                new_x = new_y
                new_y =  (self.grid_size-1)-abs(ogx-ogy)
        elif entity_dir == 7:
            if new_x == 0 or new_y == self.grid_size - 1:
                tempx = ogx
                new_x = ogy
                new_y = tempx

        if entity == "goal":
            self.goal_loc = (new_x, new_y)
            self.grid[self.goal_loc] = 1 
        elif entity == "predator":
            self.predator_loc = (new_x, new_y)
            self.grid[self.predator_loc] = -1
        elif entity == "agent":
            self.agent_loc = (new_x, new_y)
            if not self.agent_died():
                self.grid[self.agent_loc] = 9

    def time_update(self):
        # Move the obstacle and goal towards a specific direction 
        self.step_count += 1
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.update_entity("goal")
        self.update_entity("predator")
        
        with open('pwm\\pwmData.csv', 'a', newline = '') as csvfile:

            w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            w.writerow([self.predator_loc, self.goal_loc, self.agent_loc])

        return self.grid_size
    
    def observation(self):
        str1 = str(self.goal_loc[0])
        str2 = str(self.goal_loc[1])
        str3 = str(self.predator_loc[0])
        str4 = str(self.predator_loc[1])
        str5 = str(self.agent_loc[0])
        str6 = str(self.agent_loc[1])
        return int(str1 + str2 + str3 + str4 + str5 + str6)

    def step(self, action):
        steps = 0
        done = False

        if isinstance(action, np.int64) or isinstance(dir, np.ndarray):
            action = int(action)
        if isinstance(dir, tuple):
            action = int(action[0])

        while not done: 

            self.change_agent_dir(action)
            _ = self.time_update()
            self.update_entity("agent")
            reward, done = self.env_reward(steps)

            steps += 1
    
        # print(reward)
        info = {
            "state": torch.tensor((self.predator_loc, self.goal_loc, self.agent_loc)),
            "grid_size": self.grid_size
        }
        
        # return obs, reward, done, info
        obs = np.zeros(self.observation_space.shape, dtype = self.observation_space.dtype)
        obs[0] = self.predator_loc
        obs[1] = self.goal_loc
        obs[2] = self.agent_loc

        return obs, reward, done, info
        # send this inside info --> gym checks a bunch of stuff so doing this raw will cause errors
        # take that information in with the planning module
    
    def reset(self):
        # all_envs, win_count, dead_count = init_envs(1)
        init_cond = self.all_envs[0]
        predator_loc, goal_loc, agent_loc, grid_size = init_cond[0], init_cond[1], init_cond[2], 5
        self.goal_loc = goal_loc
        self.predator_loc = predator_loc
        self.agent_loc = agent_loc 
        self.ogagent_loc = agent_loc  
        self.grid_size = grid_size
        self.step_count = 0

        self.goal_dir = 6 # north
        self.predator_dir = 5 
        self.agent_dir = 6

        self.grid = np.zeros((grid_size, grid_size))
        self.grid[self.goal_loc] = 1
        self.grid[self.predator_loc] = -1
        self.grid[self.agent_loc] = 9

        obs = np.zeros(self.observation_space.shape, dtype = self.observation_space.dtype)
        obs[0] = self.predator_loc
        obs[1] = self.goal_loc
        obs[2] = self.agent_loc
        return obs

    def change_agent_dir(self, action):

        if isinstance(action, tuple):
            action = int(action[0])
        
        if type(action) != int and len(action) > 1:
            self.agent_dir = next((i for i, val in enumerate(DIR_TO_VEC) if np.all(val == action.numpy())), -1) # courtesy of https://stackoverflow.com/questions/41271997/finding-the-index-of-a-numpy-array-in-a-list
            
        else: self.agent_dir = action

    def env_reward(self, steps):
        reward = 0
        done = False
        if self.agent_won():
            reward = 1
            done = True
        elif self.agent_died():
            reward = -1
            done = True
        elif steps > 20:
            done = True
        return reward, done

    def agent_died(self):
      return self.predator_loc == self.agent_loc
    
    def agent_won(self):
      return self.goal_loc == self.agent_loc
    
    def _render(self, mode = 'human', close = False):
        return 0