import numpy as np 
import operator
import math
from tabulate import tabulate
import torch
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

class PreyPreatorEnv:
    def __init__(self, goal_loc, predator_loc, agent_loc, grid_size):
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

    def agent_died(self):
      return self.predator_loc == self.agent_loc
    
    def agent_won(self):
      return self.goal_loc == self.agent_loc

    
    def same_loc(self):
      if not self.agent_died():
        self.grid[self.agent_loc] = 9
      self.grid[self.predator_loc] = -1

    def observation(self):
      str1 = str(self.goal_loc[0])
      str2 = str(self.goal_loc[1])
      str3 = str(self.predator_loc[0])
      str4 = str(self.predator_loc[1])
      str5 = str(self.agent_loc[0])
      str6 = str(self.agent_loc[1])
      return int(str1 + str2 + str3 + str4 + str5 + str6)


    def time_update(self):
      # Move the obstacle and goal towards a specific direction 
      self.step_count += 1
      self.grid = np.zeros((self.grid_size, self.grid_size))
      self.update_entity("goal")
      self.update_entity("predator")
      # if self.step_count == 1:
      #   self.grid[self.agent_loc] = 9

      return torch.tensor((self.goal_loc, self.predator_loc, self.agent_loc)), self.grid_size

    def update_entity(self, entity):
      
      if entity == "goal":
        entity_dir = self.goal_dir
        dxy_loc = DIR_TO_VEC[entity_dir]
        ogx = self.goal_loc[0]
        ogy = self.goal_loc[1]
      elif entity == "predator":
        entity_dir = self.predator_dir
        dxy_loc = DIR_TO_VEC[entity_dir]
        ogx = self.predator_loc[0]
        ogy = self.predator_loc[1]
      elif entity == "agent":
        entity_dir = self.agent_dir
        dxy_loc = DIR_TO_VEC[entity_dir]
        ogx = self.agent_loc[0]
        ogy = self.agent_loc[1]

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

    def change_agent_dir(self, dir):
      self.agent_dir = dir

    def dist_goal(self):
      return np.linalg.norm(np.array(self.agent_loc)-np.array(self.goal_loc))/self.grid_size
    
    def dist_predator(self):
      return np.linalg.norm(np.array(self.agent_loc)-np.array(self.predator_loc))/self.grid_size

    def reward(self):
        if self.agent_loc == self.goal_loc:
          done = True
          return 1, done
        elif self.agent_loc == self.predator_loc:
          done = True
          return -1, done
        else:
          done = False
          return 0, done

    def vis_env(self): 
      table = tabulate(self.grid.T, tablefmt="fancy_grid")
      print(table)

    def copy(self):
      from copy import deepcopy
      return deepcopy(self)