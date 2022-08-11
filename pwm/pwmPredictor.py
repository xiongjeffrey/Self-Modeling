from pwm.perfectWM import deathNet as predNN
from pwm.perfectWM import goalNet as goalNN

import csv
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

"""
Take WM and use it so that you can predict the next state and calculate the reward

From this, create table of all 7/8 possible rewards for each possible action and for each action you can use the WM which tells you final state of all action and you can get the reward for all directions

s_t = [(x,y) pred, (x, ) goal, (x, y) agent]
feed into agent 
agent --> planning --> pwm --> s_t+1 w/ reward for each possible action

no learning here, no networks, but get used to the process of using the working model for planning 
"""

# Reading in states
# states[t] := [[(x, y) pred], [(x, y) goal], [(x, y) agent]]
# states[t] is the t-th row of pwmData.csv

# State is a class that allows us to store locations at time t easily

states = []
class State:
    def __init__(self, pred_loc, goal_loc, agent_loc) -> None:
        self.pred_loc = pred_loc
        self.goal_loc = goal_loc
        self.agent_loc = agent_loc

with open('pwm\\pwmData.csv', 'r') as csvfile: # courtesy of https://stackoverflow.com/questions/13428318/reading-rows-from-a-csv-file-in-python

    r = csv.reader(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    for _, line in enumerate(r):
        states.append(State([int(line[0][1]), int(line[0][4])], 
        [int(line[1][1]), int(line[1][4])],
        [int(line[2][1]), int(line[2][4])]))

# Create Agent



# Agent plans and passes to PWMs




# given all 8 possible movements, predict the reward from each of them
def predict_reward(movedir, currState, wmState, penalty):

    if currState.agent_loc + movedir == wmState.pred_loc: # if movement puts you into death
        return -1

    elif currState.agent_loc + movedir == wmState.goal_loc: # if movement puts you in goal
        return 1

    else: # returns penalty to encourage quick discovery
        #!TODO: check with Fernanda to see if this is the proper method
        return -1*penalty