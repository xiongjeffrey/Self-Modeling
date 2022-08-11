from matplotlib.pyplot import get
from pwm.perfectWM import deathNet as predNN
from pwm.perfectWM import goalNet as goalNN
from pwm.perfectWM import predictDeath as predictPred
from pwm.perfectWM import predictGoal

import numpy as np

import scipy
from scipy.spatial import distance as dst

def fear(pred_loc, goal_loc, agent_loc, mode = 'perfect'): # inputs pred_loc, goal_loc, and agent_loc WITH MOVEDIR SET
    
    #!TODO: build a neural net that optimizes for fear calculation? could be cool!

    #!TODO: integrate certainty of future position of pred and goal into calculations
    #!TODO: learn this with a GAN somehow? it's basicaly a PvP game from the perspective of the agent depending on the certainty of knowing the "opponent's" moves

    next_pred_loc = predictPred(pred_loc)
    next_goal_loc = predictGoal(goal_loc)
    
    if mode == 'perfect': # only avoids if actually hits pred
        
        if next_pred_loc == agent_loc: return 1
        else: return -1

    elif mode == 'step': # avoids if gets closer to pred; Euclidean

        if dst.euclidean(pred_loc, agent_loc) > dst.euclidean(next_pred_loc, agent_loc): return 1
        else: return -1

    elif mode == 'stepAware': # returns fear inversely correlated with distance to the predator

        if dst.euclidean(pred_loc, agent_loc) > dst.euclidean(next_pred_loc, agent_loc):
            return 1/dst.euclidean(next_pred_loc, agent_loc)
        else: return -1

    elif mode == 'stepAll': # avoids if gets closer to pred than it gets closer to goal
        
        if dst.euclidean(next_pred_loc, agent_loc) > dst.euclidean(next_goal_loc, agent_loc): return 1
        else: return -1

def scared(fear, threshold = 0) -> bool: # returns a boolean of whether or not agent should be scared given the state
    return fear > threshold