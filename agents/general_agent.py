from pstats import StatsProfile
from typing import List
import numpy as np
from psutil import STATUS_DEAD 
from prey_pred_helpers.prey_pred import PreyPreatorEnv

import csv
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from pwm.pwmHelpers import round_prediction, precode, encode
from pwm.perfectWM import DeathNN, GoalNN

def euclidean(x, y):
  sum = 0
  for i in range(len(x)):
    sum += (x[i] - y[i])**2

  return sum**0.5

# load in nets
predNN = DeathNN()
pred_state_dict = torch.load('predNN.pt')
predNN.load_state_dict(pred_state_dict)

goalNN = GoalNN()
goal_state_dict = torch.load('goalNN.pt')
goalNN.load_state_dict(goal_state_dict)

class Agent():

    """
    Plans using a perfect world model
    No innate behavior 
    params:
    - direction_space: int 8 possible actions (all directions)
    - use_planning: bool to indicate planning 
    """

    def __init__(self, direction_space, agent_params : List, policy, predWM = predNN, goalWM = goalNN):

        
        self.direction_space = direction_space
        if direction_space == 8: #!TODO: find a less crude way to do this >:(

          self.direction_space = [torch.tensor((-1, -1)), torch.tensor((-1, 0)), torch.tensor((-1, 1)), torch.tensor((0, -1)), 
          torch.tensor((0, 1)), torch.tensor((1, -1)), torch.tensor((1, 0)), torch.tensor((1, 1))]

        # agent_params: planning, innate behavior, self-model used or not
        self.planning     = agent_params[0]
        self.innate       = agent_params[1]
        self.selfmodeling = agent_params[2]

        self.actions_taken = 0 # actions taken
        self.imag_act = 0
        self.obs_received = 0 # observations perceived
        if self.planning:
            self.direction_rewards = np.zeros(len(self.direction_space)) # where predicted rewards will be stored

        self.policy = policy
        self.predWM = predWM
        self.goalWM = goalWM

        if self.selfmodeling: # instantiate a self-model
                                # a self-model is a copy of the agent ("abstracted") with no innate behavior
            self.self_model = Agent(self.direction_space, [self.planning, False, False], self.policy, self.predWM, self.goalWM)

        self.comfort_zone = 11

    def set_state_by_obs(self, obs):

        """
        Function to set the state of an imaginary Agent
        """

        goal_loc, predator_loc, self_location, grid_size = obs # unpack onto the location of entities
        self.self_model.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size)

    def set_state_by_state(self, state, grid_size = 5):

        """
        Function to set the state of an imaginary Agent
        """

        predator_loc  = tuple((int(state[0][0]), int(state[0][1])))
        goal_loc      = tuple((int(state[1][0]), int(state[1][1])))
        self_location = tuple((int(state[2][0]), int(state[2][1])))

        self.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size)
        
    def perceive_state(self, obs):
      """
      Function to recieve observations from the environment 
      obs - tuple containing the locations of goal, predator, self and grid_size
      """  
      goal_loc, predator_loc, self_location, grid_size = obs # unpack onto the location of entities

      if self.selfmodeling:
        self.self_model.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size) # create world model instantiated with the new observation
      
      self.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size)

      return self.check_done() # check if agent has to stop perceiving (due to death, tiredness or catching prey)

    def internal_state(self, act_env):
        """
        Function to return the internal state of the agent based world state
        params:
        - act_env - the current world state (position information)
        - return -1 if died, 1 if goal reached and 0 if neither
        stops at 50 due to 'tiredness'
        """
        reward = 0
        done = False
        self.obs_received += 1
        if act_env.agent_won():
            reward = 1
            done = True
        elif act_env.agent_died():
            reward = -1
            done = True
        elif self.obs_received > 50:
            done = True
        return reward, done

    def check_done(self):
      """
      Function to check whether agent has been caught, won or got tired
      """
      reward = 0
      done = False
      scared = False
      if self.reached_goal():
        reward = 1
        done = True
      elif  self.got_caught():
        reward = -1
        done = True
      elif self.got_tired():
        done = True

      if self.is_scared() and self.innate:
        done = True
        scared = True

      return reward, done, scared

    def check_selfmodel_done(self):
      """
      Function to check whether agent has been caught, won or got tired
      """

      reward = 0
      done = False
      scared = False
      if self.selfmodel_reached_goal():
        reward = 1
        done = True
      elif  self.selfmodel_got_caught():
        reward = -1
        done = True
      elif self.selfmodel_got_tired():
        done = True

      # if self.selfmodel_is_scared() and self.innate:
        # done = True
        # scared = True

      return reward, done, scared

    def is_scared(self):
      dista = np.linalg.norm(np.array(self.perfect_world_rep.agent_loc)-np.array(self.perfect_world_rep.predator_loc))
      scared = False

      if dista <= self.comfort_zone:
        scared = True
      return scared

    def get_next_loc(self, loc, vect_dir):

        next_loc = loc + vect_dir
        for j in range(len(next_loc)): # get looping

            if next_loc[j] < 0:
                next_loc[j] = self.perfect_world_rep.grid_size - 1

            if next_loc[j] >= self.perfect_world_rep.grid_size:
                next_loc[j] = 0

        return next_loc
    
    def take_action(self, planning_trial, state):
      """

      Function to take action (either through planning or model-free RL)

      planning := either 0 or 1, representing no planning or planning trial respectively
      state := (predator_coords, goal_coords, agent_coords)

      Returns a coordinate direction of movement for the agent

      """

      self.actions_taken += 1

      # need to encode states so that the NNs work
      wm_pred = self.predWM.forward(state[0]) # use this for naive PWM
      wm_goal = self.goalWM.forward(state[1]) # use this for naive PWM
      # wm_pred = round_prediction(self.predWM(encode(state[0]))) # use this for NN PWM
      # wm_goal = round_prediction(self.goalWM(encode(state[1]))) # use this for NN PWM

      if self.innate:
        triggered, dir = self.get_innate_behavior(wm_pred, wm_goal, state[2])

        if triggered: return dir
      
      if self.selfmodeling:
        
        sim_state = [wm_pred, wm_goal, state[2]]

        planned_dir = self.self_model.plan_greedy(wm_pred, wm_goal, state[2])
        return planned_dir


        # direction_rewards, _ = self.simulate_plan(sim_state) # call planning module to predict rewards for all actions
        # return np.argmax(direction_rewards)

      if self.planning and planning_trial: # naive movement
        
        planned_dir = self.plan_greedy(wm_pred, wm_goal, state[2])
        return planned_dir # select the direction with highest predicted reward or an early-exited innate-behavior direction

      else: # policy movement
        return self.direction_space[self.policy.predict(state)[0]]

    def get_random_dir(self, dir_set):
        # pops out a random element from a numpy array
        # utility is on the set of directions, popping out a random direction from the set
        
        #!TODO: self model could learn what actions are tested first, which in turn aids planning
            #!TODO: in other words, learning which ones to pop out first

        idx = random.randrange(len(dir_set))

        dir = self.direction_space[dir_set[idx]]
        dir_set = np.delete(dir_set, idx, 0)
        return dir, dir_set

    #!TODO: should be able to simulate what innate reactions would look like -- self-awareness

    #!TODO: self model constrains simulation

    def get_innate_behavior(self, pred_loc, goal_loc, agent_loc):
        """

        Given a predator/goal location one time-step ahead of the agent's location and the current location of the agent, return [whether innate behavior is triggered, direction to move]

        Checks innate behavior triggering if in a given direction of movement agent lands on the predicted location of the predator/goal

        """

        directions = np.arange(len(self.direction_space))

        while len(directions) > 0:

            dir, directions  = self.get_random_dir(directions)
            next_agent_loc = self.get_next_loc(agent_loc, dir)

            if (next_agent_loc == pred_loc.int())[0] and (next_agent_loc == pred_loc.int())[1]: # Naive Innate Behavior: move in opposite direction
                return True, -1*dir

            # elif (next_agent_loc == goal_loc.int())[0] and (next_agent_loc == goal_loc.int())[1]: 
                # return True, dir # Naive: move to position that puts you on goal


        return False, dir

    def plan_greedy(self, wm_pred, wm_goal, agent_loc, penalty = 0.1): # currently plans to a depth of one using a greedy method
                    #!TODO: make this a real learning model! multi-armed bandit sort of?

        """
        Planning module: choose greedy option using PWM predictions

        Inputs: current agent location, predicted predator location in one step, predicted goal location in one step

        Planning occurs first by looking at looking at world in next timestep: agent location in some direction, predator location in two steps, predicted goal location in two steps
            If scared in that particular state AND agent has innate behavior --> trigger innate behavior
                Currently, innate behavior is pouncing on prey or running opposite of direction of predator
                !TODO: resolve the weird resultant innate behavior triggered wrt pouncing on prey -- don't want to trigger "pouncing on prey" in the next time step while planning

            If that next step does NOT result in being in a scared state, then loop to find direction that places you closest to the prey and return that
                I.e. find direction that gets you the closest

        """

        directions = np.arange(len(self.direction_space))

        optimal_reward = np.inf

        # choose random directions to plan from; if resultant state triggers fear response, then quick-exit and return opposite direction

        while len(directions) > 0:

            dir, directions = self.get_random_dir(directions)
            next_agent_loc = self.get_next_loc(agent_loc, dir)

            next_pred_loc = self.predWM.forward(wm_pred) # use this for naive PWM
            next_goal_loc = self.goalWM.forward(wm_goal) # use this for naive PWM
            # next_pred_loc = round_prediction(self.predWM(encode(wm_pred))) # use this for NN PWM
            # next_goal_loc = round_prediction(self.goalWM(encode(wm_goal))) # use this for NN PWM

            triggered, innate_dir = self.get_innate_behavior(next_pred_loc, next_goal_loc, next_agent_loc)

            if triggered and self.innate:
                return innate_dir

            else: 

                dir_reward = np.linalg.norm(np.array(next_agent_loc)-np.array(wm_goal))
                if dir_reward < optimal_reward:

                    optimal_reward = dir_reward
                    optimal_dir    = dir

        return optimal_dir

    def simulate_plan(self, state):

      """
      
      Simulation Planning module: simulate results of the environment in all 8 directions and choose the one with the best survival

      """

      #!TODO: test if this works in other environments that aren't PreyPreatorEnv

      scare_space = np.zeros(len(self.direction_space))
      direction_rewards = np.zeros(len(self.direction_space))

      for i in range(len(self.direction_space)):

        reward, scared = self.simulate_forward(i, state) # simulate_forward uses DIR_TO_VEC

        scare_space[i] = scared
        direction_rewards[i] = reward

      return direction_rewards, scare_space

    def simulate_forward(self, act_dir, state):
        """
        Simulation only happens under a self-model scenario; otherwise, actions are undertaken directly by the agent itself via planning

        Function to simulate forward what would happen with a perfect world model and a specific directions

        Currently naive: 1 trial per scenario, absolutely not checking how frequent deaths are due to simplicity of environment
        """
        
        self.self_model.perfect_world_rep.change_agent_dir(act_dir)
        self.before_sim_obs = self.imag_act

        done = False
        scared = False
        
        while done is False:
            self.imag_act += 1
            
            _ = self.self_model.perfect_world_rep.time_update()
            self.self_model.perfect_world_rep.update_entity("agent")            
        
            reward, done, scared = self.check_selfmodel_done()

        self.self_model.set_state_by_state(state)
        
        self.imag_act = self.before_sim_obs
        return reward, scared
        
    # def update_policy(self, obs_sample):
    #     # print("***update")
    #     self.policy.update(obs_sample)
        
    def reached_goal(self):
        """
        Check whether agent reached prey
        """
        goal_self_check = (self.perfect_world_rep.agent_loc == self.perfect_world_rep.goal_loc)
        # print("Goal", goal_self_check)
        return goal_self_check

    def got_caught(self):
        """
        Check whether agent got caught by the predator
        """
        got_caught = (self.perfect_world_rep.agent_loc == self.perfect_world_rep.predator_loc)
        # print("Caught", got_caught)
        return got_caught

    def got_tired(self):   
        """
        Check whether agent got tire (took too many steps or perceived a lot)
        """
        too_many_steps = self.obs_received > 50
        # print("Tired", too_many_steps)
        return too_many_steps

    def selfmodel_reached_goal(self):
        """
        Check whether agent reached prey
        """
        goal_self_check = (self.self_model.perfect_world_rep.agent_loc == self.self_model.perfect_world_rep.goal_loc)
        # print("Goal", goal_self_check)
        return goal_self_check

    def selfmodel_got_caught(self):
        """
        Check whether agent got caught by the predator
        """
        got_caught = (self.self_model.perfect_world_rep.agent_loc == self.self_model.perfect_world_rep.predator_loc)
        # print("Caught", got_caught)
        return got_caught

    def selfmodel_got_tired(self):   
        """
        Check whether agent got tire (took too many steps or perceived a lot)
        """
        too_many_steps = self.imag_act > 50
        # print("Tired", too_many_steps)
        return too_many_steps


    def selfmodel_is_scared(self):
        dista = np.linalg.norm(np.array(self.self_model.perfect_world_rep.agent_loc)-np.array(self.self_model.perfect_world_rep.predator_loc))
        print("agent", self.self_model.perfect_world_rep.agent_loc)
        print("predator",self.self_model.perfect_world_rep.predator_loc)
        scared = False
        print("Dist", dista)
        if dista <= self.comfort_zone:
          scared = True
        print("scared",scared)
        return scared