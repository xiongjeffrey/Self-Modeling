from pstats import StatsProfile
import numpy as np 
from prey_pred_helpers.prey_pred import PreyPreatorEnv

import csv
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from pwm.pwmHelpers import round_prediction, precode, encode
from pwm.perfectWM import DeathNN, GoalNN


"""
Take WM and use it so that you can predict the next state and calculate the reward

From this, create table of all 7/8 possible rewards for each possible action and for each action you can use the WM which tells you final state of all action and you can get the reward for all directions

s_t = [(x,y) pred, (x, ) goal, (x, y) agent]
feed into agent 
agent --> planning --> pwm --> s_t+1 w/ reward for each possible action

no learning here, no networks, but get used to the process of using the working model for planning 
"""

# # Reading in states
# # states[t] := [[(x, y) pred], [(x, y) goal], [(x, y) agent]]
# # states[t] is the t-th row of pwmData.csv

def euclidean(x, y):
  sum = 0
  for i in range(len(x)):
    sum += (x[i] - y[i])**2

  return sum**0.5

# states = []
# with open('pwm\\pwmData.csv', 'r') as csvfile: # courtesy of https://stackoverflow.com/questions/13428318/reading-rows-from-a-csv-file-in-python

#     r = csv.reader(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
#     for _, line in enumerate(r):
#         states.append(np.array([[int(line[0][1]), int(line[0][4])], 
#         [int(line[1][1]), int(line[1][4])],
#         [int(line[2][1]), int(line[2][4])]]))

# load in nets
predNN = DeathNN()
pred_state_dict = torch.load('predNN.pt')
predNN.load_state_dict(pred_state_dict)

goalNN = GoalNN()
goal_state_dict = torch.load('goalNN.pt')
goalNN.load_state_dict(goal_state_dict)

class PlanningWithPWMAgent():

    """
    Plans using a perfect world model
    No innate behavior 
    params:
    - direction_space: int 8 possible actions (all directions)
    - use_planning: bool to indicate planning 
    """

    def __init__(self, direction_space, use_planning, policy, predWM = predNN, goalWM = goalNN, mode = 'imagination'):

        self.planning = use_planning
        self.direction_space = direction_space
        if direction_space == 8: #!TODO: find a less crude way to do this >:(

          self.direction_space = [torch.tensor((-1, -1)), torch.tensor((-1, 0)), torch.tensor((-1, 1)), torch.tensor((0, -1)), 
          torch.tensor((0, 1)), torch.tensor((1, -1)), torch.tensor((1, 0)), torch.tensor((1, 1))]

        self.actions_taken = 0 # actions taken
        self.imag_act = 0
        self.obs_received = 0 # observations perceived
        if self.planning:
            self.direction_rewards = np.zeros(len(self.direction_space)) # where predicted rewards will be stored

        self.policy = policy
        self.predWM = predWM
        self.goalWM = goalWM

        self.mode = mode

        self.comfort_zone = 11
        
    def perceive_state(self, obs):
      """
      Function to recieve observations from the environment 
      obs - tuple containing the locations of goal, predator, self and grid_size
      """  
      goal_loc, predator_loc, self_location, grid_size = obs # unpack onto the location of entities
      self.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size) # create world model instantiated with the new observation
      self.imaginary_world = self.perfect_world_rep.copy()
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

      if self.is_scared():
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

      # if self.selfmodel_is_scared():
      #   scared = True
      return reward, done

    def is_scared(self):
      dista = np.linalg.norm(np.array(self.perfect_world_rep.agent_loc)-np.array(self.perfect_world_rep.predator_loc))
      scared = False

      if dista <= self.comfort_zone:
        scared = True
      return scared
    
    def take_action(self, planning, state):
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

      if self.mode == 'imagination':
        scare_space, direction_rewards = self.plan(state[2], wm_pred, wm_goal) # call planning module to predict rewards for all actions

      elif self.mode == 'greedy':
        scare_space, direction_rewards = self.plan_greedy(state[2], wm_pred, wm_goal) # call planning module to predict rewards for all actions

      for i in range(len(direction_rewards)):
        
        scared = scare_space[i]
        dir = self.direction_space[i]

        if scared: # Naive: move in opposite direction
                    #!TODO: train agent to be learn an innate direction to run from fear
          return -1*dir

        else: # Naive: move to position that puts you on goal or else move in direction that places you closes to goal

          if (state[2] + dir == wm_goal.int())[0] and (state[2] + dir == wm_goal.int())[1]:
            return dir

          """
          else:
            direction_rewards[i] = 1/euclidean(dir + state[2], wm_goal)
            movement.append(dir)
          """

      # at this point, no movement will put you in a position where you are scared
      if planning: # naive movement
        return np.argmax(direction_rewards) # select the direction with highest predicted reward

      else: # naively move
        return self.policy.predict(state)

    def plan(self, agent_state, wm_pred, wm_goal):

      """
      
      Planning module: simulate results of the environment in all 8 directions and choose the one with the best survival

      """

      #!TODO: test if this works in other environments that aren't PreyPreatorEnv

      scare_space = np.zeros(len(self.direction_space))
      direction_rewards = np.zeros(len(self.direction_space))

      for i in range(len(self.direction_space)):

        dir = self.direction_space[i]
        reward, scared = self.simulate_forward(i) # simulate_forward uses DIR_TO_VEC

        scare_space[i] = scared
        direction_rewards[i] = reward

      return scare_space, direction_rewards

    
    def plan_greedy(self, agent_state, wm_pred, wm_goal, penalty = 0.1): # currently plans to a depth of one using a greedy method
                    #!TODO: make this a real learning model! multi-armed bandit sort of?

        """
        Planning module: choose greedy option using PWM predictions

        """

        scare_space = np.zeros(len(self.direction_space)) # array where corresponding direction is whether that movement is scary or not
        direction_rewards = np.zeros(len(self.direction_space))

        #!TODO: check if direction actually works like this
        # given all 8 possible movements, predict the reward from each of them
        for i in range(len(self.direction_space)): #!TODO: work with something that isn't class variable direction_rewards

            dir = self.direction_space[i]
            if (agent_state + dir == wm_pred.int())[0] and (agent_state + dir == wm_pred.int())[1]: # if movement puts you into death
                                                        # this if-else group has really messed up syntax, DON'T CHANGE IT
                direction_rewards[i] = -1
                scare_space[i] = True

            elif (agent_state + dir == wm_goal.int())[0] and (agent_state + dir == wm_goal.int())[1]: # if movement puts you in goal
                direction_rewards[i] = 1
                scare_space[i] = False

            else: # returns penalty to encourage quick discovery
                #!TODO: check with Fernanda to see if this is the proper method
                direction_rewards[i] = -1*penalty
                scare_space[i] = False

        return scare_space, direction_rewards # whether movement in any direction makes you scared
        
    def simulate_forward(self, act_dir):
        """
        Function to simulate forward what would happen with a perfect world model and a specific directions

        Currently naive: 1 trial per scenario, absolutely not checking how frequent deaths are due to simplicity of environment
        """
        self.before_sim_env = self.imaginary_world.copy() #equivalent to resetting env
        self.imaginary_world.change_agent_dir(act_dir)
        done = False
        self.before_sim_obs = self.imag_act
        scared = False
        
        while done is False:
            self.imag_act += 1
            
            return_items = self.imaginary_world.time_update()
            self.imaginary_world.update_entity("agent")            
        
            reward, done = self.check_selfmodel_done()

        self.imaginary_world = self.before_sim_env.copy()
        
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
        goal_self_check = (self.imaginary_world.agent_loc == self.imaginary_world.goal_loc)
        # print("Goal", goal_self_check)
        return goal_self_check

    def selfmodel_got_caught(self):
        """
        Check whether agent got caught by the predator
        """
        got_caught = (self.imaginary_world.agent_loc == self.imaginary_world.predator_loc)
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
        dista = np.linalg.norm(np.array(self.imaginary_world.agent_loc)-np.array(self.imaginary_world.predator_loc))
        print("agent", self.imaginary_world.agent_loc)
        print("predator",self.imaginary_world.predator_loc)
        scared = False
        print("Dist", dista)
        if dista <= self.comfort_zone:
          scared = True
        print("scared",scared)
        return scared

    


