import numpy as np 
from env import PreyPreatorEnv


class SimplePlanningBrain():
    """
    A single perfect world representation 
    No innate behavior 
    params:
    - direction_space: int 8 possible actions (all directions)
    - use_planning: bool to indicate planning 
    """
    def __init__(self, direction_space, use_planning, policy):
        self.planning = use_planning
        self.direction_space = direction_space

        self.actions_taken = 0 # actions taken
        self.obs_received = 0 # observations perceived
        if self.planning:
            self.direction_rewards = np.zeros([self.direction_space]) # where predicted rewards will be stored
        self.policy = policy

        self.comfort_zone = 11
        
    def perceive_state(self, obs):
      """
      Function to recieve observations from the environment 
      obs - tuple containing the locations of goal, predator, self and grid_size
      """  
      goal_loc, predator_loc, self_location, grid_size = obs # unpack onto the location of entities
      # self.obs_received += 1 # observations perceived
      self.perfect_world_rep = PreyPreatorEnv(goal_loc, predator_loc, self_location, grid_size) # create world model instantiated with the new observation
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

      
      return reward, done

    def is_scared(self):
      dista = np.linalg.norm(np.array(self.perfect_world_rep.agent_loc)-np.array(self.perfect_world_rep.predator_loc))
      scared = False

      if dista <= self.comfort_zone:
        scared = True
      return scared
    
    def take_action(self, planning_trial, obs_state):
      """
      Function to take action (either through planning or model-free RL)
      """
      act = self.policy.act(obs_state)

      self.actions_taken += 1
      act_direction = 0
      self.direction_rewards = np.zeros([self.direction_space])
      if planning_trial:
        scared = self.plan() # call planning moddule to predict rewards for all actions
        act_direction = np.argmax(np.asarray(self.direction_rewards)) # select the direction with highest preicted reward
        if scared:
          act_direction = self.perfect_world_rep.predator_dir

      else:
        act_direction = act[0].item()
      
        if self.is_scared():
          act_direction = self.perfect_world_rep.predator_dir

      return act_direction, act[0]
    
    def plan(self):
        """
        Planning module, uses perfect world model representation to simulate actions 
        """
        for dir in range(self.direction_space):
            reward, scared = self.simulate_forward(dir)      
            self.direction_rewards[dir] = reward
        return scared

        
    def simulate_forward(self, act_dir):
        """
        Function to simulate forward what would happen with a perfect world model and a specific directions
        """
        self.before_sim_env = self.perfect_world_rep.copy() #equivalent to resetting env
        self.perfect_world_rep.change_agent_dir(act_dir)
        done = False
        scared = False
        self.before_sim_obs = self.obs_received
        while done is False:
            self.obs_received += 1
            
            return_items = self.perfect_world_rep.time_update()
            self.perfect_world_rep.update_entity("agent")
            # reward, done = self.perceive_state(return_items)
            
        
            reward, done = self.check_done()
            if self.is_scared():
              done = True
              scared = True

        self.perfect_world_rep = self.before_sim_env.copy()
        self.obs_received = self.before_sim_obs
        return reward, scared

    def update_policy(self, obs_sample):
        # print("***update")
        self.policy.update(obs_sample)
        


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

