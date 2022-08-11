import gym
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from prey_pred_helpers.instantiate_env import *
import numpy as np
from agents.policy_innate_selfmodel_agent import PlanningAgent
from agents.policy_innate_selfmodel_pwm_agent import PlanningWithPWMAgent
from agents.general_agent import Agent
import csv

import torch

import random

from pwm.naivePWM import NaivePWM

csv_destination = 'results\\fouragents_stepwise_onlyfear081122.csv'

env_num = 5
plan_percentages = np.linspace(0, 1, 10)
trials_num = 10

init_pred = np.arange(25)
init_goal = np.arange(25)
init_agent = np.arange(25)

trial_categories = ["no_planning_deaths", "no_planning_wins", "planning_no_innate_deaths", "planning_no_innate_wins", 
        "imagination_deaths", "imagination_wins", "self_model_deaths", "self_model_wins"]

def act_with_policy(env, mlp_policy, obs):
    reward_table = []
    for i in range(100):
        action, _states = mlp_policy.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_table.append(rewards)
        # env.render()
    
    deaths = reward_table.count(-1)/100
    wins = reward_table.count(1)/100
    return deaths, wins

with open(csv_destination, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    w.writerow(trial_categories)

# set up result tracker
all_env_results = {}
for category in trial_categories:
    all_env_results[category] = []

def step_trial(env, brain, planning, state, max_steps = 20):

    done = False
    steps = 0

    while not done:

        max_dir = brain.take_action(planning, state)

        if isinstance(max_dir, np.int32): # no clue why but this is needed
            max_dir = int(max_dir)

        _, reward, done, state = env.step(max_dir)
        state = state['state']

        steps += 1
        if steps > 20:
            done = True
            reward = 0

    return reward

def driver(initial_state, mode = 'stepwise'):
    
    with open(csv_destination, 'a', newline = '') as csvfile:
        w = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        w.writerow(['Initial_State', init_state[0], init_state[1], init_state[2]])

    for plan_percent in plan_percentages:

        # determine if agent plans or acts with model-free policy 
        nums = np.random.choice([0, 1], size=trials_num, p=[1-plan_percent, plan_percent])

        # set up result tracker
        all_env_results = {}
        for category in trial_categories:
            all_env_results[category] = []

        with open(csv_destination, 'a', newline = '') as csvfile:
            w = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            w.writerow(['Plan_Percent', plan_percent])

        for _ in range(env_num):

            if mode == 'stepwise': 
                env = gym.make("gym_basic:pwm-v1")

            elif mode == 'basic': 
                env = gym.make("gym_basic:pwm-v0")

            init_cond = env.all_envs[0]

            # declare and train policy ith this env
            mlp_policy = PPO("MlpPolicy", env, verbose=1)
            mlp_policy.learn(total_timesteps = 1000)
            pred_naivePWM = NaivePWM(init_state = init_cond[0], entity_dir = 5, grid_size = 5) #!TODO: figure out a better way to do this 
            goal_naivePWM = NaivePWM(init_state = init_cond[1], entity_dir = 6, grid_size = 5) #!TODO: figure out a better way to do this 

            # setting up the agents
            # Agent params goes [planning, innate, self-modeling]
            brain_no_planning = Agent(8, [False, True, False], policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM)
            brain_planning_no_innate = Agent(8, [True, False, False], policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM)
            brain_imagine = Agent(8, [True, True, False], policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM)
            brain_self_model  = Agent(8, [True, True, True], policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM)

            brains = [brain_no_planning, brain_planning_no_innate, brain_imagine, brain_self_model]

            brain_input = (init_cond[0], init_cond[1], init_cond[2], 5)
            for brain in brains:
                brain.perceive_state(brain_input)
        
            reward_table_no_planning = []
            reward_planning_no_innate = []
            reward_table_imagination = []
            reward_table_self_model = []

            reward_tables = [reward_table_no_planning, reward_planning_no_innate, reward_table_imagination, reward_table_self_model]
            
            _ = env.reset()
            _ = env.set_grid(initial_state)

            if mode == 'stepwise':

                for trial in range(trials_num):

                    for i in range(len(brains)): 
                        
                        # brains[i] is the i'th brain to be tested
                        # append information in reward_tables[i]

                        reward = step_trial(env, brains[i], nums[trial], initial_state)
                        reward_tables[i].append(reward)

                        _ = env.reset()
                        _ = env.set_grid(initial_state)

            elif mode == 'basic':

                for trial in range(trials_num):

                    for i in range(len(brains)): 
                        
                        # brains[i] is the i'th brain to be tested
                        # append information in reward_tables[i]

                        max_dir = brains[i].take_action(nums[trial], initial_state)
                        _, reward, _, _ = env.step(max_dir)      
                        reward_tables[i].append(reward)

                        _ = env.reset()
                        _ = env.set_grid(initial_state)
            
            # record deaths and wins for this particular set of initial conditions
            j = 0
            for key in all_env_results:

                idx = (j - (j%2))//2

                if j % 2 == 0: # death tracker
                    all_env_results[key].append(reward_tables[idx].count(-1)/trials_num)
                else:
                    all_env_results[key].append(reward_tables[idx].count(1)/trials_num)

                j += 1

            with open(csv_destination, 'a', newline = '') as csvfile:
                w = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

                row = [0]*len(all_env_results)

                j = 0
                for key in all_env_results:
                    row[j] = all_env_results[key][-1]
                    j += 1
                
                w.writerow(row)

        # record overall data for plan_percent level
        with open(csv_destination, 'a', newline = '') as csvfile:
            w = csv.writer(csvfile, delimiter = ',' , quotechar = '|', quoting = csv.QUOTE_MINIMAL)

            w.writerow(['Results_at_Plan_Percent', plan_percent])

            j = 0
            row = [0]*len(all_env_results)

            for key in all_env_results:
                
                row[j] = sum(all_env_results[key])/env_num
                j += 1

            w.writerow(row)

# calling driving function across all possible combinations of states
def decode_loc(n: int, grid_size = 5): # takes integer and turns it into a coordinate tensor
    x = n // grid_size
    y = n % 5

    return (x, y)

for pred_loc in init_pred:

    for goal_loc in init_goal:

        if pred_loc == goal_loc: continue

        else:
            for agent_loc in init_agent:

                if agent_loc == goal_loc or agent_loc == pred_loc: continue

                else:

                    # Below is iterative approach

                    # pred = decode_loc(pred_loc)
                    # goal = decode_loc(goal_loc)
                    # agent = decode_loc(agent_loc)

                    # Below is random seed approach

                    pred = decode_loc(random.randrange(5))
                    goal = decode_loc(random.randrange(5))
                    agent = decode_loc(random.randrange(5))

                    init_state = torch.tensor((pred, goal, agent))

                    print('State')
                    print(init_state)

                    driver(init_state, mode = 'stepwise')