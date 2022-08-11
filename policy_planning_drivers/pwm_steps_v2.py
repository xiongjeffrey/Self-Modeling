import gym
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from prey_pred_helpers.instantiate_env import *
import numpy as np
from agents.policy_innate_selfmodel_agent import PlanningAgent
from agents.policy_innate_selfmodel_pwm_agent import PlanningWithPWMAgent
import csv

import torch

from pwm.naivePWM import NaivePWM

csv_destination = 'results\\imagination_greedy_comparison_stepwise072722.csv'

env_num = 5
plan_percentages = np.linspace(0, 1, 5)

init_pred = np.arange(25)
init_goal = np.arange(25)
init_agent = np.arange(25)

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
    w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    w.writerow(['Imagination_Deaths', 'Imagination_Wins', 'Greedy_Deaths', 'Greedy_Wins', 'Policy_Deaths', 'Policy_Wins'])

planning_dict = {}
results_by_planpercent = {"imagination_deaths": [], "imagination_wins": [], "greedy_deaths": [], "greedy_wins": [], "policy_deaths": [], "policy_wins": []}

trials_num = 100

def step_trial(env, brain, planning, state, max_steps = 20):

    done = False
    steps = 0

    while not done:

        max_dir = brain.take_action(planning, state)
        _, reward, done, state = env.step(max_dir)
        state = state['state']

        steps += 1
        if steps > 20:
            done = True
            reward = 0

    return reward

def driver(initial_state):
    
    with open(csv_destination, 'a', newline = '') as csvfile:
        w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        w.writerow(['Initial_State', init_state[0], init_state[1], init_state[2]])

    for plan_percent in plan_percentages:

        # determine if agent plans or acts with model-free policy 
        nums = np.random.choice([0, 1], size=trials_num, p=[1-plan_percent, plan_percent])
        all_env_results = {"imagination_deaths": [], "imagination_wins": [], "greedy_deaths": [], "greedy_wins": [], "policy_deaths": [], "policy_wins": []}

        with open(csv_destination, 'a', newline = '') as csvfile:
            w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            w.writerow(['Plan_Percent', plan_percent])

        for _ in range(env_num):

            env = gym.make("gym_basic:pwm-v1")
            init_cond = env.all_envs[0]

            # declare and train policy ith this env
            mlp_policy = PPO("MlpPolicy", env, verbose=1)
            mlp_policy.learn(total_timesteps = 1000)
            pred_naivePWM = NaivePWM(init_state = init_cond[0], entity_dir = 5, grid_size = 5) #!TODO: figure out a better way to do this 
            goal_naivePWM = NaivePWM(init_state = init_cond[1], entity_dir = 6, grid_size = 5) #!TODO: figure out a better way to do this 

            # setting up the agents
            brain_imagine = PlanningWithPWMAgent(8, True, policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM, mode = 'imagination')
            brain_greedy  = PlanningWithPWMAgent(8, True, policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM, mode = 'greedy')

            brain_input = (init_cond[0], init_cond[1], init_cond[2], 5)
            brain_imagine.perceive_state(brain_input)
            brain_greedy.perceive_state(brain_input)
        
            reward_table_imagination = []
            reward_table_greedy = []

            _ = env.reset()
            _ = env.set_grid(initial_state)

            for trial in range(trials_num):

                reward = step_trial(env, brain_imagine, nums[trial], initial_state)
                reward_table_imagination.append(reward)

                _ = env.reset()
                _ = env.set_grid(initial_state)

                reward = step_trial(env, brain_greedy, nums[trial], initial_state)
                reward_table_greedy.append(reward)

                _ = env.reset()
                _ = env.set_grid(initial_state)
            
            # record deaths and wins for this particular set of initial conditions 
            imagination_deaths = reward_table_imagination.count(-1)/trials_num
            imagination_wins = reward_table_imagination.count(1)/trials_num

            greedy_deaths = reward_table_greedy.count(-1)/trials_num
            greedy_wins = reward_table_greedy.count(1)/trials_num

            policy_deaths, policy_wins = act_with_policy(env, mlp_policy, initial_state)

            all_env_results["imagination_deaths"].append(imagination_deaths)
            all_env_results["imagination_wins"].append(imagination_wins)
            all_env_results["greedy_deaths"].append(greedy_deaths)
            all_env_results["greedy_wins"].append(greedy_wins)
            all_env_results["policy_deaths"].append(policy_deaths)
            all_env_results["policy_wins"].append(policy_wins)

            with open(csv_destination, 'a', newline = '') as csvfile:
                w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
                w.writerow([all_env_results["imagination_deaths"][-1], all_env_results["imagination_wins"][-1], all_env_results["greedy_deaths"][-1], 
                    all_env_results["greedy_wins"][-1], all_env_results["policy_deaths"][-1], all_env_results["policy_wins"][-1]])


        imagination_deaths_by_planpercent = sum(all_env_results["imagination_deaths"])/env_num
        imagination_wins_by_planpercent = sum(all_env_results["imagination_wins"])/env_num
        greedy_deaths_by_planpercent = sum(all_env_results["greedy_deaths"])/env_num
        greedy_wins_by_planpercent = sum(all_env_results["greedy_wins"])/env_num
        policy_deaths_by_planpercent = sum(all_env_results["policy_deaths"])/env_num
        policy_wins_by_planpercent = sum(all_env_results["policy_wins"])/env_num

        with open(csv_destination, 'a', newline = '') as csvfile:
            w = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            w.writerow(['Results_at_Plan_Percent', plan_percent])
            w.writerow([imagination_deaths_by_planpercent, imagination_wins_by_planpercent, greedy_deaths_by_planpercent, 
                greedy_wins_by_planpercent, policy_deaths_by_planpercent, policy_wins_by_planpercent])

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

                    pred = decode_loc(pred_loc)
                    goal = decode_loc(goal_loc)
                    agent = decode_loc(agent_loc)

                    init_state = torch.tensor((pred, goal, agent))

                    print('State')
                    print(init_state)

                    driver(init_state)