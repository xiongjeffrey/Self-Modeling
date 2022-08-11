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

env_num = 10
plan_percentages = np.linspace(0,1,10)

def act_with_policy(env, mlp_policy, obs):
    reward_table = []
    for i in range(100):
        action, _states = mlp_policy.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_table.append(rewards)
        env.render()
    
    deaths = reward_table.count(-1)/100
    wins = reward_table.count(1)/100
    return deaths, wins

planning_dict = {}
results = []
trials_num = 100
for plan_percent in plan_percentages:

    # determine if agent plans or acts with model-free policy 
    nums = np.random.choice([0, 1], size=trials_num, p=[1-plan_percent, plan_percent])
    all_env_results = {"death_percent":[],"win_percent":[]}

    for en in range(env_num):
        env = gym.make("gym_basic:pwm-v1")
        init_cond = env.all_envs[0]

        # declare and train policy ith this env
        mlp_policy = PPO("MlpPolicy", env, verbose=1)
        mlp_policy.learn(total_timesteps = 1000)

        pred_naivePWM = NaivePWM(init_state = init_cond[0], entity_dir = 5, grid_size = 5) #!TODO: figure out a better way to do this 
        goal_naivePWM = NaivePWM(init_state = init_cond[1], entity_dir = 6, grid_size = 5) #!TODO: figure out a better way to do this 

        brain = PlanningWithPWMAgent(8, True, policy = mlp_policy, predWM = pred_naivePWM, goalWM = goal_naivePWM)
        brain_input = (init_cond[0], init_cond[1], init_cond[2], 5)
        brain.perceive_state(brain_input)
    
        reward_table = []

        _ = env.reset()
        state = {'state': torch.tensor((init_cond[0], init_cond[1], init_cond[2]))} # initial state

        for trial in range(trials_num):
            
            done = False
            obs = state['state']
            steps = 0

            while not done: 

                max_dir = brain.take_action(nums[trial], state['state'])
                obs, reward, done, state = env.step(max_dir)

                steps += 1
                if steps > 20:
                    done = True
                    reward = 0
            
            reward_table.append(reward)

            _ = env.reset()
            state = {'state': torch.tensor((init_cond[0], init_cond[1], init_cond[2]))} # initial state

            # env.render()
        
        deaths = reward_table.count(-1)/trials_num
        wins = reward_table.count(1)/trials_num

        # deaths, wins = act_with_policy(env,mlp_policy)
        all_env_results["death_percent"].append(deaths)
        all_env_results["win_percent"].append(wins)


    alldeaths = sum(all_env_results["death_percent"])/env_num
    allwins = sum(all_env_results["win_percent"])/env_num
    results.append([plan_percent, alldeaths, allwins])

    planning_dict[plan_percent] = all_env_results

for plan, dict in planning_dict.items():
    print("Plan Percent:", plan)
    print(dict)

# with open('results/planning_policy1.csv', 'w') as f:
#     spamwriter = csv.writer(f, delimiter=',')
#     spamwriter.writerow(["plan_trials", "win_percent", "died_percent"])
#     for val in results:
#         spamwriter.writerow(val)

