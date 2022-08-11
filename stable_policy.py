import gym
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from prey_pred_helpers.instantiate_env import *
import numpy as np
from agents.policy_innate_selfmodel_agent import PlanningAgent
from agents.policy_innate_selfmodel_pwm_agent import PlanningWithPWMAgent


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

def act_with_planning(env, obs):
    reward_table = []
    for i in range(100):
        brain.take_action
        max_dir, act = brain.take_action(1,obs)
        obs, rewards, dones, info = env.step(max_dir)
        reward_table.append(rewards)
        env.render()
    
    deaths = reward_table.count(-1)/100
    wins = reward_table.count(1)/100
    return deaths, wins

env_num = 10

all_env_results = {"death_percent":[],"win_percent":[]}

# wipe the csv
f = open('pwm\\pwmData.csv', "w+") # wiping the file
f.close()

for i in range(env_num):

    # nums = np.random.choice([0, 1], size=100, p=[1-plan_percentages[1], plan_percentages[1]])
    env = gym.make("gym_basic:basic-v0")





    # declare and train policy ith this env
    mlp_policy = PPO("MlpPolicy", env, verbose=1)
    mlp_policy.learn(total_timesteps=3000)

    obs = env.reset()

   
    deaths, wins = act_with_policy(env,mlp_policy, obs)
    all_env_results["death_percent"].append(deaths)
    all_env_results["win_percent"].append(wins)


   
print(all_env_results)



