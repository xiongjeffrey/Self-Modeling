import gym
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from prey_pred_helpers.instantiate_env import *
import numpy as np
from agents.policy_innate_selfmodel_agent import PlanningAgent
from agents.policy_innate_selfmodel_pwm_agent import PlanningWithPWMAgent
import csv

env_num = 1
plan_percentages = np.linspace(0,1,10)

def act_with_policy(env, mlp_policy,obs):
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
trials_num = 1000
for plan_percent in plan_percentages:

    # determine if agent plans or acts with model-free policy 
    nums = np.random.choice([0, 1], size=trials_num, p=[1-plan_percent, plan_percent])
    all_env_results = {"death_percent":[],"win_percent":[]}

    for en in range(env_num):
        env = gym.make("gym_basic:basic-v0")
        init_cond = env.all_envs[0]

        # declare and train policy ith this env
        mlp_policy = PPO("MlpPolicy", env, verbose=1)
        mlp_policy.learn(total_timesteps=1)

        brain = PlanningWithPWMAgent(8, True, mlp_policy)
        brain_input = (init_cond[0], init_cond[1], init_cond[2], 5)
        brain.perceive_state(brain_input)
    
        reward_table = []
        obs = env.reset()

        for trial in range(trials_num):
            max_dir, act = brain.take_action(True, obs)
            obs, rewards, dones, info = env.step(max_dir)
            reward_table.append(rewards)
            env.render()

        
        deaths = reward_table.count(-1)+reward_table.count(-1)/trials_num
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



with open('results/planning_policy1.csv', 'w') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(["plan_trials", "win_percent", "died_percent"])
    for val in results:
        spamwriter.writerow(val)

