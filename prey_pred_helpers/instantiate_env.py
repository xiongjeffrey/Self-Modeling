from prey_pred_helpers.prey_pred import PreyPreatorEnv
from itertools import product
from random import sample



def init_envs(num_env):
    all_envs = []
    env_count = 0
    win_count, dead_count = 0,0
    while env_count < num_env:
        init_cond = sample(list(product(range(5), repeat=2)), k=3)
        pass_test, winc, deadc = check_dies_wins(init_cond)
        win_count += winc
        dead_count += deadc
        if pass_test:
            all_envs.append(init_cond)
            env_count += 1

    return all_envs, win_count, dead_count
    


def check_dies_wins(init_cond):
    env = PreyPreatorEnv(init_cond[0], init_cond[1], init_cond[2], 5)
    possible_rewards = []
    dead_count = 0
    win_count = 0
    # calculate possible deaths and winnings
    all_dir_steps = []
    dir_dict = {}
    for dir in range(8):
        test_env = env.copy() #equivalent to resetting env
        test_env.change_agent_dir(dir)
        done = False
        steps = 0
        while done is False:
            steps += 1
            _ = test_env.time_update()
            test_env.update_entity("agent")
            reward = 0
            reward, done = env_reward(steps, test_env)
            possible_rewards.append(reward)
            
        all_dir_steps.append(steps)

        dir_dict[dir] = reward
    
    

    passes = False
    possible_death = False
    possible_win = False
    if 1 in dir_dict.values():
        win_count += 1
        possible_win = True
    
    if -1 in dir_dict.values():
        dead_count += 1
        possible_death = True

    passes = possible_death and possible_win
       
    return passes, win_count, dead_count
   


def env_reward(steps, env):
    reward = 0
    done = False
    if env.agent_won():
        reward = 1
        done = True
    elif env.agent_died():
        reward = -1
        done = True
    elif steps > 20:
        done = True
    return reward, done


