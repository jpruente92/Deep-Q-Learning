import random
import sys
import time
from collections import deque
import copy


from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch

from Agents import Agent_py_torch
from Profile import Profile
from Hyperparameter import Hyperparameters



def profiling(profile):
    print("total_number_learn_calls:\t", profile.total_number_learn_calls)
    print("total_number_sampling_calls:\t", profile.total_number_sampling_calls)
    print("total_number_evaluate_calls:\t", profile.total_number_evaluate_calls)
    print("total_time_sampling:\t", profile.total_time_sampling)
    print("total_time_learning:\t", profile.total_time_learning)
    print("\ttotal_time_evaluation:\t", profile.total_time_evaluation)
    print("\ttotal_time_training:\t", profile.total_time_training)
    print("\ttotal_time_soft_update:\t", profile.total_time_soft_update)
    print("\ttotal_time_samples_to_environment_values:\t", profile.total_time_samples_to_environment_values)
    print("\ttotal_time_updating_priorities:\t", profile.total_time_updating_priorities)
    print("\ttotal_time_introducing_isw:\t", profile.total_time_introducing_isw)
    print()





def dqn_gym(agent,env,param):
    stats = agent.profile
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    B= B_start
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # compute action
            action = agent.act(state, eps)
            # get information from environment
            next_state, reward, done, info = env.step(action)
            # save experience and train network
            agent.step(state, action, reward, next_state, done,B)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon
        B = min(B_end,B_increase*B)         # increase B
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if with_profile:
                profiling(stats)
        if np.mean(scores_window)>=param.VAL_ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            number_episodes_til_solved=i_episode-100
            break
    return scores

def dqn_unity(agent,brain_name,env,param):
    stats = agent.profile
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = param.EPS_START                    # initialize epsilon
    B= param.B_START
    for i_episode in range(1, param.NR_EPISODES+1):
        env_info = env.reset(train_mode=param.ENV_TRAIN)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(param.MAX_NR_STEPS):
            # compute action
            action = agent.act(state, eps)
            # get information from environment
            env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            # save experience and train network
            agent.step(state, action, reward, next_state, done,B)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        eps = max(param.EPS_END, param.EPS_DECAY*eps)   # decrease epsilon
        B = min(param.B_END,param.B_INCREASE*B)         # increase B
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if param.WITH_PROFILING:
                profiling(stats)
        if np.mean(scores_window)>=param.VAL_ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            number_episodes_til_solved=i_episode-100
            break

    return scores





def test_parameters(file_name,problem_name,unity,param,stats,list_of_parameters_to_print,list_of_parameters,list_of_values):
    if(len(list_of_parameters)<1):
        original_stdout = sys.stdout
        with open(file_name, 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # do the computation
            if unity:
                start_agent_unity(problem_name,param)
            else:
                start_agent_gym(problem_name,param)
            print('####################################################################')
            print('Results:')
            print('Running time:\t', stats.total_running_time)
            print('# episodes til solved:\t',stats.number_episodes_til_solved)
            print('--------------------------------------------------------------------')
            print('Parameters:')
            for f in list_of_parameters_to_print:
                print("\t",param.namelist[f],":",param.parameterlist[f])
            print('####################################################################')
            print("\n\n")
        sys.stdout=original_stdout# Reset the standard output to its original value
        stats=Profile()
    else:
        recursive_list_of_parameters = copy.deepcopy(list_of_parameters)
        del recursive_list_of_parameters[0]
        recursive_list_of_values = copy.deepcopy(list_of_values)
        del recursive_list_of_values[0]
        for val in list_of_values[0]:
            param.parameterlist[list_of_parameters[0]]=val
            test_parameters(file_name,problem_name,unity,param,stats,list_of_parameters_to_print,recursive_list_of_parameters,recursive_list_of_values)







def start_agent_gym(problem_name):
    # load environment from gym
    env = gym.make(problem_name)
    env.seed(0)
    state_dimension =env.observation_space.shape
    number_actions=env.action_space.n
    stats = Profile()
    if PY_TORCH:
        filename_local=problem_name+"_model_local.pth"
        filename_target=problem_name+"_model_target.pth"
        agent =Agent_py_torch(state_shape=state_dimension, number_actions=number_actions, hidden_layers=hidden_layers,
                              filename_local=filename_local, filename_target=filename_target, seed=0, profile=stats)
    else:
        filename_local=problem_name+"_model_local.model"
        filename_target=problem_name+"_model_target.model"
        agent =Agent_tensorflow(state_shape=state_dimension, number_actions=number_actions,hidden_layers=hidden_layers,
                              filename_local=filename_local,filename_target=filename_target, seed=0,stats=stats)
    start_time = time.time()
    scores = dqn(agent,profiling)
    print("Time for learning:",(time.time()-start_time))
    # save model
    agent.qnetwork_local.save(filename_local)
    agent.qnetwork_target.save(filename_target)
    if PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()


def start_agent_unity(problem_name,param):
    stats = Profile()
    env_name="./"+problem_name+".exe"
    env_name="C:/Users/Jonas/Desktop/Reinforcement Learning Class/deep-reinforcement-learning/p1_navigation/Banana_Windows_x86_64/Banana.exe"
    env = UnityEnvironment(file_name=env_name)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # get information about states and actions
    env_info = env.reset(train_mode=param.ENV_TRAIN)[brain_name]
    state_dimension =len(env_info.vector_observations[0])
    number_actions=brain.vector_action_space_size

    filename_local=problem_name+"_model_local.pth"
    filename_target=problem_name+"_model_target.pth"
    agent =Agent_py_torch(state_shape=state_dimension, number_actions=number_actions,filename_local=filename_local, filename_target=filename_target, seed=0, profile=stats,param=param)
    start_time = time.time()
    scores = dqn_unity(agent=agent,brain_name=brain_name,env=env,param=param)
    env.close()
    print("Time for learning:",(time.time()-start_time))
    if param.SAVE:
        agent.qnetwork_local.save(filename_local)
        agent.qnetwork_target.save(filename_target)
    if param.PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

