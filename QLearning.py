import sys
import time
from collections import deque
import copy


import numpy as np
import matplotlib.pyplot as plt
import torch

from Agents import Agent
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
    eps = param.EPS_START                    # initialize epsilon
    B= param.B_START
    for i_episode in range(1, param.NR_EPISODES+1):
        state = env.reset()
        score = 0
        for t in range(param.MAX_NR_STEPS):
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


def start_agent_unity(problem_name,param,seed):
    from unityagents import UnityEnvironment
    stats = Profile()
    env_name="./"+problem_name
    env = UnityEnvironment(file_name=env_name)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # get information about states and actions
    env_info = env.reset(train_mode=param.ENV_TRAIN)[brain_name]
    state_dimension =len(env_info.vector_observations[0])
    number_actions=brain.vector_action_space_size


    filename_local=param.FILENAME_FOR_LOADING+"_model_local.pth"
    filename_target=param.FILENAME_FOR_LOADING+"_model_target.pth"
    agent =Agent(state_shape=state_dimension, number_actions=number_actions, filename_local=filename_local, filename_target=filename_target, seed=seed, profile=stats, param=param)
    start_time = time.time()
    scores = dqn_unity(agent=agent,brain_name=brain_name,env=env,param=param)
    env.close()
    print("Time for learning:",(time.time()-start_time))
    if param.SAVE:
        filename_local=param.FILENAME_FOR_SAVING+"_model_local.pth"
        filename_target=param.FILENAME_FOR_SAVING+"_model_target.pth"
        agent.qnetwork_local.save(filename_local)
        agent.qnetwork_target.save(filename_target)
    if param.PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(param.PLOTNAME)
        plt.show()

def start_agent_gym(problem_name,param,seed):
    import gym
    stats = Profile()
    env = gym.make(problem_name)
    env.seed(seed)
    # get information about states and actions
    state_dimension =env.observation_space.shape
    number_actions=env.action_space.n

    filename_local=param.FILENAME_FOR_LOADING+"_model_local.pth"
    filename_target=param.FILENAME_FOR_LOADING+"_model_target.pth"
    agent =Agent(state_shape=state_dimension, number_actions=number_actions, filename_local=filename_local, filename_target=filename_target, seed=seed, profile=stats, param=param)
    start_time = time.time()
    scores = dqn_gym(agent=agent,env=env,param=param)
    env.close()
    print("Time for learning:",(time.time()-start_time))
    if param.SAVE:
        filename_local=param.FILENAME_FOR_SAVING+"_model_local.pth"
        filename_target=param.FILENAME_FOR_SAVING+"_model_target.pth"
        agent.qnetwork_local.save(filename_local)
        agent.qnetwork_target.save(filename_target)
    if param.PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(param.PLOTNAME)
        plt.show()

