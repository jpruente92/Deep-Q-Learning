import random
import sys
import time

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch


from Agents import Agent_py_torch
from Profile import Profile


PY_TORCH=True






# network architecture
hidden_layers=[64,64]






# print statistics and all hyperparameters into a file
def print_stats_to_file(file_name):
    original_stdout = sys.stdout
    with open(file_name, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('\ri {}\tj {}'.format(i, j), end="")
        print('Running time:\t', total_running_time)
        print('# episodes til solved:\t',number_episodes_til_solved)
        print('SOFT_UPDATE:\t',SOFT_UPDATE)
        print('UPDATE_TARGET_EVERY:\t',UPDATE_TARGET_EVERY)
        print('DOUBLE_Q:\t',DOUBLE_Q)
        print('PRIORITIZED_EXP_REPLAY:\t',PRIORITIZED_EXP_REPLAY)
        print('BUFFER_SIZE:\t',BUFFER_SIZE)
        print('BATCH_SIZE:\t',BATCH_SIZE)
        print('GAMMA:\t',GAMMA)
        print('TAU:\t',TAU)
        print('LR:\t',LR)
        print('UPDATE_EVERY:\t',UPDATE_EVERY)
        print("\n\n")
    sys.stdout = original_stdout # Reset the standard output to its original value


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





def dqn_gym(agent,n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
        ,B_start=0.4, B_end=1.00, B_increase=1.005,with_profile=False):
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
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            number_episodes_til_solved=i_episode-100
            break
    return scores

def dqn_unity(agent,brain_name,env,n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
        ,B_start=0.4, B_end=1.00, B_increase=1.005,with_profile=False,train=True):
    stats = agent.profile
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    B= B_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
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
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon
        B = min(B_end,B_increase*B)         # increase B
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if with_profile:
                profiling(stats)
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            number_episodes_til_solved=i_episode-100
            break

    return scores

def test_parameters(SOFT_UPDATE_values,UPDATE_TARGET_EVERY_values,DOUBLE_Q_values,PRIORITIZED_EXP_REPLAY_values,
                    BUFFER_SIZE_values,BATCH_SIZE_values,GAMMA_values,TAU_values,LR_values,UPDATE_EVERY_values,):
    global env,start_time,running_timeLOAD,SOFT_UPDATE,UPDATE_TARGET_EVERY,DOUBLE_Q,PRIORITIZED_EXP_REPLAY,BUFFER_SIZE,\
        BATCH_SIZE,GAMMA,TAU,LR,UPDATE_EVERY
    for s in SOFT_UPDATE_values:
        SOFT_UPDATE=s
        for ute in UPDATE_TARGET_EVERY_values:
            UPDATE_TARGET_EVERY=ute
            for dq in DOUBLE_Q_values:
                DOUBLE_Q=dq
                for per in PRIORITIZED_EXP_REPLAY_values:
                    PRIORITIZED_EXP_REPLAY=per
                    for bf in BUFFER_SIZE_values:
                        BUFFER_SIZE=bf
                        for bs in BATCH_SIZE_values:
                            BATCH_SIZE=bs
                            for g in GAMMA_values:
                                GAMMA=g
                                for t in TAU_values:
                                    TAU=t
                                    for lr in LR_values:
                                        LR=lr
                                        for ue in UPDATE_EVERY_values:
                                            UPDATE_EVERY=ue
                                            agent =Agent(state_shape=state_dimension, number_actions=number_actions,
                                                         hidden_layers=hidden_layers,state_dict_local=None,state_dict_target=None, seed=0)
                                            start_time = time.time()
                                            dqn(agent)
                                            running_time = time.time()-start_time
                                            print_stats_to_file("Parameters_test.txt")
                                            env.reset()


def start_agent_gym(problem_name,PLOT,profiling):
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


def start_agent_unity(problem_name,PLOT,profiling,train):
    stats = Profile()
    env = UnityEnvironment(file_name="/"+problem_name+".exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # get information about states and actions
    env_info = env.reset(train_mode=True)[brain_name]
    state_dimension =len(env_info.vector_observations[0])
    number_actions=brain.vector_action_space_size

    filename_local=problem_name+"_model_local.pth"
    filename_target=problem_name+"_model_target.pth"
    agent =Agent_py_torch(state_shape=state_dimension, number_actions=number_actions, hidden_layers=hidden_layers,filename_local=filename_local, filename_target=filename_target, seed=0, profile=stats)
    start_time = time.time()
    scores = dqn_unity(agent=agent,brain_name=brain_name,env=env,with_profile=profiling,train=train)
    env.close()
    print("Time for learning:",(time.time()-start_time))
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

