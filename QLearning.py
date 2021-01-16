import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from Agents import Agent
from Profile import Profile
from Hyperparameter import *


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


def dqn(agent, env):
    stats = agent.profile
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = EPS_START  # initialize epsilon
    B = B_START
    for i_episode in range(1, NR_EPISODES + 1):
        state = env.reset()
        score = 0
        for t in range(MAX_NR_STEPS):
            # compute action
            action = agent.act(state, eps)
            # get information from environment
            next_state, reward, done, info = env.step(action)
            # save experience and train network
            agent.step(state, action, reward, next_state, done, B)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(EPS_END, EPS_DECAY * eps)  # decrease epsilon
        fraction = min(i_episode / NR_EPISODES, 1.0)
        B = B + fraction * (1.0 - B)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if WITH_PROFILING:
                profiling(stats)
        if np.mean(scores_window) >= VAL_ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores



def start_agent(env, seed):

    stats = Profile()
    # get information about states and actions
    state_dimension = env.get_state_dim()
    number_actions = env.get_nr_actions()

    filename_local = "Neural_networks/" + FILENAME_FOR_LOADING + "_model_local.pth"
    filename_target = "Neural_networks/" + FILENAME_FOR_LOADING + "_model_target.pth"
    agent = Agent(state_shape=state_dimension, number_actions=number_actions, filename_local=filename_local,
                  filename_target=filename_target, seed=seed, profile=stats)
    start_time = time.time()
    scores = dqn(agent=agent, env=env)
    print("Time for learning:", (time.time() - start_time))
    if SAVE:
        filename_local = FILENAME_FOR_SAVING + "_model_local.pth"
        filename_target = FILENAME_FOR_SAVING + "_model_target.pth"
        agent.qnetwork_local.save(filename_local)
        agent.qnetwork_target.save(filename_target)
    if PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(PLOTNAME)
        plt.show()
