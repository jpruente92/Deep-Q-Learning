import random
import sys
import time


import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from Neural_networks import QNetwork_pytorch


# experience replay and fixed Q targets (soft and hard update) are always active

# problem name
problem_name='LunarLander-v2'

# Hyperparameters
LOAD = False            # loading neural networks from file
PY_TORCH = True         # if true pytorch else tensorflow
SOFT_UPDATE = True      # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
UPDATE_TARGET_EVERY = 20 # how often to do the hard update for targets
DOUBLE_Q= False          # applying double q learning
PRIORITIZED_EXP_REPLAY= False          # applying prioritized experience replay
BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 0.0005             # learning rate
UPDATE_EVERY = 4        # how often to update the network

# network architecture
hidden_layers=[64,64]

# Statistics
number_episodes_til_solved=-1
running_time=-1
total_number_learn_calls=0


# use gpu if available else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# load environment from gym
env = gym.make(problem_name)
env.seed(0)
state_dimension =env.observation_space.shape
number_actions=env.action_space.n


print(state_dimension,number_actions)

# print statistics and all hyperparameters into a file
def print_stats_to_file(file_name):
    original_stdout = sys.stdout
    with open(file_name, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Running time:\t',running_time)
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


# helper functions

# compute dimension of state space if flattened into a single vector
def state_dim_to_int(state_dimension):
    return np.array(state_dimension).sum()


# class for the agent
class Agent():

    def __init__(self, state_shape, number_actions,hidden_layers,filename_local,filename_target, seed):
        state_dimension = state_dim_to_int(state_shape)
        self.state_dimension = state_dimension
        self.number_actions = number_actions
        self.hidden_layers = hidden_layers
        self.seed = random.seed(seed)
        self.max_priority=1

        # 2 networks: qnetwork_local is used for acting and qnetwork_target for learning the values in qnetwork_local
        # values of qnetwork_target are either softly or hard updated to match values of qnetwork_local
        if PY_TORCH:
            self.qnetwork_local = QNetwork_pytorch(state_dimension, number_actions,hidden_layers, seed,LR,LOAD,filename_local,device).to(device)
            self.qnetwork_target = QNetwork_pytorch(state_dimension, number_actions,hidden_layers, seed,LR,LOAD,filename_target,device).to(device)
        else:
            self.qnetwork_local = QNetwork_pytorch(state_dimension, number_actions,hidden_layers, seed,LR,LOAD,filename_local)
            self.qnetwork_target = QNetwork_pytorch(state_dimension, number_actions,hidden_layers, seed,LR,LOAD,filename_target)


        # Replay memory for sampling from former experiences
        self.memory = ReplayBuffer(number_actions, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    # doing one step in the environment: saving experience in replay buffer and learn
    def step(self, state, action, reward, next_state, done,A,B):
        # Save experience in replay memory with maximum priority
        self.memory.add(state, action, reward, next_state, done,self.max_priority)
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1)%(UPDATE_EVERY*UPDATE_TARGET_EVERY)
        if (self.t_step% UPDATE_EVERY) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences,probabilities = self.memory.sample(A)
                self.learn(experiences,probabilities, GAMMA,B)

    # returns the action following the epsilon-greedy policy
    def act(self, state, eps=0.):
        # if greedy action is selected
        if random.random() > eps:
            action_values = self.qnetwork_local.evaluate(state)
            # return best action
            return np.argmax(action_values.cpu().data.numpy())
        # if random action is selected
        else:
            # return random action
            return random.choice(np.arange(self.number_actions))



    def learn(self, samples,probabilities, gamma,B):
        global total_number_learn_calls
        total_number_learn_calls+=1
        states, actions, rewards, next_states, dones = self.samples_to_environment_values(samples)
        print("nextstates shape\t",next_states.shape)
        if(DOUBLE_Q):
            # compute action values with local network
            action_values = self.qnetwork_local.evaluate(next_states)
            # determine best action
            best_actions = action_values.argmax(1)
            # compute q targets with target network and choose best action
           # Q_targets_next=torch.tensor([self.qnetwork_target(next_state)[int(next_action)] for next_action,next_state in zip(best_actions,next_states)]).unsqueeze(1)
            Q_targets_next=np.array([self.qnetwork_target.evaluate(next_state)[int(next_action)] for next_action,next_state in zip(best_actions,next_states)])
        else:
            # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets_next = self.qnetwork_target.evaluate(next_states).max(1)[0]

        print("r\t",rewards.shape)
        print("Q_targets_next\t",Q_targets_next.shape)
        print("dones\t",dones.shape)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_expected = np.array([self.qnetwork_local.evaluate(state)[action] for action,state in zip(actions,states)])

        if PRIORITIZED_EXP_REPLAY:
            # update priorities
            td_errors = Q_targets-Q_expected
            for i, sample in enumerate(samples):
                # set priority of the given samples, small positive priority has to be guaranteed
                priority=float(max(abs(td_errors[i]),0.01))
                sample.priority=priority
                # update max priority
                self.max_priority=max(priority,self.max_priority)

            # multiply Q-targets with importance-sampling weight
            for i,prob in enumerate(probabilities):
                importance_sampling_weight = (1.0/len(self.memory)/probabilities[i])**B
                # print ("\tios",importance_sampling_weight)
                Q_targets[i]=Q_targets[i]*importance_sampling_weight

        self.qnetwork_local.train_nn(Q_expected,Q_targets)

        if SOFT_UPDATE:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        elif(self.t_step% UPDATE_TARGET_EVERY) == 0:
            self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


    def samples_to_environment_values(self,samples):
        # states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)
        states = np.array([e.state for e in samples if e is not None])
        actions = np.array([e.action for e in samples if e is not None])
        rewards = np.array([e.reward for e in samples if e is not None])
        next_states = np.array([e.next_state for e in samples if e is not None])
        dones = np.array([e.done for e in samples if e is not None])
        return states,actions,rewards,next_states,dones

class Experience:
    def __init__(self, state, action, reward, next_state,done,priority):
        self.state=state
        self.action=action
        self.reward = reward
        self.next_state=next_state
        self.done=done
        self.priority=priority


class ReplayBuffer:

    def __init__(self, number_actions, buffer_size, batch_size, seed):
        self.number_actions = number_actions
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done,priority):
        new_experience = Experience(state, action, reward, next_state, done,priority)
        self.memory.append(new_experience)

    def sample(self,A):
        if PRIORITIZED_EXP_REPLAY:
            priorities=[mem.priority**A for mem in self.memory]
            mysum= sum(priorities)
            probabilities=[p/mysum for p in priorities]
            draw = np.random.choice(len(probabilities), self.batch_size,replace=False,p=probabilities)
            samples=[self.memory[i] for i in draw]
            probabilities_of_samples=[probabilities[i] for i in draw]
        else:
            samples = random.sample(self.memory, k=self.batch_size)
            probabilities_of_samples=[]

        return samples, probabilities_of_samples

    def __len__(self):
        return len(self.memory)




def dqn(agent,n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,A_start=1.0, A_end=0.01, A_decay=0.995
        ,B_start=0.01, B_end=1.00, B_increase=1.005):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    A = A_start
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
            agent.step(state, action, reward, next_state, done,A,B)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon
        A = max(A_end, A_decay*A)           # decrease A
        B = min(B_end,B_increase*B)         # increase B
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
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


def start_agent(LOAD,filename,PLOT):
    if PY_TORCH:
        filename_local=problem_name+"_model_local.pth"
        filename_target=problem_name+"_model_target.pth"
    else:
        filename_local=problem_name+"_model_local.model"
        filename_target=problem_name+"_model_target.model"
    agent =Agent(state_shape=state_dimension, number_actions=number_actions,hidden_layers=hidden_layers,filename_local=filename_local,filename_target=filename_target, seed=0)
    start_time = time.time()
    scores = dqn(agent)
    print("Time for learning:",(time.time()-start_time))
    # save model
    if not LOAD:
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

start_agent(False,problem_name+"_PEXPR_",True)
