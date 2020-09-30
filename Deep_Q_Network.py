import sys
import time
from builtins import str

import numpy as np
import random
from collections import namedtuple, deque

import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


# experience replay and fixed Q targets (soft and hard update) are always active

# Hyperparameters
SOFT_UPDATE = False      # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
UPDATE_TARGET_EVERY = 20 # how often to do the hard update for targets
DOUBLE_Q= False          # applying double q learning
PRIORITIZED_EXP_REPLAY= False          # applying prioritized experience replay
BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 0.0005             # learning rate
UPDATE_EVERY = 4        # how often to update the network

# Statistics
number_episodes_til_solved=-1
running_time=-1


# use gpu if available else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
#
# changable code
problem_name='LunarLander-v2'
hidden_layers=[64,64]
#
#
#

# load environment from gym
env = gym.make(problem_name)
env.seed(0)
state_dimension =env.observation_space.shape
number_actions=env.action_space.n


print(state_dimension,number_actions)


def print_stats_to_file(file_name):
    original_stdout = sys.stdout
    with open(file_name, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Running time:\t',running_time)
        print('# episodes til solved:\t',number_episodes_til_solved)
        sys.stdout = original_stdout # Reset the standard output to its original value



def state_dim_to_int(state_dimension):
    return np.array(state_dimension).sum()

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dimension, number_actions, hidden_layers, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_dimension, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], number_actions)

    def forward(self, state):
        x=state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        return self.output(x)

class Agent():

    def __init__(self, state_shape, number_actions,hidden_layers,state_dict_local,state_dict_target, seed):
        state_dimension = state_dim_to_int(state_shape)
        self.state_dimension = state_dimension
        self.number_actions = number_actions
        self.hidden_layers = hidden_layers
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_dimension, number_actions,hidden_layers, seed).to(device)
        self.qnetwork_target = QNetwork(state_dimension, number_actions,hidden_layers, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # load weights and biases
        if state_dict_local is not None:
            self.qnetwork_local.load_state_dict(state_dict_local)
            self.qnetwork_target.load_state_dict(state_dict_target)
            print("Values loaded")

        # Replay memory
        self.memory = ReplayBuffer(number_actions, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1)%(UPDATE_EVERY*UPDATE_TARGET_EVERY)
        if (self.t_step% UPDATE_EVERY) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set evaluation mode
            self.qnetwork_local.eval()
            # for evaluation no grad bc its faster
            with torch.no_grad():
                # compute action values
                action_values = self.qnetwork_local(state)
            # set training mode
            self.qnetwork_local.train()
            # return best action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.number_actions))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        if(DOUBLE_Q):
            # set evaluation mode
            self.qnetwork_local.eval()
            with torch.no_grad():
                # compute action values with local network
                action_values = self.qnetwork_local(next_states)
                # determine best action
                best_actions = action_values.argmax(1)
                # compute q targets with target network and choose best action
                Q_targets_next=torch.tensor([self.qnetwork_target(next_state)[int(next_action)] for next_action,next_state in zip(best_actions,next_states)]).unsqueeze(1)
                # print(action_values[0],best_actions[0],"\t",self.qnetwork_target(next_states[0]),Q_targets_next[0])
            # set training mode
            self.qnetwork_local.train()
            # # Get the best action for next states from the local network
            # best_actions = [self.qnetwork_local.forward(next_state).detach().argmax() for next_state in next_states]
            # # Put the next state in target network and choose the previously computed best action (Double Q Learning)
            # Q_targets_next=torch.tensor([self.qnetwork_target(next_state).detach()[int(next_action)] for next_action,next_state in zip(best_actions,next_states)]).unsqueeze(1)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Train the local network
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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

class memory:
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)




def dqn(agent,n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # compute action
            action = agent.act(state, eps)
            # get indormation from environment
            next_state, reward, done, info = env.step(action)
            # save experience and train network
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
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
    agent =Agent(state_shape=state_dimension, number_actions=number_actions,hidden_layers=hidden_layers,state_dict_local=None,state_dict_target=None, seed=0)
    if LOAD:
        state_dict_local = torch.load(filename+"_qnetwork_local.pth")
        state_dict_target = torch.load(filename+"_qnetwork_target.pth")
        agent = Agent(state_shape=state_dimension, number_actions=number_actions,hidden_layers=hidden_layers,state_dict_local=state_dict_local, state_dict_target=state_dict_target, seed=0)
    start_time = time.time()
    scores = dqn(agent)
    print("Time for learning:",(time.time()-start_time))
    # save model
    if not LOAD:
        torch.save(agent.qnetwork_local.state_dict(), filename+"_qnetwork_local.pth")
        torch.save(agent.qnetwork_target.state_dict(), filename+"_qnetwork_target.pth")
    if PLOT:
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
