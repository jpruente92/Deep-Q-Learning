import time

import numpy as np
import random
from collections import deque


from Neural_networks import QNetwork_pytorch

import torch
import torch.nn.functional as F

# experience replay and fixed Q targets (soft and hard update) are always active
# Hyperparameters
LOAD = False            # loading neural networks from file
PY_TORCH = True         # if true pytorch else tensorflow
SOFT_UPDATE = True      # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
UPDATE_TARGET_EVERY = 20 # how often to do the hard update for targets
DOUBLE_Q= False          # applying double q learning
PRIORITIZED_EXP_REPLAY= True          # applying prioritized experience replay
BUFFER_SIZE = 10000    # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 0.0005             # learning rate
UPDATE_EVERY = 4        # how often to update the network



# helper functions

# compute dimension of state space if flattened into a single vector
def state_dim_to_int(state_dimension):
    return np.array(state_dimension).sum()


# class for the agent
class Agent_py_torch():

    def __init__(self, state_shape, number_actions, hidden_layers, filename_local, filename_target, seed, profile):
        state_dimension = state_dim_to_int(state_shape)
        self.profile=profile
        self.state_dimension = state_dimension
        self.number_actions = number_actions
        self.hidden_layers = hidden_layers
        self.seed = random.seed(seed)
        self.max_priority=1
        # use gpu if available else cpu
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork_pytorch(state_dimension, number_actions, hidden_layers, seed, LR, LOAD, filename_local, self.device, self.profile).to(self.device)
        self.qnetwork_target = QNetwork_pytorch(state_dimension, number_actions, hidden_layers, seed, LR, LOAD, filename_target, self.device, self.profile).to(self.device)
        # Replay memory for sampling from former experiences
        self.memory = ReplayBuffer(number_actions, BUFFER_SIZE, BATCH_SIZE, seed, profile)
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
            state =torch.from_numpy(state).float().to(self.device)
            action_values = self.qnetwork_local.evaluate(state,False)
            # return best action
            return np.argmax(action_values.cpu().data.numpy())
        # if random action is selected
        else:
            # return random action
            return random.choice(np.arange(self.number_actions))



    def learn(self, samples,probabilities, gamma,B):
        self.profile.total_number_learn_calls+=1
        start_time=time.time()
        states, actions, rewards, next_states, dones = self.samples_to_environment_values(samples)
        if(DOUBLE_Q):
            # compute action values with local network
            action_values = self.qnetwork_local.evaluate(next_states,False)
            # determine best action
            best_actions = action_values.argmax(1)
            # compute q targets with target network and choose best action
            Q_targets_next=torch.tensor([self.qnetwork_target.evaluate(next_state,True)[int(next_action)] for next_action,next_state in zip(best_actions,next_states)]).unsqueeze(1)
            # Q_targets_next=np.array([self.qnetwork_target.evaluate(next_state)[int(next_action)] for next_action,next_state in zip(best_actions,next_states)])
        else:
            Q_targets_next = self.qnetwork_target.evaluate(next_states,True).detach().max(1)[0].unsqueeze(1)
            # Q_targets_next = self.qnetwork_target.evaluate(next_states).max(1)[0]
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local.evaluate(states,True).gather(1, actions)
        # Q_expected = np.array([self.qnetwork_local.evaluate(state)[action] for action,state in zip(actions,states)])

        start_time_update=time.time()
        if PRIORITIZED_EXP_REPLAY:
            # update priorities
            td_errors = Q_targets-Q_expected
            for i, sample in enumerate(samples):
                # set priority of the given samples, small positive priority has to be guaranteed
                priority=float(max(abs(td_errors[i]),0.01))
                sample.priority=priority
                # update max priority
                self.max_priority=max(priority,self.max_priority)
            self.profile.total_time_updating_priorities+=(time.time() - start_time_update)
            start_time_isw=time.time()
            # multiply Q-targets with importance-sampling weight
            for i,prob in enumerate(probabilities):
                importance_sampling_weight = (1.0/len(self.memory)/probabilities[i])**B
                # print ("\tios",importance_sampling_weight)
                Q_targets[i]*=importance_sampling_weight
            self.profile.total_time_introducing_isw+=(time.time() - start_time_isw)


        # Train the local network
        start_time_training=time.time()
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.qnetwork_local.optimizer.zero_grad()
        loss.backward()
        self.qnetwork_local.optimizer.step()
        self.profile.total_time_training+=(time.time() - start_time_training)

        if SOFT_UPDATE:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        elif(self.t_step% UPDATE_TARGET_EVERY) == 0:
            self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.profile.total_time_learning+=(time.time() - start_time)

    def soft_update(self, local_model, target_model, tau):
        start_time=time.time()
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        self.profile.total_time_soft_update+=(time.time() - start_time)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


    def samples_to_environment_values(self,samples):
        start_time=time.time()
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(self.device)
        self.profile.total_time_samples_to_environment_values+=(time.time() - start_time)
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

    def __init__(self, number_actions, buffer_size, batch_size, seed, profile):
        self.number_actions = number_actions
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.profile = profile

    def add(self, state, action, reward, next_state, done,priority):
        new_experience = Experience(state, action, reward, next_state, done,priority)
        self.memory.append(new_experience)

    def sample(self,A):
        start_time=time.time()
        self.profile.total_number_sampling_calls+=1
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
        self.profile.total_time_sampling+=(time.time() - start_time)
        return samples, probabilities_of_samples

    def __len__(self):
        return len(self.memory)
