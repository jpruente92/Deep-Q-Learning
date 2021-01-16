import time
from builtins import print

import numpy as np
import random

from torch.nn import SmoothL1Loss

from Hyperparameter import *

from Neural_networks import QNetwork, Dueling_Network
from Replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F


# helper functions

# compute dimension of state space if flattened into a single vector
def state_dim_to_int(state_dimension):
    return np.array(state_dimension).sum()


# class for the agent
class Agent:

    def __init__(self, state_shape, number_actions, filename_local, filename_target, seed, profile):
        state_dimension = state_dim_to_int(state_shape)
        self.profile = profile
        self.state_dimension = state_dimension
        self.number_actions = number_actions
        random.seed(seed)
        self.max_priority = 1000
        # use gpu if available else cpu
        # self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        if DUELING_NETWORK:
            self.qnetwork_local = Dueling_Network(state_dimension, number_actions, seed, filename_local, self.device,
                                                  self.profile).to(self.device)
            self.qnetwork_target = Dueling_Network(state_dimension, number_actions, seed, filename_target, self.device,
                                                   self.profile).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_dimension, number_actions, seed, filename_local, self.device,
                                           self.profile).to(self.device)
            self.qnetwork_target = QNetwork(state_dimension, number_actions, seed, filename_target, self.device,
                                            self.profile).to(self.device)
        # Replay memory for sampling from former experiences
        self.memory = ReplayBuffer(number_actions, seed, profile)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    # doing one step in the environment: saving experience in replay buffer and learn
    def step(self, state, action, reward, next_state, done, B):
        # Save experience in replay memory with maximum priority
        self.memory.add(state, action, reward, next_state, done, self.max_priority, A)
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % (UPDATE_EVERY * UPDATE_TARGET_EVERY)
        if (self.t_step % UPDATE_EVERY) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                nodes, probabilities = self.memory.sample()
                self.learn(nodes, probabilities, GAMMA, B)

    # returns the action following the epsilon-greedy policy
    def act(self, state, eps=0.):
        # if greedy action is selected
        if random.random() > eps:
            state = torch.from_numpy(state).float().to(self.device)
            action_values = self.qnetwork_local.evaluate(state, False)
            # return best action
            return np.argmax(action_values.cpu().data.numpy())
        # if random action is selected
        else:
            # return random action
            return random.choice(np.arange(self.number_actions))

    def learn(self, samples, probabilities, gamma, B):
        self.profile.total_number_learn_calls += 1
        start_time = time.time()
        states, actions, rewards, next_states, dones = self.samples_to_environment_values(samples)
        if (DOUBLE_Q):
            # compute action values with local network
            action_values = self.qnetwork_local.evaluate(next_states, False)
            # determine best action
            best_actions = action_values.argmax(1)
            # compute q targets with target network and choose best action
            Q_targets_next = torch.tensor([self.qnetwork_target.evaluate(next_state, False)[int(next_action)]
                                           for next_action, next_state in zip(best_actions, next_states)]).unsqueeze(1)
        else:
            Q_targets_next = torch.tensor(
                [self.qnetwork_target.evaluate(next_state, False).detach().max(1)[0].unsqueeze(1) for next_state in
                 next_states])
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local.evaluate(states, True).gather(1, actions)
        # Compute loss for each dimension
        elementwise_loss = []
        for q_ex, q_tar in zip(Q_expected, Q_targets):
            elementwise_loss.append(self.unweighted_smooth_l1_loss(q_ex, q_tar))

        start_time_update = time.time()
        if PRIORITIZED_EXP_REPLAY:
            # update priorities
            for node, q_exp, q_tar in zip(samples, Q_expected, Q_targets):
                td_error = abs(q_exp.detach().numpy() - q_tar.detach().numpy())
                # for i, node in enumerate(samples):
                #     td_error = elementwise_loss[i].detach().cpu().numpy()
                # set priority of the given samples, small positive priority has to be guaranteed
                priority = float(max(abs(td_error), 1e-6))
                # print(td_errors[i],"\t",priority)
                self.memory.memory.sum_tree.update_priority(node, priority)
                # update max priority
                self.max_priority = max(priority, self.max_priority)
            self.profile.total_time_updating_priorities += (time.time() - start_time_update)
            start_time_isw = time.time()
            # multiply Q-targets and Q expected with importance-sampling weight
            max_importance_sampling_weight = 0
            importance_sampling_weights = []
            for i, prob in enumerate(probabilities):
                importance_sampling_weight = (1.0 / len(self.memory) / probabilities[i]) ** B
                importance_sampling_weights.append(importance_sampling_weight)
                max_importance_sampling_weight = max(importance_sampling_weight, max_importance_sampling_weight)
            for i in range(len(elementwise_loss)):
                elementwise_loss[i] *= importance_sampling_weights[i] / max_importance_sampling_weight
            self.profile.total_time_introducing_isw += (time.time() - start_time_isw)

        # Train the local network
        start_time_training = time.time()

        # Minimize the loss
        self.qnetwork_local.optimizer.zero_grad()
        loss = torch.mean(torch.stack(elementwise_loss)).unsqueeze(0)
        loss.backward()
        self.qnetwork_local.optimizer.step()
        self.profile.total_time_training += (time.time() - start_time_training)

        if SOFT_UPDATE:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        elif (self.t_step % UPDATE_TARGET_EVERY) == 0:
            self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.profile.total_time_learning += (time.time() - start_time)

    def soft_update(self, local_model, target_model, tau):
        start_time = time.time()
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.profile.total_time_soft_update += (time.time() - start_time)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def samples_to_environment_values(self, samples):
        start_time = time.time()
        if PRIORITIZED_EXP_REPLAY:
            samples = [node.value for node in samples]
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(
            self.device)
        self.profile.total_time_samples_to_environment_values += (time.time() - start_time)
        return states, actions, rewards, next_states, dones

    def unweighted_smooth_l1_loss(self, input, target):
        t = torch.abs(input - target)
        return torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
