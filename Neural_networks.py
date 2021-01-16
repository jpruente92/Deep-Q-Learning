# class for generating the neural network
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from Hyperparameter import *


class QNetwork(torch.nn.Module):

    def __init__(self, state_dimension, number_actions, seed, filename, device, profile):
        super(QNetwork, self).__init__()
        self.profile=profile
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_dimension, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], number_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.device=device
        # load weights and biases if given
        if LOAD:
            self.load_state_dict(torch.load(filename))
            print("Values loaded")

    def forward(self, state):
        x=state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        return self.output(x)

    def evaluate(self, state, requires_grad):
        start_time=time.time()
        self.profile.total_number_evaluate_calls+=1
        # set evaluation mode
        self.eval()
        if requires_grad:
            action_values = self.forward(state)
        # for evaluation no grad bc its faster
        else:
            with torch.no_grad():
                # compute action values
                action_values = self.forward(state)
        # set training mode
        self.train()
        self.profile.total_time_evaluation+=(time.time() - start_time)
        return action_values



    def save(self,filename):
        torch.save(self.state_dict(), filename)


class Dueling_Network(torch.nn.Module):
    def __init__(self, state_dimension, number_actions, seed, filename, device, profile):
        super(Dueling_Network, self).__init__()
        self.profile=profile
        self.seed = torch.manual_seed(seed)
        # network that is shared by both streams
        self.shared_network = nn.Sequential(
            nn.Linear(state_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # stream for the values
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # stream for the advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, number_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.device=device
        # load weights and biases if given
        if LOAD:
            self.load_state_dict(torch.load(filename))
            print("Values loaded")


    def forward(self, state):
        x=state
        shared_values=self.shared_network(x)
        state_value=self.value_stream(shared_values)
        advantage_values=self.advantage_stream(shared_values)
        quality_values=(advantage_values - advantage_values.mean())+state_value
        return quality_values

    def evaluate(self, state, requires_grad):
        start_time=time.time()
        self.profile.total_number_evaluate_calls+=1
        # set evaluation mode
        self.eval()
        if requires_grad:
            action_values = self.forward(state)
        # for evaluation no grad bc its faster
        else:
            with torch.no_grad():
                # compute action values
                action_values = self.forward(state)
        # set training mode
        self.train()
        self.profile.total_time_evaluation+=(time.time() - start_time)
        return action_values

    def save(self,filename):
        torch.save(self.state_dict(), filename)

