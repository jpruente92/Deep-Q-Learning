# class for generating the neural network
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from Hyperparameter import Hyperparameters


class QNetwork(torch.nn.Module):

    def __init__(self, state_dimension, number_actions, seed, filename, device, profile,param):
        super(QNetwork, self).__init__()
        self.param = param
        self.profile=profile
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_dimension, self.param.hidden_layers[0])])
        layer_sizes = zip(self.param.hidden_layers[:-1], self.param.hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(self.param.hidden_layers[-1], number_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.param.LR)
        self.device=device
        # load weights and biases if given
        if self.param.LOAD:
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




