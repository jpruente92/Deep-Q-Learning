# class for generating the neural network
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class QNetwork_pytorch(torch.nn.Module):

    def __init__(self, state_dimension, number_actions, hidden_layers, seed,LR,LOAD,filename,device):
        super(QNetwork_pytorch, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_dimension, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], number_actions)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
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

    def evaluate(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set evaluation mode
        self.qnetwork_local.eval()
        # for evaluation no grad bc its faster
        with torch.no_grad():
            # compute action values
            action_values = self.forward(state)
            # set training mode
        self.qnetwork_local.train()
        return action_values

    def save(self,filename):
        torch.save(self.state_dict(), filename+"_qnetwork_local.pth")

class QNetwork_tensorflow():
    def __init__(self, state_dimension, number_actions, hidden_layers, seed,LR,LOAD,filename):
        self.seed = seed
        if LOAD:
            model =tf.keras.models.load_model(filename)
            print("Values loaded")
        else:
            # create model
            self.model= tf.keras.models.Sequential()
            # Input Layer: Flatten layer for input transformation
            self.model.add(tf.keras.layers.Flatten())
            # hidden layers
            for hl_size in hidden_layers:
                self.model.add(tf.keras.layers.Dense(hl_size, activation=tf.nn.relu))
            # output layer
            self.model.add(tf.keras.layers.Dense(number_actions, activation=tf.nn.linear))
            # specify optimizer and loss
            model.compile(optimizer='adam',loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

    def evaluate(self, state):
        return self.model.predict([state])

    def save(self,filename):
        self.model.save(filename)


