import torch
import torch.nn as nn

import torch.optim as optim
import time

import numpy as np
import csv

# function th`at embeds dataset into one-hots
# each coordinate is turned into a 2D one-hot where a 1 is placed where the position is on the map
# then flatten the one-hot into an input vector
width = 5
height = 5

def precode(pos, width = 5, height = 5):
    map = np.zeros((height, width))
    map[pos[0]][pos[1]] = 1

    return map

def encode(pos, width = 5, height = 5):
    map = precode(pos, width, height)
    map = map.flatten()
    return map

def decode(map, width = 5, height = 5): # returns coordinate given an encoded map
    
    for i in range(len(map)):

        if map[i] == 1:

            col = i % height
            row = int(i/width)

            return torch.tensor((row, col))

# predict movement of death + goal using RNN

# making CTRNN
class CTRNN(nn.Module):

    def __init__(self, input_size, hidden_size, dt = None, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # setting a forgetting-rate
        self.tau = 100
        if dt == None:
            self.alpha = 1
        else:
            self.alpha = dt/self.tau

        self.in2h = nn.Linear(in_features = input_size, out_features = hidden_size) # linear transformation from input layer to hidden layer
        self.h2h  = nn.Linear(in_features = hidden_size, out_features = hidden_size) # linear transformation from hidden layer back into hidden lyaer

    def init_hidden(self, input_shape): # creating a blank set of hidden neurons

        batch_size = input_shape[1] 
        return torch.zeros(batch_size, self.hidden_size)

    def recur(self, input, hidden):

        newHidden = torch.relu(self.in2h(input) + self.h2h(hidden)) #!TODO: why use relu?
        newHidden = hidden * (1 - self.alpha) + newHidden * self.alpha # this step "forgets" a portion of the previous timestep

        return newHidden

    def forward(self, input, hidden = None):
        
        if hidden == None: # initialize a hideen layer
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps  = range(input.size(0))

        for i in steps:
            hidden = self.recur(input[i], hidden) # getting new hidden state
                                                    # each input[i] is a new input layer at timestep i
            output.append(hidden)

        output = torch.stack(output, dim = 0)
        return output, hidden

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # runs a CTRNN followed by a simple MLP for the final output
        self.rnn = CTRNN(input_size, hidden_size, **kwargs) 
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, input):

        rnnOut, _ = self.rnn(input)
        out = self.fc(rnnOut)

        return out, rnnOut

# making ConvNet
class ConvNN(nn.Module):

    def __init__(self, in_channels, out_channels, kern, stride, **kwargs):
        super().__init__()

        # creating a CNN with some generic parameter shapes between each layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kern, stride = stride) 
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(out_channels, 16, kern, stride)
        self.fc1   = nn.Linear(16, 5 * kern * kern)
        self.fc2   = nn.Linear(5 * kern * kern, 3 * kern * kern)
        self.fc3   = nn.Linear(3 * kern * kern, 2 * kern)

    def forward(self, input):

        output = []
        steps  = range(input.size(0))

        for i in steps:
            x = input[i]

            x = self.pool(torch.relu(self.conv1(x))) #!TODO: why use relu?
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

            output.append(x)

        output = torch.stack(output, dim = 0)
        return output

# combining to form CNN-CTRNN
class PWMRNN(nn.Module):

    def __init__(self, cnnkwargs, rnnkwargs, **kwargs):
        super().__init__()

        self.cnn = ConvNN(**cnnkwargs) # setting a generic CNN

        self.rnn_input_size = cnnkwargs['kern']*2 #!TODO: figure out a better way to do this
                                                    # getting the size of the CNN output so that we can feed that as the input to the RNN
        
        self.rnn = RNN(self.rnn_input_size, **rnnkwargs)

    def forward(self, input):

        input = self.cnn(input)
        input = self.rnn(input)

        return input


class DeathNN(nn.Module):

    def __init__(self, inputx = 5, inputy = 5, alpha = 0.001) -> None:
        super(DeathNN, self).__init__()

        self.width = inputx
        self.height = inputy

        self.pred = nn.Sequential(
            nn.Linear(self.width*self.height, 16), 
            nn.ReLU(), #!TODO: ask Fernanda about difference between activation functions
            nn.Linear(16, 2)  # predict movement, which is a 2-dimensional movement vector
        )

    def forward(self, pos):

        if type(pos) != torch.Tensor: # not sure why but code breaks about 60% of the time if this isn't used        
            pos = torch.from_numpy(pos).float()
        nextpos = self.pred(pos)

        return nextpos

class GoalNN(nn.Module):

    def __init__(self, inputx = 5, inputy = 5, alpha = 0.001) -> None:
        super(GoalNN, self).__init__()

        self.width = inputx
        self.height = inputy

        self.pred = nn.Sequential(
            nn.Linear(self.width*self.height, 16),
            nn.ReLU(), #!TODO: ask Fernanda about difference between activation functions
            nn.Linear(16, 2)  # predict movement, which is a 2-dimensional movement vector
        )

    def forward(self, pos):
        
        pos = torch.from_numpy(pos).float()
        nextpos = self.pred(pos)

        return nextpos

def round_prediction(x): # we want all outputs to be integer coordinates
    for i in range(len(x)):
        x[i] = int(x[i] + 0.5)

    return x

"""
def predictDeath(loc):
    return round_prediction(deathNet(loc))

def predictGoal(loc):
    return round_prediction(goalNet(loc))
    """