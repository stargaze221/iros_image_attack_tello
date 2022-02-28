import torch
import torch.nn as nn
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


class FeedforwardNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        #self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.drop1 = nn.Dropout(.5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 2)
        #self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.drop2 = nn.Dropout(.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, self.output_size)
        self.fc3.weight.data.uniform_(-EPS,EPS)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.fc1(x)
        drop1 = self.drop1(hidden1)
        relu1 = self.relu1(drop1)
        hidden2 = self.fc2(relu1)
        drop2 = self.drop2(hidden2)
        relu2 = self.relu2(drop2)
        output = self.fc3(relu2)
        #output = self.sigmoid(output)
        return output

