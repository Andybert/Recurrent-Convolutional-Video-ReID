import torch
import torchvision
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import torch.legacy as legacy
import os
import argparse


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class CNN_RNN(nn.Module):

    def __init__(self, nFilters, opt):
        super(CNN_RNN, self).__init__()
        self.padDim = 4
        self.poolsize = 2
        self.stepsize = 2
        self.inputchannels = 5
        self.filtsize = 5
        self.nFilters = nFilters
        self.opt = opt
        self.pad1 = nn.ZeroPad2d((self.padDim, self.padDim, self.padDim, self.padDim))
        self.conv1 = nn.Conv2d(in_channels=self.inputchannels, out_channels=self.nFilters[0], kernel_size=(self.filtsize, self.filtsize), stride=(1, 1))
        self.activation1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d((self.poolsize, self.poolsize), stride=(self.stepsize, self.stepsize))
        self.pad2 = nn.ZeroPad2d((self.padDim, self.padDim, self.padDim, self.padDim))
        self.conv2 = nn.Conv2d(in_channels=self.nFilters[0], out_channels=self.nFilters[1], kernel_size=(self.filtsize, self.filtsize), stride=(1, 1))
        self.activation2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d((self.poolsize, self.poolsize), stride=(self.stepsize, self.stepsize))
        self.pad3 = nn.ZeroPad2d((self.padDim, self.padDim, self.padDim, self.padDim))
        self.conv3 = nn.Conv2d(in_channels=self.nFilters[1], out_channels=self.nFilters[2], kernel_size=(self.filtsize, self.filtsize), stride=(1, 1))
        self.activation3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d((self.poolsize, self.poolsize), stride=(self.stepsize, self.stepsize))
        self.dropout = nn.Dropout(self.opt['dropoutFrac'])
        self.fc = nn.Linear(self.nFilters[2] * 10 * 8, self.opt['embeddingSize'])
        self.rnn = nn.RNN(input_size=self.opt['embeddingSize'], hidden_size=self.opt['embeddingSize'], num_layers=1,
                          nonlinearity='tanh', bias=True, batch_first=False, dropout=self.opt['dropoutFracRNN'])

    def forward(self, x):
        x = self.pool1(self.activation1(self.conv1(self.pad1(x))))
        x = self.pool2(self.activation2(self.conv2(self.pad2(x))))
        x = self.pool3(self.activation3(self.conv3(self.pad3(x))))
        x = x.view(-1, self.nFilters[2] * 10 * 8)
        x = self.dropout(x)
        x = self.fc(x).unsqueeze(1)
        # x = self.dropout(x)
        h_0 = Variable(torch.cuda.DoubleTensor(1, 1, self.opt['embeddingSize']).zero_())
        x, _ = self.rnn(x, h_0)
        x = torch.mean(x, 0)
        x = x.view(-1, self.opt['embeddingSize'])
        return x


class FullModel(nn.Module):

    def __init__(self, nFilters, opt):
        super(FullModel, self).__init__()
        self.nFilters = nFilters
        self.opt = opt
        self.net_P1 = CNN_RNN(self.nFilters, self.opt)
        # self.net_P2 = CNN_RNN(self.nFilters, self.opt)
        self.fc1 = nn.Linear(self.opt['embeddingSize'], self.opt['train_category_num'])
        # self.fc2 = nn.Linear(self.opt['embeddingSize'], self.opt['train_category_num'])
        self.pdist = nn.PairwiseDistance(2)

    def forward(self, p1, p2):
        P1 = self.net_P1(p1)
        P2 = self.net_P1(p2)
        softout1 = self.fc1(P1)
        softout2 = self.fc1(P2)
        pairout = self.pdist(P1, P2)
        # print 'pairout: ', pairout
        return P1, P2, softout1, softout2, pairout

# nFilters = [16, 32, 32]
# opt = {'clip': 5, 'lr': 0.001, 'dropoutFrac': 0.6, 'dropoutFracRNN': 0.6, 'embeddingSize': 128, 'sampleSeqLength': 16, 'scale_num': 4, 'train_category_num': 150}
# net = FullModel(nFilters, opt)
# print net
