"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()


        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.C=torch.zeros(hidden_dim)
        self.h=torch.zeros(hidden_dim)

        self.Wfx = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wfx, mode='fan_out', nonlinearity='sigmoid')
        #self.Wfx.requires_grad=True
        nn.init.kaiming_normal_(self.Wfx, mode='fan_out', nonlinearity='sigmoid')
        self.Wfh = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wfh, mode='fan_out', nonlinearity='sigmoid')
        self.bf = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)

        self.Wix = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wix, mode='fan_out', nonlinearity='sigmoid')
        self.Wih = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wih, mode='fan_out', nonlinearity='sigmoid')
        self.bi = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)

        self.Wgx = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wgx, mode='fan_out', nonlinearity='sigmoid')
        self.Wgh = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wgh, mode='fan_out', nonlinearity='sigmoid')
        self.bg = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)

        self.Wox = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wox, mode='fan_out', nonlinearity='sigmoid')
        self.Woh = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Woh, mode='fan_out', nonlinearity='sigmoid')
        self.bo = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)

        self.Wph = nn.Parameter(torch.ones(hidden_dim,num_classes),requires_grad=True)
        nn.init.kaiming_normal_(self.Wph, mode='fan_out', nonlinearity='sigmoid')
        self.bp = nn.Parameter(torch.zeros(num_classes),requires_grad=True)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        ####################### 
        for s in range(self.batch_size):
            self.C = torch.zeros(self.hidden_dim)
            self.h=torch.zeros(self.hidden_dim)
            for t in range(self.seq_length):
                sig = torch.nn.Sigmoid()
                tanh = torch.nn.Tanh()
                f=sig(torch.matmul(x[s,t,:],self.Wfx) + torch.matmul(self.h,self.Wfh) + self.bf)
                i=sig(torch.matmul(x[s,t,:],self.Wix) + torch.matmul(self.h,self.Wih) + self.bi)
                g=sig(torch.matmul(x[s,t,:],self.Wgx) + torch.matmul(self.h,self.Wgh) + self.bg)
                o=sig(torch.matmul(x[s,t,:],self.Wox) + torch.matmul(self.h,self.Woh) + self.bo)
                self.C = g*i + self.C*f
                self.h = o*tanh(self.C)
        y=torch.matmul(self.h,self.Wph)+self.bp
        return y
        ########################
        # END OF YOUR CODE    #
        #######################
