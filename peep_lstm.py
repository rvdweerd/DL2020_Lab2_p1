"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.C=torch.zeros(batch_size,hidden_dim)
        self.h=torch.zeros(batch_size,hidden_dim)
        self.device=device

        self.embedding = torch.nn.Embedding(num_classes+1,input_dim,padding_idx=0)

        self.Wfx = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        self.Wfc = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        self.bf = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wfx, mode='fan_out', nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.Wfc, mode='fan_out', nonlinearity='sigmoid')

        self.Wix = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        self.Wic = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        self.bi = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wix, mode='fan_out', nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.Wic, mode='fan_out', nonlinearity='sigmoid')
        
        self.Wox = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        self.Woc = nn.Parameter(torch.ones(hidden_dim,hidden_dim),requires_grad=True)
        self.bo = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wox, mode='fan_out', nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.Woc, mode='fan_out', nonlinearity='sigmoid')
        
        self.Wcx = nn.Parameter(torch.ones(input_dim,hidden_dim),requires_grad=True)
        self.bc = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.Wcx, mode='fan_out', nonlinearity='sigmoid')

        self.Wph = nn.Parameter(torch.ones(hidden_dim,num_classes),requires_grad=True)
        self.bp = nn.Parameter(torch.zeros(num_classes),requires_grad=True)
        nn.init.kaiming_normal_(self.Wph, mode='fan_out', nonlinearity='sigmoid')
        
        self.lsm=nn.LogSoftmax(dim=1)
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        x_in =  x.squeeze(2).type(torch.LongTensor).to(self.device)
        #x_in =  x.type(torch.LongTensor).to(self.device)
        x=self.embedding(x_in)
        
        self.C = torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        self.h=torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        for t in range(self.seq_length):
            
            f=self.sig(torch.matmul(x[:,t,:],self.Wfx) + torch.matmul(self.C,self.Wfc) + self.bf)
            i=self.sig(torch.matmul(x[:,t,:],self.Wix) + torch.matmul(self.C,self.Wic) + self.bi)
            o=self.sig(torch.matmul(x[:,t,:],self.Wox) + torch.matmul(self.C,self.Woc) + self.bo)
            self.C = (self.sig(torch.matmul(x[:,t,:],self.Wcx)+self.bc))*i + self.C*f
            
            # Check whether sequences in batch at this timestep have a token as entry, set state to zero if so
            resetVec=(x_in[:,t]>0).type(torch.FloatTensor).to(self.device) 
            self.C = torch.einsum('ij,i->ij',self.C,resetVec)
            self.h = self.tanh(self.C)*o
        p=torch.matmul(self.h,self.Wph)+self.bp
        
        y_hat = self.lsm(p)
        return y_hat
        

    def numTrainableParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        ########################
        # END OF YOUR CODE    #
        #######################

