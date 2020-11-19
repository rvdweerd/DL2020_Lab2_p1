###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM

import numpy as np

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

###############################################################################


def train(config):
    np.random.seed(0)
    torch.manual_seed(0)


    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        #config.num_classes = config.input_length
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = config.input_length*4+2-1



    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            config.input_length, config.input_dim,
            config.num_hidden, 2,#config.num_classes, #overridden to avoid errors, lstm was optimized for binary predictions
            config.batch_size, device
        ).to(device)
        print('LIST init:')
        print('input_len',config.input_length)
        print('input_dim',config.input_dim)
        print('num_hidden',config.num_hidden)
        print('num_classes',2)#config.num_classes) #overridden to avoid errors, lstm was optimized for binary predictions
        print('batch_size',config.batch_size)
        print('learning_rate',config.learning_rate)
        print('device',config.device)
        print('Number of trainable parameters: ',model.numTrainableParameters())
    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    if config.model_type == 'LSTM': # LSTM optimized for binary classification, using BCE
        loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        loss_function = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Setup tensorboard writer
    writer=SummaryWriter(f'runs/LSTM/MiniBatchSize {config.batch_size}/lr {config.learning_rate}')
    acc_plt=[]
    loss_plt=[]
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)

        # Compute the loss, gradients and update network parameters
        if config.model_type == 'LSTM':
            loss = loss_function(log_probs, batch_targets.float())
        else:
            loss = loss_function(log_probs, batch_targets)

        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        if config.model_type == 'LSTM': # using binary classification, not hot vectors
            correct = ((log_probs>0)==batch_targets).sum().item()
            accuracy = correct / log_probs.size(0)
        else:
            predictions = torch.argmax(log_probs, dim=1)
            correct = (predictions == batch_targets).sum().item()
            accuracy = correct / log_probs.size(0)
        
        acc_plt.append(accuracy)
        loss_plt.append(loss)
        writer.add_scalar('Training loss',loss,global_step=step)
        writer.add_scalar('Accuracy',accuracy,global_step=step)
        # print(predictions[0, ...], batch_targets[0, ...])

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 60 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        # Check if training is finished
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break
    
    if config.model_type == 'LSTM':
        # Test accuracy calculation
        model.eval()
        with torch.no_grad():
            correct=0
            total=0
            numBatchesTestEval=5
            test_loss=0
            for step, (x, t) in enumerate(data_loader):
            #for k in range(numBatchesTestEval):
                #x,t=next(iter(data_loader))
                if device.type=='cuda':
                    x.to(device)
                    x=x.cuda()
                    t.to(device)
                    t=t.cuda()
                #x_e = embedding(x.squeeze(2).type(torch.LongTensor))
                #pred=lstm(x_e)
                log_probs=model(x)
                #predictions = torch.argmax(log_probs, dim=1)
                #correct += (predictions == batch_targets).sum().item()
                correct += ((log_probs>0) == t).sum().item()
                total += log_probs.size(0)
                test_loss += loss_function(log_probs,t.float())/numBatchesTestEval
                if step==numBatchesTestEval:
                    break
            test_accuracy=correct/total
        model.train()
        print('Test accuracy: ',test_accuracy)
    
    plt.plot(acc_plt,label='Train accuracy')
    plt.plot(loss_plt,label='Train loss')
    plt.title('Train loss and accuracy curves, Binary Palindrome LSTM',fontsize=15)
    plt.xlabel('Training step (mini batch)',fontsize=15)
    plt.ylabel('Loss (Cat.CE)',fontsize=15)
    note1 = 'seq_len='+str(config.input_length)+', num_hidden='+str(config.num_hidden)+', in_dim='+str(config.input_dim)
    note2 = 'bsize=' + str(config.batch_size) + ', lr=%.1E' %config.learning_rate
    plt.text(0,0.15, note1)
    plt.text(0,0.1, note2)
    plt.legend()
    axes=plt.axes()
    axes.set_ylim(0,1.05)
    plt.show()
    
    plt.plot(acc_plt)
    plt.show()
    writer.add_hparams({'lr':config.learning_rate,'bsize':config.batch_size},{'accuracy':test_accuracy,'loss':test_loss})
    print('Done training.')
    ###########################################################################
    ###########################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    #parser.add_argument('--dataset', type=str, default='randomcomb',
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    #parser.add_argument('--model_type', type=str, default='biLSTM',
    parser.add_argument('--model_type', type=str, default='LSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    #parser.add_argument('--input_length', type=int, default=6,
    parser.add_argument('--input_length', type=int, default=2,#10,
                        help='Length of an input sequence')
    #parser.add_argument('--input_dim', type=int, default=1,
    parser.add_argument('--input_dim', type=int, default=20,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    #parser.add_argument('--learning_rate', type=float, default=0.001,
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=400,#3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    # Misc params
    #parser.add_argument('--device', type=str, default="cpu",
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    config = parser.parse_args()

    # Train the model
    train(config)
