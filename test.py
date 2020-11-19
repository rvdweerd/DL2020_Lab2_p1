from datasets import BinaryPalindromeDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from lstm import LSTM

from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#import torch.utils.data as data
N_PALIN = 2
seq_len = N_PALIN*4+1
#BATCH_SIZE = 256
N_CLASSES=1 # just binary classifier
INPUT_DIM=10
HIDDEN_DIM=256
LEARNING_RATE=1e-4



if torch.cuda.is_available():
    device = torch.device('cuda')
    # added seed for GPU
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
else:
    device = torch.device('cpu')
print('device used:',device)

bp_dataset = BinaryPalindromeDataset(2)

batch_sizes=[8,256]
for BATCH_SIZE in batch_sizes:
    writer=SummaryWriter(f'runs/LSTM/MiniBatchSize {BATCH_SIZE}')
    data_loader = torch.utils.data.DataLoader(bp_dataset, batch_size=BATCH_SIZE)
    lstm = LSTM(seq_len,INPUT_DIM,HIDDEN_DIM,N_CLASSES,BATCH_SIZE,device)
    lstm.to(device)
    loss_module = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss_module.to(device)
    optimizer = torch.optim.Adam(lstm.parameters(),lr=LEARNING_RATE)
    print('Number of trainable parameters: ',count_parameters(lstm))
    acc_plt=[]
    loss_plt=[]

    for i in range(50):
        data_inputs, data_labels = next(iter(data_loader))
        if device.type=='cuda':
            data_inputs=data_inputs.to(device=device)
            #data_inputs.to(device)
            #data_inputs=data_inputs.cuda()
            data_labels=data_labels.to(device=device)
            #data_labels.to(device)
            #data_labels=data_labels.cuda()
    
        #x_emb=embedding((data_inputs.squeeze(2)).type(torch.LongTensor))
        #pred = lstm(x_emb)
        pred = lstm(data_inputs)
        loss=loss_module(pred,data_labels.float())
        optimizer.zero_grad()
        # if i==0:
        #     loss.backward(retain_graph=True)
        # else:
        #     loss.backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(),max_norm=1)
        optimizer.step()
        loss_plt.append(loss.item())
        writer.add_scalar('Training loss',loss,global_step=i)
        #writer.add_scalar('Training acc')
        if i%10 == 0:
            lstm.eval()
            with torch.no_grad():
                correct=0
                total=0
                for k in range(5):
                    x,t=next(iter(data_loader))
                    if device.type=='cuda':
                        x.to(device)
                        x=x.cuda()
                        t.to(device)
                        t=t.cuda()
                    #x_e = embedding(x.squeeze(2).type(torch.LongTensor))
                    #pred=lstm(x_e)
                    pred=lstm(x)
                    correct += ((pred>0)==t).sum().item()
                    total += t.size(0)
            lstm.train()
            acc_plt.append(correct/total)
            print('accuracy',acc_plt[-1])

        print('step ',i,'loss=',loss)
    writer.add_hparams({'lr':LEARNING_RATE,'bsize':BATCH_SIZE},{'accuracy':acc_plt[-1],'loss':sum(loss_plt)/len(loss_plt)})
    plt.plot(loss_plt)
    plt.show()
    plt.plot(acc_plt)
    plt.show()