from datasets import BinaryPalindromeDataset

import numpy as np
import torch
from lstm import LSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#import torch.utils.data as data
N_PALIN = 2
seq_len = N_PALIN*4+1
BATCH_SIZE = 1
N_CLASSES=1 # just binary classifier
INPUT_DIM=10
HIDDEN_DIM=15

if torch.cuda.is_available():
    device = torch.device('cuda')
    # added seed for GPU
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
else:
    device = torch.device('cpu')
print('device used:',device)

embedding = torch.nn.Embedding(N_CLASSES+2,INPUT_DIM,padding_idx=0)
bp_dataset = BinaryPalindromeDataset(2)
data_loader = torch.utils.data.DataLoader(bp_dataset, batch_size=BATCH_SIZE)

lstm = LSTM(seq_len,INPUT_DIM,HIDDEN_DIM,N_CLASSES,BATCH_SIZE,device)
loss_module = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=1e-3)
print('Number of trainable parameters: ',count_parameters(lstm))
torch.autograd.set_detect_anomaly(True)
for i in range(100):
    data_inputs, data_labels = next(iter(data_loader))
    x_emb=embedding((data_inputs.squeeze(2)).type(torch.LongTensor))
    pred = lstm(x_emb)
    loss=loss_module(pred,data_labels.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('step ',i,'loss=',loss)
