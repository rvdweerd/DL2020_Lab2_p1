from datasets import BinaryPalindromeDataset

import numpy as np
import torch
#import torch.utils.data as data


bp_dataset = BinaryPalindromeDataset(2)
data_loader = torch.utils.data.DataLoader(bp_dataset, batch_size=3)
data_inputs, data_labels = next(iter(data_loader))
print("Data inputs", data_inputs.shape, "\n", data_inputs)
#print("Data labels", data_labels.shape, "\n", data_labels)
# added comment local laptop
# comment added on dekstop


