import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Hidden layer containing, in order:
- batchnorm
- non linearity
- dropout
-
as described here: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
'''
class BatchnormDropout(torch.nn.Module):
    def __init__(self, inputSize, dropout_rate, act_fun = F.relu):
        super(BatchnormDropout, self).__init__()
        self.batchnorm = nn.BatchNorm1d(inputSize)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_fun = act_fun
        self.apply_dropout = dropout_rate > 0

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.act_fun(x)
        if self.apply_dropout:
            x = self.dropout(x)
        return x
