import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Hidden layer containing, in order:
- batchnorm
- sigmoid
- dropout
-
as described here: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
'''
class HiddenLayerBlock(torch.nn.Module):
    def __init__(self, inputSize, outputSize, dropout_rate, act_fun = torch.sigmoid):
        super(HiddenLayerBlock, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)
        self.batchnorm = nn.BatchNorm1d(outputSize)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply_dropout = dropout_rate > 0
        self.act_fun = act_fun

    def forward(self, x):
        o = self.linear(x)
        o = self.batchnorm(o)
        o = self.act_fun(o)
        if self.apply_dropout:
            o = self.dropout(o)
        return o
