import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from hidden_block import HiddenLayerBlock

class Understanding(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, dropout_rate, num_hidden_layers, act_fun):
        super(Understanding, self).__init__()

        # parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.act_fun = act_fun

        self.hiddenLayers = nn.ModuleList([HiddenLayerBlock(self.inputSize, self.hiddenSize, self.dropout_rate, self.act_fun)] + [HiddenLayerBlock(self.hiddenSize, self.hiddenSize, self.dropout_rate, self.act_fun) for i in range(self.num_hidden_layers)])
        self.output_linear = nn.Linear(self.hiddenSize, self.outputSize)
        self.output_activation = torch.relu # ReLU to avoid negative values, that now should not happen

    def forward(self, x):
        x = x.view(-1, self.inputSize).float()
        for hiddenLayer in self.hiddenLayers:
            x = hiddenLayer(x)

        x = self.output_linear(x)
        x = self.output_activation(x)
        return x

    def save_model(self, dict_output, path):
        torch.save(dict_output, path)
