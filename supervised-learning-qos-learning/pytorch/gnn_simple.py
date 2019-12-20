import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import TopKPooling, NNConv, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from batchnorm_dropout import BatchnormDropout
from understanding import Understanding
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, dropout = 0, pooling_ratio = 0.8, num_nodes = 12, num_edges = 30, out_channel = 128, hidden_size_gnn = 256):
        super(Net, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_node_feature = 2
        self.apply_batchnorm = False

        hidden_space, output_space = int(hidden_size_gnn / 2), 11# self.num_nodes ** 2 - self.num_nodes

        self.conv1 = GCNConv(self.num_node_feature * (self.num_nodes - 1), hidden_size_gnn)
        self.bn1 = BatchNorm1d(hidden_size_gnn)
        self.conv2 = GCNConv(hidden_size_gnn, hidden_space)
        self.bn2 = BatchNorm1d(hidden_space)
        self.conv3 = GCNConv(hidden_space, hidden_space)
        self.bn3 = BatchNorm1d(hidden_space)
        self.conv4 = GCNConv(hidden_space, hidden_space)
        self.bn4 = BatchNorm1d(hidden_space)
        self.conv5 = GCNConv(hidden_space, hidden_space)
        self.bn5 = BatchNorm1d(hidden_space)
        self.vanilla = Understanding(hidden_space, int(np.power(hidden_space * output_space, 0.5)), output_space, dropout, 1, torch.relu)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.apply_batchnorm:
            x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.apply_batchnorm:
            x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        if self.apply_batchnorm:
            x = self.bn3(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        if self.apply_batchnorm:
            x = self.bn4(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        if self.apply_batchnorm:
            x = self.bn5(x)
        return self.vanilla(x)

    def save_model(self, dict_output, path):
        torch.save(dict_output, path)
