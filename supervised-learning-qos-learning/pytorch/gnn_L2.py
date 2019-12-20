import torch
from torch_geometric.nn import TopKPooling, NNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from batchnorm_dropout import BatchnormDropout
from understanding import Understanding
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, dropout = 0, pooling_ratio = 0.8, num_nodes = 12, num_edges = 30, out_channel = 32, hidden_size_gnn = 192):
        super(Net, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.feature_per_OD = 2
        self.num_total_features_per_node = self.feature_per_OD * (self.num_nodes - 1)
        self.output_space = self.num_nodes ** 2 - self.num_nodes
        # TODO solo per prova: out_channel = self.output_space

        self.nn_1 = Understanding(num_edges, hidden_size_gnn, out_channel * self.num_total_features_per_node, dropout, 1, F.relu)
        self.conv1 = NNConv(self.num_total_features_per_node, out_channel, self.nn_1)
        self.bn1 = BatchnormDropout(out_channel, dropout)
        self.pool1 = TopKPooling(out_channel, ratio=pooling_ratio)

        self.nn_2 = Understanding(num_edges, hidden_size_gnn, out_channel * out_channel, dropout, 1, F.relu)
        self.conv2 = NNConv(out_channel, out_channel, self.nn_2)
        self.bn2 = BatchnormDropout(out_channel, dropout)
        self.pool2 = TopKPooling(out_channel, ratio=pooling_ratio)

        self.nn_3 = Understanding(num_edges, hidden_size_gnn, out_channel * out_channel, dropout, 1, F.relu)
        self.conv3 = NNConv(out_channel, out_channel, self.nn_3)
        self.bn3 = BatchnormDropout(out_channel, dropout)
        self.pool3 = TopKPooling(out_channel, ratio=pooling_ratio)

        self.lin1 = torch.nn.Linear(self.num_nodes * out_channel, 96) # if I add the global pooling, out_channel * 2
        self.bn1 = torch.nn.BatchNorm1d(96)
        self.act1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(96, 48)
        self.bn2 = torch.nn.BatchNorm1d(48)
        self.act2 = torch.nn.ReLU()
        self.lin3 = torch.nn.Linear(48, self.output_space)
        self.act3 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        batch_size = int(x.size(0) / self.num_nodes)
        # x = x.float()  # for any situation on which it isn't, i.e., dropped packets: long -> float required
        # if edge_attr is not None:
        #     edge_attr = edge_attr.float().view(-1, self.num_edges)
        """
        x = self.conv1(x, edge_index.view(self.num_edges * 2, -1), edge_attr)
        # x = self.bn1(x)
        
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.conv2(x, edge_index.view(self.num_edges * 2, -1), edge_attr)
        # x = self.bn2(x)
        
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.conv3(x, edge_index.view(self.num_edges * 2, -1), edge_attr)
        # x = self.bn3(x)
        
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1 + x2 + x3
        
        x = self.vanilla(x)
        """
        x = F.relu(self.conv1(x, edge_index.view(self.num_edges * 2, -1), edge_attr))
        # x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        """
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index.view(self.num_edges * 2, -1), edge_attr))

        # x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index.view(self.num_edges * 2, -1), edge_attr))

        # x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        """
        x = x.view(batch_size, -1)
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        return x

    def save_model(self, dict_output, path):
        torch.save(dict_output, path)