import torch
from torch_geometric.nn import TopKPooling, NNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from batchnorm_dropout import BatchnormDropout
from understanding import Understanding
class Net(torch.nn.Module):
    def __init__(self, dropout = 0.4, pooling_ratio = 0.8, num_nodes = 12, num_edges = 30, out_channel = 128, hidden_size_gnn = 128):
        super(Net, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_node_feature = 2


        # self.nn_1 = Understanding(num_edges, hidden_size_gnn, out_channel * self.num_nodes * self.num_node_feature, dropout, num_nodes, F.relu)
        # self.conv1 = NNConv(self.num_nodes * self.num_node_feature, out_channel, self.nn_1)  # input features for each node is the number of node itself, as it is the traffic to each possible D
        # self.bn1 = BatchnormDropout(out_channel, dropout) # applies also: ReLU
        # self.pool1 = TopKPooling(out_channel, ratio=pooling_ratio)

        # self.nn_2 = Understanding(num_edges, hidden_size_gnn, out_channel * out_channel, dropout, num_nodes, F.relu)
        # self.conv2 = NNConv(out_channel, out_channel, self.nn_2)  # input features for each node is the number of node itself, as it is the traffic to each possible D
        # self.bn2 = BatchnormDropout(out_channel, dropout)
        # self.pool2 = TopKPooling(out_channel, ratio=pooling_ratio)

        """
        self.lin1 = torch.nn.Linear(2 * out_channel, 512)
        self.bn4 = BatchnormDropout(512, dropout)
        self.lin2 = torch.nn.Linear(512, 256)
        self.bn5 = BatchnormDropout(256, dropout)
        self.lin3 = torch.nn.Linear(256, self.num_nodes ** 2)
        self.final_act = F.relu
        """

        self.conv1 = SAGEConv(self.num_nodes* self.num_node_feature, hidden_size_gnn)
        self.pool1 = TopKPooling(hidden_size_gnn, ratio=0.8)
        self.conv2 = SAGEConv(hidden_size_gnn, hidden_size_gnn)
        self.pool2 = TopKPooling(hidden_size_gnn, ratio=0.8)
        self.conv3 = SAGEConv(hidden_size_gnn, hidden_size_gnn)
        self.pool3 = TopKPooling(hidden_size_gnn, ratio=0.8)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.num_nodes ** 2)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.relu(self.lin3(x)).squeeze(1)

        return x

    def save_model(self, dict_output, path):
        torch.save(dict_output, path)
