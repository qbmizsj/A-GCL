import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs import GCNConv
from unsupervised.convs import SAGEConv
from unsupervised.convs import GATConv
from unsupervised.convs.wgin_conv import WGINConv
from unsupervised.convs import GraphConv



class GATv2Encoder(torch.nn.Module):
    def __init__(self, num_dataset_features, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard",
                 is_infograph=False):
        super(GATv2Encoder, self).__init__()

        self.pooling_type = pooling_type
        # emb_dim -> out_channels
        # num_dataset_features-> in_channels
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.fc1 = torch.nn.Linear(emb_dim, 8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.fc2 = torch.nn.Linear(8, 2)
        #
        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                conv = GCNConv(emb_dim, emb_dim)
                #nn = Linear(emb_dim, emb_dim)
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                #nn = Linear(num_dataset_features, emb_dim)
                conv = GCNConv(num_dataset_features, emb_dim)
            #conv = WGINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        xs = []
        for i in range(self.num_gc_layers):
            ##
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        #x = F.normalize(x, dim=1)
        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            return xpool, x

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(batch, x, edge_index, None, edge_weight)
                ###
                x = self.bn1(F.relu(self.fc1(x)))
                x = F.dropout(x, p=0.5, training=self.training)
                x = F.log_softmax(self.fc2(x), dim=-1)

                pred = x.max(dim=1)[1]
                x = pred
                ###
                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
