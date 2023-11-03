from this import d
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from unsupervised.convs import GCNConv
from unsupervised.convs import SAGEConv
from unsupervised.convs import GATConv
from unsupervised.convs import GATv2Conv
from unsupervised.convs.wgin_conv import WGINConv


class UniEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard",
                 is_infograph=False):
        super(UniEncoder, self).__init__()

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

        
        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                #conv = GCNConv(emb_dim, emb_dim)
                #nn = Linear(emb_dim, emb_dim)
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                #conv = GCNConv(num_dataset_features, emb_dim)
                #nn = Linear(num_dataset_features, emb_dim)
            conv = WGINConv(nn)

            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, batch, x, edge_index, edge_attr, edge_weight, batch_aug_edge_weight):
        xs = []
        print("batch_aug_edge_weight:", batch_aug_edge_weight.shape)
        bernoulli_weight = batch_aug_edge_weight[2,:]
        #split = batch.tolist()
        adj = edge_index[0,:].tolist()
        bernoulli_adj = batch_aug_edge_weight[0,:].tolist()
        #idce_batch = []
        idce_adj = []
        best_adj = []
    
        for idx in range(batch[-1]):
            idc_adj = adj.index((idx+1)*116)
            idce_adj.append(idc_adj)
            idc_adj = bernoulli_adj.index((idx+1)*116)
            best_adj.append(idc_adj)
        
        #print("edge_index[:,-1]:", edge_index[:,-1])
        for i in range(len(idce_adj)+1):
            if i == len(idce_adj):
                data = edge_weight[idce_adj[-1]:]
                best = bernoulli_weight[best_adj[-1]:]
                temp = edge_index[:,idce_adj[-1]:]
                best_matrix = batch_aug_edge_weight[:2,best_adj[-1]:]
            elif i == 0:
                data = edge_weight[:idce_adj[i]]
                best = bernoulli_weight[:best_adj[i]]
                temp = edge_index[:,:idce_adj[i]]
                best_matrix = batch_aug_edge_weight[:2,:best_adj[i]]
            else:
                data = edge_weight[idce_adj[i-1]:idce_adj[i]]
                best = bernoulli_weight[best_adj[i-1]:best_adj[i]]
                temp = edge_index[:,idce_adj[i-1]:idce_adj[i]]
                best_matrix = batch_aug_edge_weight[:2,best_adj[i-1]:best_adj[i]]

            #print("i:", i)
            temp = temp.cpu().numpy()
            data = data.cpu().numpy()
            dif = 116*i*np.ones(temp[0,:].shape)
            row = temp[0,:] - dif
            col = temp[1,:] - dif
            graph = coo_matrix((data, (row, col)), shape=(116, 116)).toarray()

            data = best.cpu().numpy()
            best_matrix = best_matrix.cpu().numpy()
            dif = 116*i*np.ones(best_matrix[0,:].shape)
            row = best_matrix[0,:] - dif
            col = best_matrix[1,:] - dif
            best_graph = coo_matrix((data, (row, col)), shape=(116, 116)).toarray()
                        
            #print("best_graph:", best_graph[:3,:3])

            subgraph = np.multiply(graph, best_graph)
            
            #subgraph = F.normalize(subgraph, dim=1)
            #subgraph = normalize(subgraph, axis=1)
            sp_subgraph = sp.coo_matrix(subgraph)
            sp_weight = sp_subgraph.data
            subgraph = torch.from_numpy(subgraph).cuda()
            subgraph = subgraph.nonzero(as_tuple=False).t().contiguous()
            #sp_weight = torch.Tensor(sp_subgraph.data)
            sp_weight = torch.from_numpy(sp_weight).cuda()
            
            if i == 0:
                edge_index_l = subgraph
                edge_weight_l = sp_weight
            else:
                #print("edge_index_l:", edge_index_l.shape)
                #print("subgraph:", subgraph.shape)    
                edge_index_l = torch.cat([edge_index_l, subgraph], -1)
                edge_weight_l = torch.cat([edge_weight_l, sp_weight], -1)

        for i in range(self.num_gc_layers):
            #x = self.convs[i](x, edge_index, edge_weight)
            #print("x, edge_index_l, edge_weight_l:", x.is_cuda, edge_index_l.is_cuda, edge_weight_l.is_cuda)
            
            x = self.convs[i](x, edge_index_l, edge_weight_l)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        # compute graph embedding using pooling
        ##
        #x = F.normalize(x, dim=1)
        
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