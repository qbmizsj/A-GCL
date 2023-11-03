import torch
from torch.nn import Sequential, Linear, ReLU
##############
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(self, encoder, proj_hidden_dim=300, nclass: int = 2):
        super(Classifier, self).__init__()

        ##############
        self.dim1 = 8
        self.dim2 = 8
        ##############

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        #self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True), 
                                    #Linear(proj_hidden_dim, proj_hidden_dim), ReLU(inplace=True))

        ##############
        self.fc1 = torch.nn.Linear(proj_hidden_dim, self.dim1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        #self.fc2 = torch.nn.Linear(self.dim1, self.dim2)
        #self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        #self.fc3 = torch.nn.Linear(self.dim2, nclass)
        self.fc3 = torch.nn.Linear(self.dim2, nclass)
        ##############

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):

        z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

        #z = self.proj_head(z)
        # z shape -> Batch x proj_hidden_dim

        ##############
        z = self.bn1(F.relu(self.fc1(z)))
        z = F.dropout(z, p=0.5, training=self.training)
        #z = self.bn2(F.relu(self.fc2(z)))
        #z = F.dropout(z, p=0.5, training=self.training)
        output = F.log_softmax(self.fc3(z), dim=-1)
        ##############

        return output, node_emb

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        # x and x_aug shape -> Batch x proj_hidden_dim

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

        return loss
