import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import args
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)**0.5
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias
class self_loop_attention_GCN(nn.Module):
    def __init__(self, nfeat, nhead, dropout=0.05, ratio=1):
        super(self_loop_attention_GCN, self).__init__()
        nhid = nfeat * ratio
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.nhead = nhead
        self.a = Parameter(torch.FloatTensor(nhead, nfeat, 1))
        self.reset_parameters()
    def laplacian_matrix(self, A, features):
        D = torch.diag(A.sum(1))
        D_hat = D ** (-0.5)
        D_hat[torch.isinf(D_hat)] = 0
        I = torch.eye(A.shape[0], device=A.device)
        expanded_features = torch.unsqueeze(features, 0)
        attention_scores = [F.softmax(torch.matmul(expanded_features, self.a[i]), dim=2) for i in range(self.nhead)]
        combined_attention_score = torch.mean(torch.cat(attention_scores, dim=0), dim=0)
        enhanced_attention = torch.diag(combined_attention_score.squeeze(-1)) 
        return D_hat @ (A + enhanced_attention + I) @ D_hat
    def reset_parameters(self):
        if self.a is not None:
            stdv = 1. / self.a.size(1) ** 0.5
            self.a.data.uniform_(-stdv, stdv)
    def forward(self, adj, features_matrix):
        L_adj = self.laplacian_matrix(adj, features_matrix)
        x = F.relu(self.gc1(features_matrix, L_adj))   
        x = F.dropout(x, self.dropout, training=self.training)
        return x
