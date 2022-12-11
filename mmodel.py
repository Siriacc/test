import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNConv
import torch_sparse

class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
        super(FastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        fastGTNs = []
        fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))

        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(args.node_dim, num_class)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, A, X, target_x, target, num_nodes=None, epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        H_, Ws = self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        y = self.linear(H_[target_x])
        loss = self.loss(y, target.squeeze())
        return loss, y, Ws

class FastGTN(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None):
        super(FastGTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = args.num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        args.w_in = w_in
        self.w_out = args.node_dim
        self.num_class = num_class
        self.num_layers = args.num_layers

        layers = []
        layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, args=args))
        self.layers = nn.ModuleList(layers)

        self.Ws = []
        for i in range(self.num_channels):
            self.Ws.append(GCNConv(in_channels=self.w_in, out_channels=self.w_out).weight)
        self.Ws = nn.ParameterList(self.Ws)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)

    def forward(self, A, X, num_nodes, epoch=None):
        Ws = []
        X_ = [X@W for W in self.Ws]
        H = [X@W for W in self.Ws]

        for i in range(self.num_layers):
            H, W = self.layers[i](H, A, num_nodes, epoch=epoch, layer=i+1)
            Ws.append(W)

        for i in range(self.num_channels):
            if i==0:
                H_ = F.relu(H[i])
            else:
                H_ = torch.cat((H_, F.relu(H[i])), dim=1) # Question:H_维度是多少？

        H_ = F.relu(self.linear1(H_))
        return H_, Ws

class FastGTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, args=None):
        super(FastGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args)
        self.args = args

    def forward(self, H_, A, num_nodes, epoch=None, layer=None):
        result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
        W =[W1]
        Hs = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            # mat_a是一个8994*8994的稀疏矩阵; H [8994,64]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            H = torch.sparse.mm(mat_a, H_[i]) # 稀疏矩阵乘法
            Hs.append(H)
        return Hs, W

class FastGTConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, args=None, pre_trained=None):
        super(FastGTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.scale = nn.Parameter((torch.Tensor([0.1]), requires_grad=False))
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.1)

    def forward(self, A, num_nodes, epoch=None, layer=None):
        weight = self.weight
        filter = F.softmax(weight, dim=1) # α(k)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j ==0 :
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            # 矩阵压缩，相同位置的值相加
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))

        return results, filter
























