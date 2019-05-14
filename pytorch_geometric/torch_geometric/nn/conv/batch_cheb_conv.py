import torch
from torch.nn import Parameter
from torch_sparse import spmm
from data_utils.batch_spmm import batch_spmm
from torch_geometric.utils import remove_self_loops
import numpy as np

from ..inits import uniform

from torch_scatter import scatter_add



class FixBatchChebConv(torch.nn.Module):
    """
    the implemention of batch cheb conv, the input must be in the same graph structure
    edge_index should be bidirectional： if node 0 and 1 is linked, then both (0,1) and (1,0) should appear in edge_index
    """

    def __init__(self, in_channels, out_channels, K, num_nodes, edge_index, edge_weight=None, bias=True, device=None):
        super(FixBatchChebConv, self).__init__()
        device = device if device is not None else torch.device('cpu')
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            # self.bias = Parameter(torch.Tensor(num_nodes, out_channels))

        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_edges = row.size(0)

        if edge_weight is None:
            edge_weight = torch.ones((num_edges, ))

        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # sparse matrix multiplication is not compatible with multi-gpu, so we use dense mat mul
        '''
        self.K = K
        self.edge_index = edge_index.to(device)
        self.num_nodes = num_nodes
        self.lap = lap.to(device)

        '''
        self.K = K
        self.num_nodes = num_nodes
        lap = torch.sparse_coo_tensor(edge_index, lap, torch.Size([num_nodes, num_nodes])).to_dense()
        if device == torch.device('cuda'):
            self.edge_index = edge_index.cuda()
            self.lap = lap.cuda()


    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x):
        """"""
        K, lap, edge_index, num_nodes = self.K, self.lap, self.edge_index, self.num_nodes
        assert(num_nodes == x.shape[1])
        # Perform filter operation recurrently.
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if K > 1:

            # Tx_1 = spmm(edge_index, lap, num_nodes, x)

            # Tx_1 = spmm(edge_index, lap, num_nodes, x.permute(1, 0, 2).reshape(num_nodes, -1))
            # Tx_1 = Tx_1.reshape(num_nodes, -1, self.in_channels).permute(1, 0, 2)

            # Tx_1 = batch_spmm(edge_index, lap, num_nodes, x)

            # sparse matrix multiplication is not compatible with multi-gpu, so we use dense mat mul
            Tx_1 = torch.matmul(lap, x)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):

            # Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0

            # Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1.permute(1, 0, 2).reshape(num_nodes, -1))
            # Tx_2 = Tx_2.reshape(num_nodes, -1, self.in_channels).permute(1, 0, 2) - Tx_0

            # Tx_2 = 2 * batch_spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0

            # sparse matrix multiplication is not compatible with multi-gpu, so we use dense mat mul
            Tx_2 = 2 * torch.matmul(lap, Tx_1) - Tx_0

            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))



# upsample layer for ChebConv, just like 1D upsample
# the input is in shape B * V * F
class Upsample(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')  # should use nearest sample modde

    def forward(self, x):
        x = x.permute(0, 2, 1)          # from B* V * F to B * F * V
        x = self.up(x)                  # upsample
        x = x.permute(0, 2, 1)          # back to B * scale_factor*V * I
        return x


class ComaUpsample(torch.nn.Module):
    def __init__(self, up_matrix, device=None):
        super().__init__()
        device = device if device is not None else torch.device('cpu')
        self.device = device

        tmp = up_matrix.tocoo()
        self.index = torch.from_numpy(np.stack([tmp.row, tmp.col])).long().to(device)
        self.value = torch.from_numpy(tmp.data).float().to(device)
        self.shape = tmp.shape

    def forward(self, x):
        channels = x.shape[-1]
        x = spmm(self.index, self.value, self.shape[0], x.permute(1, 0, 2).reshape(self.shape[1], -1))
        x = x.reshape(self.shape[0], -1, channels).permute(1, 0, 2)
        return x


def adj2edge(adj):
    edge = adj.nonzero()
    edge = torch.from_numpy(np.stack(edge))
    return edge.long()


def face2adj(faces, vertex_num):
    # get the adj matrix in V * V
    max_node = -1
    for face in faces:
        max_node = max(max_node, max(face))
    offset = -1 if max_node >= vertex_num else 0

    adj = torch.zeros(vertex_num, vertex_num)
    for face in faces:
        for i in range(len(face)):
            j = (i + 1) % len(face)
            adj[face[i] + offset, face[j] + offset] = 1
            adj[face[j] + offset, face[i] + offset] = 1
    return adj

# change the index
def perm_data(x, indices):
    '''
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    '''
    if indices is None:
        return x

    x = torch.cat((x, x.new_zeros([x.shape[0], indices.shape[0] - x.shape[1], x.shape[2]])), dim=1)
    y = x[:, indices, :]
    return y