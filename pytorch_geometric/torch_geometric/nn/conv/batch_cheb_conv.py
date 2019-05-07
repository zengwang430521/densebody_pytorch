import torch
from torch.nn import Parameter
from torch_sparse import spmm
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops

from ..inits import uniform


class FixBatchChebConv(torch.nn.Module):
    """
    the implemention of batch cheb conv, the input must be in the same graph structure
    edge_index should be bidirectional： if node 0 and 1 is linked, then both (0,1) and (1,0) should appear in edge_index
    """

    def __init__(self, in_channels, out_channels, K, num_nodes, edge_index, edge_weight=None, bias=True):
        super(FixBatchChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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
        self.K = K
        self.lap = lap
        self.edge_index = edge_index
        self.num_nodes = num_nodes

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
            Tx_1 = spmm(edge_index, lap, num_nodes, x.permute(1, 0, 2).reshape(num_nodes, -1))
            Tx_1 = Tx_1.reshape(num_nodes, -1, self.in_channels).permute(1, 0, 2)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):
            # Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1.permute(1, 0, 2).reshape(num_nodes, -1))
            Tx_2 = Tx_2.reshape(num_nodes, -1, self.in_channels).permute(1, 0, 2) - Tx_0

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


def perm_data_back(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    form the clustering tree to original one
    """
    if indices is None:
        return x

    N, M = x.shape
    xnew = np.empty((N, M))
    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, j] = x[:, i]
    return xnew


# change the index
def perm_data(x, indices):
    '''
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    '''
    if indices is None:
        return x


    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i] = x[:,j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:,i] = np.zeros(N)
    return xnew