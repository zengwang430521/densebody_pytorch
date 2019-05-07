from pytorch_geometric.torch_geometric.nn import ChebConv
import pytorch_geometric.torch_geometric.nn.conv.batch_cheb_conv as BCGN
import torch

net = ChebConv(in_channels=3, out_channels=4, K=3)
x = torch.rand([11, 3])
edges = torch.randint(low=0, high=11, size=[2, 10]).long()
y = net(x, edges)

x = torch.rand([8, 11, 3])
edges = torch.randint(low=0, high=11, size=[2, 10]).long()
batch_net = BCGN.FixBatchChebConv(in_channels=3, out_channels=4, K=3, num_nodes=11, edge_index=edges)
y = batch_net(x)
y = 1

