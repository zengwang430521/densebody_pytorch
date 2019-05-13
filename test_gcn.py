from pytorch_geometric.torch_geometric.nn import ChebConv
import pytorch_geometric.torch_geometric.nn.conv.batch_cheb_conv as BCGN
import torch
from data_utils import objfile
import scipy
from data_utils import coarsening
from models import graph_networks


net = ChebConv(in_channels=3, out_channels=4, K=3)

x = torch.rand([11, 3])
edges = torch.randint(low=0, high=11, size=[2, 10]).long()
y = net(x, edges)
print(net.state_dict().keys())

x = torch.rand([8, 11, 3])
edges = torch.randint(low=0, high=11, size=[2, 10]).long()
batch_net = BCGN.FixBatchChebConv(in_channels=3, out_channels=4, K=3, num_nodes=11, edge_index=edges)
params = list(batch_net.named_parameters())

print(batch_net.state_dict().keys())

y = batch_net(x)

template_path = './parameter/template.obj'
vertices, faces = objfile.read_obj(template_path)
adj = BCGN.face2adj(faces, vertices.shape[0])
A = scipy.sparse.csr_matrix(adj)
level = 8
channels = [8 * 2 ** (i//2) for i in range(level+2)]

graphs, perm = coarsening.coarsen(A, levels=level, self_connections=True)
perm = torch.LongTensor(perm)
perm_back = torch.zeros(perm.shape[0], dtype=torch.long)
temp = torch.LongTensor(range(perm.shape[0]))
perm_back[perm] = temp

x = torch.Tensor(range(A.shape[0])).unsqueeze(0).unsqueeze(2)
y = BCGN.perm_data(x, perm)
x_back = BCGN.perm_data(y, perm_back)
temp = x_back[:, :x.shape[1], :] - x
decoder = graph_networks.GraphDecoder(adjs=graphs[::-1], channels=channels[::-1], nz=256)

x = torch.ones(2, 256)
y = decoder(x)

temp = 1