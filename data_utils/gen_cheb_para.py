import pytorch_geometric.torch_geometric.nn.conv.batch_cheb_conv as BCGN
import torch
from data_utils import objfile
import scipy
from data_utils import coarsening
import pickle
from models import graph_networks

template_path = '../parameter/template.obj'
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

data = {'graphs': graphs,
        'perm': perm,
        'perm_back': perm_back}
file_name = '../parameter/paras_{}.pkl'.format(level)

with open(file_name, 'wb') as f:
    pickle.dump(data, f)

print('graph sampling completed.')
