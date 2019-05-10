import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import os
import numpy as np
from cv2 import imread, imwrite, connectedComponents
from pytorch_geometric.torch_geometric.nn import ChebConv
import pytorch_geometric.torch_geometric.nn.conv.batch_cheb_conv as BCGN



#####################
#   Initializers    #
#####################

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', device=torch.device('cuda')):
    init_weights(net, init_type)
    return net.to(device)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        # norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        norm_layer = nn.BatchNorm2d
    elif layer_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = nn.ReLU(inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = nn.LeakyReLU(0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = nn.ELU(inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_decoder(netD, adjs, perm_back, vertex_num, channels, nz, norm='batch', nl='lrelu', init_type='xavier', device=None):
    net = None
    if netD == 'cheb-gcn':
        net = ChebGcnDecoder(adjs, perm_back, vertex_num, channels, nz, norm, nl, device)
    else:
        raise NotImplementedError('Decoder model name [%s] is not recognized' % netD)

    return init_net(net, init_type, device)




#####################
#      Networks     #
#####################

#####  Decoder #####

class ChebGcnDecoder(nn.Module):
    def __init__(self, adjs, perm_back, vertex_num, channels, nz,  norm='batch', nl='lrelu', device=None):
        super(ChebGcnDecoder, self).__init__()
        assert (len(adjs) + 1) == len(channels)
        device = device if device is not None else torch.device('cpu')
        self.device = device
        self.vertex_num = vertex_num
        self.graph_size = [adj.shape[0] for adj in adjs]
        self.channels = channels
        self.perm_back = perm_back.to(self.device)

        fc_out = self.graph_size[0] * channels[0]
        fc_dim = nz * 4
        self.fc = nn.Sequential(
            nn.Linear(nz, fc_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc_dim, fc_out),
        ).to(device)

        layers = []
        for i in range(len(adjs)):
            adj = adjs[i]
            num_nodes = self.graph_size[i]
            edge_index = BCGN.adj2edge(adj)
            in_channels = channels[i]
            out_channels = channels[i + 1]

            layers.append(BCGN.FixBatchChebConv(in_channels=in_channels, out_channels=out_channels, K=3,
                                                num_nodes=num_nodes, edge_index=edge_index, device=device))
            layers.append(get_non_linearity(layer_type=nl))

            if i < len(adjs) - 1:
                layers.append(BCGN.Upsample(scale_factor=2))
            else:
                layers.append(BCGN.FixBatchChebConv(in_channels=out_channels, out_channels=3, K=3, num_nodes=num_nodes,
                                                    edge_index=edge_index, device=device))

        self.conv = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.graph_size[0], -1)
        x = self.conv(x)
        x = BCGN.perm_data(x, self.perm_back)
        x = x[:, :self.vertex_num, :]
        return x


