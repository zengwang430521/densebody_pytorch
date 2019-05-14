import torch
import torch.nn as nn
import torch.nn.parallel

from .base_model import BaseModel
from . import networks
from . import graph_networks


class GCNModel(BaseModel):
    def name(self):
        return 'GCNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt, paras):
        BaseModel.initialize(self, opt)
        self.loss_names = ['L1', 'TV']
        self.model_names = ['encoder', 'decoder']
        self.ngpu = opt.ngpu
        self.decoder = graph_networks.define_decoder(netD=opt.netD, paras=paras,
                                                     channels=opt.gcn_channels, nz=opt.nz, norm=opt.norm, nl=opt.nl,
                                                     init_type=opt.init_type, device=self.device)

        self.encoder = networks.define_encoder(opt.im_size, opt.nz, opt.nchannels, netE=opt.netE, ndown=opt.ndown,
                                               norm=opt.norm, nl=opt.nl, init_type=opt.init_type, device=self.device)

        print(self.decoder.state_dict().keys())
        # params = list(self.decoder.named_parameters())

        if opt.phase == 'train':
            self.L1_loss = torch.nn.L1Loss()
            # self.TV_loss = networks.TotalVariationLoss(opt.uv_prefix, self.device)
            self.encoder.train()
            self.decoder.train()
            self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_enc)
            self.optimizers.append(self.optimizer_dec)

        else:
            self.encoder.eval()
            self.decoder.eval()

    '''
        forward: Train one step, return loss calues
    '''

    def train_one_batch(self, data):
        image = data[0].to(self.device)
        mesh_gt = data[1].to(self.device)

        if image.is_cuda and self.ngpu > 1:
            mesh_re = nn.parallel.data_parallel(self.encoder, image, range(self.ngpu))
            mesh_re = nn.parallel.data_parallel(self.decoder, mesh_re, range(self.ngpu))
        else:
            mesh_re = self.decoder(self.encoder(image))

        mesh_re = mesh_re * 255.0
        l1_loss = self.L1_loss(mesh_gt / 255.0, mesh_re / 255.0)
        total_loss = l1_loss  # + self.opt.tv_weight * tv_loss

        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        total_loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()

        # for visualize
        self.real_image = image[0].detach()
        self.mesh_gt = mesh_gt[0].detach()
        self.mesh_re = mesh_re[0].detach()

        return {
            'l1': l1_loss.item(),
            'total': total_loss.item()
        }

    def get_current_visuals(self):
        # return: real image, overlap image, real mesh, fake mesh,
        visuals = {
            'real_image': self.real_image,
            'real_mesh': self.mesh_gt,
            'fake_mesh': self.mesh_re,
        }
        return visuals
