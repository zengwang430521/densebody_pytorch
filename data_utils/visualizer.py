from .smpl_torch_batch import SMPLModel
from .uv_map_generator import UV_Map_Generator
import os
from cv2 import imwrite
import torch
import numpy as np
from skimage.draw import circle
import pickle
from . import objfile

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer():
    def __init__(self, opt):
        os.chdir(opt.project_root + '/data_utils')
        self.UV_sampler = UV_Map_Generator(
            UV_height=opt.im_size,
            UV_pickle=opt.uv_prefix+'.pickle'
        )
        # Only use save obj 
        self.model = SMPLModel(
            device=None,
            model_path = './model_lsp.pkl',
        )
        os.chdir(opt.project_root)
        if opt.phase == 'train':
            self.save_root = '{}/{}/visuals/'.format(opt.checkpoints_dir, opt.name)
        elif opt.phase == 'test':
            self.save_root = '{}/{}/visuals/'.format(opt.results_dir, opt.name)
        else:
            self.save_root = '{}/{}/{}_in_the_wild/'.format(opt.results_dir, opt.name, opt.dataset)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)
    
    @staticmethod
    def tensor2im(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy uint8 [0,255] (HWC)
        return ((tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) * 127.5).astype(np.uint8)
    
    @staticmethod    
    def tensor2numpy(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy float [0,1] (HWC)
        return (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) / 2.
    
    def save_results(self, visual_dict, epoch, batch):
        img_name = self.save_root + '{:03d}_{:05d}.png'.format(epoch, batch)
        obj_name = self.save_root + '{:03d}_{:05d}.obj'.format(epoch, batch)
        ply_name = self.save_root + '{:03d}_{:05d}.ply'.format(epoch, batch)
        imwrite(img_name, 
            self.tensor2im(torch.cat([im for im in visual_dict.values()], dim=2))
        )
        fake_UV = visual_dict['fake_UV']
        resampled_verts = self.UV_sampler.resample(self.tensor2numpy(fake_UV))
        self.UV_sampler.write_ply(ply_name, resampled_verts)
        self.model.write_obj(resampled_verts, obj_name)


class MeshVisualizer():
    def __init__(self, opt):
        os.chdir(opt.project_root + '/data_utils')

        # Only use save obj
        model_path = './model_lsp.pkl'
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        # 20190330: lsp 14 joint regressor
        self.joint_regressor = torch.from_numpy(np.array(params['joint_regressor'].T.todense())).float()
        self.faces = params['f']

        os.chdir(opt.project_root)
        if opt.phase == 'train':
            self.save_root = '{}/{}/visuals/'.format(opt.checkpoints_dir, opt.name)
        elif opt.phase == 'test':
            self.save_root = '{}/{}/visuals/'.format(opt.results_dir, opt.name)
        else:
            self.save_root = '{}/{}/{}_in_the_wild/'.format(opt.results_dir, opt.name, opt.dataset)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)

    @staticmethod
    def tensor2im(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy uint8 [0,255] (HWC)
        return ((tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) * 127.5).astype(np.uint8)

    @staticmethod
    def tensor2numpy(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy float [0,1] (HWC)
        return (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) / 2.

    def overlap(self, im, mesh, joints):
        shape = im.shape[0:2]
        height = im.shape[0]

        for p2d in mesh:
            im[height -1 - p2d[1], p2d[0]] = [127, 127, 127]

        for j2d in joints:
            rr, cc = circle(height -1 - j2d[1], j2d[0], 2, shape)
            im[rr, cc] = [255, 0, 0]
        return im

    def vis_vertices(self, figname, vertices_real, vertices_fake):
        # make sure that vertex is in shape vertex_num * 3
        if vertices_real.shape[1] != 3: vertices_real = vertices_real.transpose(1, 0)
        if vertices_fake.shape[1] != 3: vertices_fake = vertices_fake.transpose(1, 0)

        # transform vertex to numpy type
        vertices_real = np.array(vertices_real)
        vertices_fake = np.array(vertices_fake)

        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        # fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2)
        ax.scatter(vertices_real[:, 0], vertices_real[:, 1], vertices_real[:, 2], c='b', s=1, alpha=0.5)
        ax.scatter(vertices_fake[:, 0], vertices_fake[:, 1], vertices_fake[:, 2], c='r', s=1, alpha=0.5)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')

        ax.view_init(elev=90, azim=-90)

        '''
        ver_min = np.concatenate((vertices_real, vertices_fake), axis=0).min(axis=0)
        ver_max = np.concatenate((vertices_real, vertices_fake), axis=0).max(axis=0)
        plt.xlim(ver_min[0], ver_max[0])
        plt.ylim(ver_min[1], ver_max[1])
        '''

        plt.xlim(-15, 270)
        plt.ylim(-15, 270)

        plt.draw()

        plt.savefig(figname)
        plt.close()

    def save_results(self, data, epoch, batch):
        image = data['real_image'].detach().cpu()
        vertices_real = data['real_mesh'].detach().cpu()
        vertices_fake = data['fake_mesh'].detach().cpu()

        img_name = self.save_root + '{:03d}_{:05d}.png'.format(epoch, batch)
        obj_name = self.save_root + '{:03d}_{:05d}.obj'.format(epoch, batch)
        fig_name = self.save_root + '{:03d}_{:05d}_mesh.png'.format(epoch, batch)

        self.vis_vertices(fig_name, vertices_real, vertices_fake)
        objfile.write_obj(obj_name, vertices_fake, self.faces)

        visual_list = []
        im = self.tensor2im(image)
        visual_list.append(im)

        joints_fake = torch.mm(self.joint_regressor.permute(1, 0), vertices_fake)[:14]
        mesh_2d = vertices_fake[:, :-1].numpy().astype(np.uint8)
        joints_2d = joints_fake[:, :-1].numpy().astype(np.uint8)
        im_lap_f = self.overlap(im.copy(), mesh_2d, joints_2d)
        visual_list.append(im_lap_f)

        joints_real = torch.mm(self.joint_regressor.permute(1, 0), vertices_real)[:14]
        mesh_2d = vertices_real[:, :-1].numpy().astype(np.uint8)
        joints_2d = joints_real[:, :-1].numpy().astype(np.uint8)
        im_lap_r = self.overlap(im.copy(), mesh_2d, joints_2d)
        visual_list.append(im_lap_r)

        imwrite(img_name, np.concatenate(visual_list, axis=1))

