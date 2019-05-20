from data_utils.densebody_dataset import DenseBodyDataset
from data_utils.mesh_dataset import MeshDataset
from data_utils import visualizer as vis
from models import create_model
import sys
from sys import platform
import os
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from models.gcn_model import GCNModel
import torch
import pickle
import time
from torch.utils.data import DataLoader

# default options
def TrainOptions(debug=False):
    parser = ArgumentParser()

    # platform specific options
    windows_root = 'D:/data'
    linux_root = '/home/wzeng/mydata/h3.6m'  # change to you dir
    data_root = linux_root if platform == 'linux' else windows_root
    num_threads = 4 if platform == 'linux' else 0
    batch_size = 16 if platform == 'linux' else 4
    folder_name = 'images_washed'


    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints')
    parser.add_argument('--image_folder', type=str, default=folder_name)  # the name of the folder
    parser.add_argument('--max_dataset_size', type=int, default=-1)
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--vertex_num', type=int, default=6890, help='vertex number of template')
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--name', type=str, default='resnet_coma_h36m')
    parser.add_argument('--num_threads', default=num_threads, type=int, help='# sthreads for loading data')

    # model options
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn'])
    parser.add_argument('--netE', type=str, default='resnet', choices=['resnet', 'vggnet', 'mobilenet'])
    parser.add_argument('--netD', type=str, default='coma-gcn', choices=['cheb-gcn', 'coma-gcn'])
    parser.add_argument('--nz', type=int, default=256, help='latent dims')
    parser.add_argument('--level', type=int, default=4, help='level of decoder')
    parser.add_argument('--gcn_channels', type=list, default=None, help='the channels of gcn decoder')
    parser.add_argument('--ndown', type=int, default=6, help='downsample times')
    parser.add_argument('--nchannels', type=int, default=64, help='conv channels')
    parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
    parser.add_argument('--nl', type=str, default='lrelu', choices=['relu', 'lrelu', 'elu'])
    parser.add_argument('--init_type', type=str, default='xavier',
                        choices=['xavier', 'normal', 'kaiming', 'orthogonal'])

    # training options
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')

    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=50,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--save_result_freq', type=int, default=500,
                        help='frequency of showing training results on screen')
    parser.add_argument('--save_epoch_freq', type=int, default=2,
                        help='frequency of saving checkpoints at the end of epochs')

    # optimization options
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='plateau', choices=['lambda', 'step', 'plateau'])
    parser.add_argument('--lr_decay_iters', type=int, default=100,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--tv_weight', type=float, default=10, help='toal variation loss weights')
    opt = parser.parse_args()

    opt.project_root = os.path.dirname(os.path.realpath(__file__))

    if debug:
        opt.batch_size = 2
        opt.save_result_freq = 2
        opt.save_epoch_freq = 1
        opt.max_dataset_size = 10
        opt.num_threads = 0
        opt.niter = 2
        opt.niter_decay = 2

    return opt


def get_parameters(opt):
    if opt.netD == 'cheb-gcn':
        para_file = './parameter/paras_{}.pkl'.format(opt.level)
        with open(para_file, 'rb') as f:
            paras = pickle.load(f)

    elif opt.netD == 'coma-gcn':
        para_file = './parameter/paras_coma_{}.pkl'.format(opt.level)
        with open(para_file, 'rb') as f:
            paras = pickle.load(f, encoding='latin1')
    else:
        raise NotImplementedError('Decoder model name [%s] is not recognized' % opt.netD)

    return paras

#
# sys.path.append('{}/models'.format(project_root))
# sys.path.append('{}/data_utils'.format(project_root))

if __name__ == '__main__':
    # Change this to your gpu id.
    # The program is fixed to run on a single GPU

    np.random.seed(9608)
    opt = TrainOptions(debug=False)
    dataset = MeshDataset(data_root=opt.data_root, image_folder=opt.image_folder, phase=opt.phase, max_size=opt.max_dataset_size,)
    batchs_per_epoch = len(dataset) // opt.batch_size  # drop last batch
    print('#training images = %d' % len(dataset))

    # load graph parameters
    if opt.gcn_channels == None:
        opt.gcn_channels = [min(16 * 2 ** (i // 1), 64) for i in range(opt.level + 2)][::-1]
        # opt.gcn_channels = [8 * 2 ** (i // 1) for i in range(opt.level + 2)][::-1]

    paras = get_parameters(opt)

    model = GCNModel()
    model.initialize(opt, paras)
    model.setup(opt)

    visualizer = vis.MeshVisualizer(opt)
    total_steps = 0

    # put something in txt file
    file_log = open(os.path.join(opt.checkpoints_dir, opt.name, 'log.txt'), 'w')

    for epoch in range(opt.load_epoch + 1, opt.niter + opt.niter_decay + 1):
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        data_stream = tqdm(data_loader, ncols=120)
        batch_len = data_loader.__len__()
        print('Epoch %d: start training' % epoch)
        loss_metrics = 0
        i = 0
        for data in data_stream:
            # give up the last batch for BN layers
            if len(data[0]) < opt.batch_size:
                continue
            i += 1
            loss_dict = model.train_one_batch(data)
            loss_metrics = loss_dict['total']

            # change tqdm info
            tqdm_info = '%d/%d ' % (i, batch_len)
            for k, v in loss_dict.items():
                tqdm_info += ' %s: %.6f' % (k, v)
            data_stream.set_description(tqdm_info)

            if (i + 1) % opt.save_result_freq == 0:
                file_log.write('epoch {} iter {}: {}\n'.format(epoch, i, tqdm_info))
                file_log.flush()
                visualizer.save_results(model.get_current_visuals(), epoch, i)

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
        print('Epoch %d training finished' % epoch)
        if epoch > opt.niter:
            model.update_learning_rate(metrics=loss_metrics)

    '''
    rand_perm = np.arange(batchs_per_epoch)

    for epoch in range(opt.load_epoch + 1, opt.niter + opt.niter_decay + 1):
 
        # set loop information
        print('Epoch %d: start training' % epoch)
        np.random.shuffle(rand_perm)
        loop = tqdm(range(batchs_per_epoch), ncols=120)
        loss_metrics = 0
        for i in loop:
            start = time.time()

            data = dataset[rand_perm[i] * opt.batch_size: (rand_perm[i] + 1) * opt.batch_size]

            end = time.time()
            time1 = end - start

            torch.cuda.synchronize()
            start = time.time()

            loss_dict = model.train_one_batch(data)

            torch.cuda.synchronize()
            end = time.time()
            time2 = end-start



            loss_metrics = loss_dict['total']
            # change tqdm info
            tqdm_info = ''
            for k, v in loss_dict.items():
                tqdm_info += ' %s: %.6f' % (k, v)
            tqdm_info += '  loading data:{}, train:{}'.format(time1, time2)
            loop.set_description(tqdm_info)

            if (i + 1) % opt.save_result_freq == 0:
                file_log.write('epoch {} iter {}: {}\n'.format(epoch, i, tqdm_info))
                file_log.flush()
                visualizer.save_results(model.get_current_visuals(), epoch, i)

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
        print('Epoch %d training finished' % epoch)
        if epoch > opt.niter:
            model.update_learning_rate(metrics=loss_metrics)
    '''
    file_log.close()

