"""
Evaluation of H3.6M.
Sample call from hmr:
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=<model_to_eval>
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=/home/kanazawa/projects/hmr_v2/models/model.ckpt-667589
"""
from argparse import ArgumentParser
import itertools
from time import time
from os.path import exists, join, expanduser, split
import os
from torchvision import transforms
from PIL import Image
from cv2 import imread
from evals.test_dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import create_model
from data_utils.uv_map_generator import UV_Map_Generator
from data_utils.smpl_torch_batch import SMPLModel



# -- Utils ---
def get_h36m_seqs(protocol=2):
    action_names = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WakingDog', 'WalkTogether'
    ]
    print('Protocol %d!!' % protocol)
    if protocol == 2:
        trial_ids = [0]
        cam_ids = [3]
    else:
        trial_ids = [0, 1]
        cam_ids = range(0, 4)

    sub_ids = [9, 11]
    all_pairs = [
        p
        for p in list(
            itertools.product(*[sub_ids, action_names, trial_ids, cam_ids]))
    ]
    # Corrupt mp4 file
    all_pairs = [p for p in all_pairs if p != (11, 'Directions', 1, 0)]

    return all_pairs, action_names


def tensor2numpy(tensor):
    # input: cuda tensor (CHW) [-1,1]; output: numpy float [0,1] (HWC)
    return (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) / 2.

im_trans = transforms.Compose([
    Image.fromarray,
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])


def gen_sequence(model, UV_sampler, smpl, seq_info, save_dir, opt):

    sub_id, action, trial_id, cam_id = seq_info
    file_seq_name = 'S%d_%s_%d_cam%01d' % (sub_id, action, trial_id, cam_id)
    print('%s' % (file_seq_name))

    save_path = join(save_dir, file_seq_name)
    if exists(save_path):
        pass
    else:
        os.makedirs(save_path)
        # Run the model!
        t0 = time()
        dataset = ImageDataset(opt.data_root + '/test/' + file_seq_name)
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers), pin_memory=True)
        data_stream = tqdm(data_loader)
        for images, names in data_stream:
            images = images.to(model.device)
            UVs = model.forward(images)
            for i in range(len(UVs)):
                UV = UVs[i]
                vert = UV_sampler.resample(tensor2numpy(UV))
                image_name = names[i]
                temp = image_name.split('/')
                obj_name = temp[-1].split('.')[0] + '.obj'
                smpl.write_obj(vert, os.path.join(save_path, obj_name))
        t1 = time()
        print(t1-t0)
    return 0


def generate_obj(opt):
    # load the model
    opt.phase = 'test'
    model = create_model(opt)
    model.setup(opt)

    o = os.getcwd()
    d = os.path.dirname(__file__) + '/../data_utils'
    os.chdir(d)

    UV_sampler = UV_Map_Generator(
        UV_height=opt.im_size,
        UV_pickle=opt.uv_prefix + '.pickle'
    )

    smpl = SMPLModel(
        device=None,
        model_path='./model_lsp.pkl',
    )

    os.chdir(o)

    # Figure out the save name.
    protocol = opt.protocol
    all_pairs, actions = get_h36m_seqs(protocol)

    for itr, seq_info in enumerate(all_pairs):
        print('%d/%d' % (itr, len(all_pairs)))
        gen_sequence(model, UV_sampler, smpl, seq_info, opt.save_root, opt)


def TrainOptions(debug=False):
    parser = ArgumentParser()

    # dataset options
    # platform specific options
    data_root = '/home/wzeng/mydata/h3.6m'  # change to you dir
    batch_size = 2

    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--save_root', type=str, default='../../results')
    parser.add_argument('--checkpoints_dir', type=str, default='../../checkpoints')
    parser.add_argument('--name', type=str, default='densebody_resnet_h36m')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

    parser.add_argument('--uv_map', type=str, default='smpl_fbx',
                        choices=['radvani', 'vbml_close', 'vbml_spaced', 'smpl_fbx'])
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--protocol', type=int, default='2')

    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    # model options
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vggnet', 'mobilenet'])
    parser.add_argument('--netD', type=str, default='convres', choices=['convres', 'conv-up'])
    parser.add_argument('--nz', type=int, default=256, help='latent dims')
    parser.add_argument('--ndown', type=int, default=6, help='downsample times')
    parser.add_argument('--nchannels', type=int, default=64, help='conv channels')
    parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
    parser.add_argument('--nl', type=str, default='relu', choices=['relu', 'lrelu', 'elu'])
    parser.add_argument('--init_type', type=str, default='xavier',
                        choices=['xavier', 'normal', 'kaiming', 'orthogonal'])
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

    # training options

    # optimization options
    opt = parser.parse_args()

    opt.uv_prefix = opt.uv_map + '_template'
    opt.save_root = os.path.join(opt.save_root, opt.name)
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


if __name__ == '__main__':
    opt = TrainOptions(debug=False)
    generate_obj(opt)