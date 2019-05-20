"""
Evaluation of H3.6M.
Sample call from hmr:
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=<model_to_eval>
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=/home/kanazawa/projects/hmr_v2/models/model.ckpt-667589
"""
from argparse import ArgumentParser
import itertools
import numpy as np
import deepdish as dd
from time import time
from os.path import exists, join, expanduser, split
import os
from os import makedirs
from evals.eval_util import compute_errors
from data_utils.smpl_torch_batch import SMPLModel
import glob
import torch

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


def evaluate_sequence(seq_info, smpl, opt):
    sub_id, action, trial_id, cam_id = seq_info

    file_seq_name = 'S%d_%s_%d_cam%01d' % (sub_id, action, trial_id, cam_id)
    print('%s' % (file_seq_name))

    gt_path = join(opt.gt_root, file_seq_name)
    gt_names = glob.glob(os.path.join(gt_path, '*.npy'))
    gt_names.sort()
    re_path = join(opt.re_root, file_seq_name)
    save_path = join(opt.re_root, file_seq_name + '_pred.h5')

    if exists(save_path):
        results = dd.io.load(save_path)
        errors = results['errors']
        errors_pa = results['errors_pa']

    else:
        betas = []
        poses = []
        trans = []
        gt3ds = []
        results = {}
        for gt_name in gt_names:
            gt = np.load(gt_name, allow_pickle=True).item()
            gt3ds.append(gt['gt3ds'])

            temp = gt_name.split('/')
            pred_name = temp[-1].split('.')[0] + '.npz'
            if not exists(join(re_path, pred_name)):
                continue
            pred = np.load(join(re_path, pred_name))
            betas.append(pred['shape'])
            poses.append(pred['pose'])
            trans.append(pred['trans'])

        gt3ds = np.stack(gt3ds)
        betas, poses, trans = np.stack(betas), np.stack(poses), np.stack(trans)
        betas, poses, trans = torch.from_numpy(betas).float(), torch.from_numpy(poses).float(), torch.from_numpy(trans).float()
        verts, pred3ds = smpl(betas, poses, trans)
        pred3ds = pred3ds.detach().cpu().numpy()[:, :14, :]

        # Evaluate!
        # Joints 3D is COCOplus format now. First 14 is H36M joints
        # Convert to mm!
        errors, errors_pa = compute_errors(gt3ds * 1000., pred3ds * 1000.)

        results['errors'] = errors
        results['errors_pa'] = errors_pa
        # Save results
        dd.io.save(save_path, results)

    return errors, errors_pa


def TrainOptions(debug=False):
    parser = ArgumentParser()
    parser.add_argument('--gt_root', type=str, default='/home/wzeng/mydata/h3.6m/test')
    parser.add_argument('--re_root', type=str, default='../../fitting_results')
    parser.add_argument('--name', type=str, default='densebody_resnet_h36m')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--protocol', type=int, default='2')

    # optimization options
    opt = parser.parse_args()
    opt.re_root = join(opt.re_root, opt.name)
    opt.project_root = os.path.dirname(os.path.realpath(__file__))
    return opt


def main(config):
    # Figure out the save name.
    smpl = SMPLModel(
        device=None,
        model_path='../data_utils/model_lsp.pkl',
    )

    protocol = config.protocol
    all_pairs, actions = get_h36m_seqs(protocol)

    all_errors = {}
    all_errors_pa = {}
    raw_errors, raw_errors_pa = [], []
    for itr, seq_info in enumerate(all_pairs):
        print('%d/%d' % (itr, len(all_pairs)))
        sub_id, action, trial_id, cam_id = seq_info

        errors, errors_pa = evaluate_sequence(seq_info, smpl, config)

        mean_error = np.mean(errors)
        mean_error_pa = np.mean(errors_pa)
        med_error = np.median(errors)
        raw_errors.append(errors)
        raw_errors_pa.append(errors_pa)
        print('====================')
        print('mean error: %g, median: %g, PA mean: %g' % (mean_error, med_error, mean_error_pa))
        raws = np.hstack(raw_errors)
        raws_pa = np.hstack(raw_errors_pa)
        print('Running average - mean: %g, median: %g' % (np.mean(raws),
                                                          np.median(raws)))
        print('Running average - PA mean: %g, median: %g' %
              (np.mean(raws_pa), np.median(raws_pa)))
        print('====================')
        if action in all_errors.keys():
            all_errors[action].append(mean_error)
            all_errors_pa[action].append(mean_error_pa)
        else:
            all_errors[action] = [mean_error]
            all_errors_pa[action] = [mean_error_pa]

    all_act_errors = []
    all_act_errors_pa = []
    for act in actions:
        print('%s mean error %g, PA error %g' % (act, np.mean(all_errors[act]),
                                                 np.mean(all_errors_pa[act])))
        all_act_errors.append(np.mean(all_errors[act]))
        all_act_errors_pa.append(np.mean(all_errors_pa[act]))

    print('--for %s--' % config.load_path)
    print('Average error over all seq (over action) 3d: %g, PA: %g' %
          (np.mean(all_act_errors), np.mean(all_act_errors_pa)))

    act_names_in_order = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
        'TakingPhoto', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
        'Smoking', 'Waiting', 'WakingDog', 'Walking', 'WalkTogether'
    ]
    act_error = [
        '%.2f' % np.mean(all_errors[act]) for act in act_names_in_order
    ]
    act_PA_error = [
        '%.2f' % np.mean(all_errors_pa[act]) for act in act_names_in_order
    ]

    act_names_in_order.append('Average')
    act_error.append('%.2f' % np.mean(all_act_errors))
    act_PA_error.append('%.2f' % np.mean(all_act_errors_pa))
    print('---for excel---')
    print(', '.join(act_names_in_order))
    print(', '.join(act_error))
    print('With Alignment:')
    print(', '.join(act_PA_error))

    err_pa = np.hstack(raw_errors_pa)
    MPJPE = np.mean(np.hstack(raw_errors))
    PA_MPJPE = np.mean(err_pa)
    print('Average error over all joints 3d: %g, PA: %g' % (MPJPE, PA_MPJPE))

    err = np.hstack(raw_errors)
    median = np.median(np.hstack(raw_errors))
    pa_median = np.median(np.hstack(err_pa))
    print(
        'Percentiles 90th: %.1f 70th: %.1f 50th: %.1f 30th: %.1f 10th: %.1f' %
        (np.percentile(err, 90), np.percentile(err, 70),
         np.percentile(err, 50), np.percentile(err, 30),
         np.percentile(err, 10)))

    print('MPJPE: %.2f, PA-MPJPE: %.2f, Median: %.2f, PA-Median: %.2f' %
          (MPJPE, PA_MPJPE, median, pa_median))



if __name__ == '__main__':
    opt = TrainOptions(debug=False)
    main(opt)