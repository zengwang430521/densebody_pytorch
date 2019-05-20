"""
Evaluation of H3.6M.
Sample call from hmr:
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=<model_to_eval>
python -m src.benchmark.evaluate_h36m --batch_size=500 --load_path=/home/kanazawa/projects/hmr_v2/models/model.ckpt-667589
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import flags
import numpy as np
import deepdish as dd
from time import time
from os.path import exists, join, expanduser, split
from os import makedirs
# import tensorflow as tf
from evals.config import get_config
# from ..util import renderer as vis_util
# from ..RunModel import RunModel
from evals.eval_util import compute_errors
# from ..datasets.common import read_images_from_tfrecords

kPredDir = '/tmp/hmr_output'
# Change to where you saved your tfrecords
kTFDataDir = '/scratch1/projects/tf_datasets/tf_records_human36m_wjoints'

flags.DEFINE_string('pred_dir', kPredDir,
                    'where to save model output of h36m')
flags.DEFINE_string('tfh36m_dir', kTFDataDir,
                    'data dir: top of h36m in tf_records')
flags.DEFINE_integer(
    'protocol', 1,
    'If 2, then only frontal cam (3) and trial 1, if 1, then all camera & trials'
)
flags.DEFINE_boolean(
    'vis', False, 'If true, visualizes the best and worst 30 results.')

model = None
sess = None
# For visualization.
renderer = None
extreme_errors, contents = [], []


# -- Draw Utils ---
def draw_content(content, config):
    global renderer
    input_img = content['image']
    vert = content['vert']
    joint = content['joint']
    cam = content['cam']
    img_size = config.img_size
    # Undo preprocessing
    img = (input_img + 1) * 0.5 * 255
    tz = renderer.flength / (0.5 * img_size * cam[0])
    trans = np.hstack([cam[1:], tz])
    vert_shifted = vert + trans
    # Draw
    skel_img = vis_util.draw_skeleton(img, joint)
    rend_img = renderer(vert_shifted, img_size=(img_size, img_size))
    another_vp = renderer.rotated(vert_shifted, 90, img_size=(img_size, img_size), do_alpha=False)
    another_vp = vis_util.draw_text(another_vp, {"diff_viewpoint": 90})

    tog0 = np.hstack((img, rend_img))
    tog1 = np.hstack((skel_img, another_vp))

    all_img = np.vstack((tog0, tog1)).astype(np.uint8)
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(all_img)
    # plt.axis('off')
    # import ipdb; ipdb.set_trace()

    return all_img


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


# -- Core: ---
def get_data(seq_name, config):
    """
    Read preprocessed image from tfrecords.
    """
    global sess
    '''
    if sess is None:
        sess = tf.Session()
    '''
    tf_path = join(
        expanduser(config.tfh36m_dir), 'test', seq_name + '.tfrecord')
    '''
    images, kps, gt3ds = read_images_from_tfrecords(
        tf_path, img_size=config.img_size, sess=sess)
    '''
    return images, gt3ds





def add_visuals(errors, results, images):
    global extreme_errors, contents
    # Record extreme ones
    sort_inds = np.argsort(errors)[::-1]
    # Save top/worst 10.
    for i in range(10):
        ind = sort_inds[i]
        content = {
            'vert': results['verts'][ind],
            'joint': results['joints'][ind],
            'image': images[ind],
            'cam': results['cams'][ind],
        }
        extreme_errors.append(errors[ind])
        contents.append(content)
        # Save best too.
        best_ind = sort_inds[-(i + 1)]
        content = {
            'vert': results['verts'][best_ind],
            'joint': results['joints'][best_ind],
            'image': images[best_ind],
            'cam': results['cams'][best_ind],
        }
        extreme_errors.append(errors[best_ind])
        contents.append(content)


def evaluate_sequence(model, seq_info, save_dir):
    sub_id, action, trial_id, cam_id = seq_info


    file_seq_name = 'S%d_%s_%d_cam%01d' % (sub_id, action, trial_id, cam_id)
    print('%s' % (file_seq_name))

    save_path = join(save_dir, file_seq_name)
    if exists(save_path):
        pass
    else:
        # Run the model!
        t0 = time()
        images, gt3ds = get_data(file_seq_name, config)
        results = run_model(images, model)

        results = run_model(images, config)
        t1 = time()
        print('Took %g sec for %d imgs' % (t1 - t0, len(results['verts'])))

        # Evaluate!
        # Joints 3D is COCOplus format now. First 14 is H36M joints
        pred3ds = results['joints3d'][:, :14, :]
        # Convert to mm!
        errors, errors_pa = compute_errors(gt3ds * 1000., pred3ds * 1000.)

        results['errors'] = errors
        results['errors_pa'] = errors_pa
        # Save results
        dd.io.save(save_path, results)

    if config.vis:
        add_visuals(errors, results, images)

    return errors, errors_pa


def main(config):
    # Figure out the save name.
    pred_dir = get_pred_dir(config.pred_dir, config.load_path)
    protocol = config.protocol

    all_pairs, actions = get_h36m_seqs(protocol)

    all_errors = {}
    all_errors_pa = {}
    raw_errors, raw_errors_pa = [], []
    for itr, seq_info in enumerate(all_pairs):
        print('%d/%d' % (itr, len(all_pairs)))
        sub_id, action, trial_id, cam_id = seq_info

        errors, errors_pa = evaluate_sequence(seq_info, pred_dir)

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

    if config.vis:
        global extreme_errors, contents
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(1)
        plt.clf()
        sort_inds = np.argsort(extreme_errors)[::-1]
        for i in range(30):
            bad_ind = sort_inds[i]
            bad_error = extreme_errors[bad_ind]
            bad_img = draw_content(contents[bad_ind], config)
            plt.figure(1)
            plt.clf()
            plt.imshow(bad_img)
            plt.axis('off')
            plt.title('%d-th worst, mean error %.2fmm' % (i, bad_error))

            good_ind = sort_inds[-(i + 1)]
            good_error = extreme_errors[good_ind]
            good_img = draw_content(contents[good_ind], config)
            plt.figure(2)
            plt.clf()
            plt.imshow(good_img)
            plt.axis('off')
            plt.title('%d-th best, mean error %.2fmm' % (i, good_error))
            import ipdb;
            ipdb.set_trace()




if __name__ == '__main__':
    config = get_config()
    main(config)