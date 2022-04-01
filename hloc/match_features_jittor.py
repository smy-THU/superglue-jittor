from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint
import matplotlib.cm as cm

from .utils.viz import *
from .utils.parsers import names_to_pair

import sys, os
sys.path.append(os.path.dirname(__file__)+"../pack/jittor/python")

import os

use_fp16 = 1

confs = {
    'superglue-aslfeat': {
        'output': 'matches-superglue-aslfeat-jittor_nd100',
        'model': {
            'name': 'superglue',
            'descriptor_dim': 128,
            'sinkhorn_iterations': 25,
            'match_threshold': 0.01,
            'keypoint_position_dim': 2,
            'use_dual_softmax': True,
            'GNN_layers': ['self', 'cross'] * 9,
            'weights': 'superglue_cyberfeat_1800_nd_100.pkl',
        },
    },
}


def main(conf, pairs, features, export_dir, dataset_path='', exhaustive=False):
    import jittor as jt
    jt.flags.use_cuda = 1
    from pack import superglue_infer
    from pack.superglue_infer import SuperGlue
    superglue_infer.split_size = 1  # TODO: can delete
    convopt = 0 # TODO 1 or 0
    if convopt == 1:
        os.environ.setdefault('conv_opt', '1')
    model = SuperGlue(conf['model'])

    if convopt == 1:
        state_dict = jt.load('pack/' + conf['model']['weights'][:-4] + '_convopt1.pkl')
        model.load_state_dict(state_dict)
    else:
        state_dict = jt.load('pack/' + conf['model']['weights'])
        model.load_state_dict(state_dict)

    model.eval()

    jt.flags.profiler_enable = int(os.environ.get("profiler", "0"))

    logging.info('Matching local features with configuration:' f'\n{pprint.pformat(conf)}')

    feature_path = Path(export_dir, features + '.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')

    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    # elif exhaustive:
    #     logging.info(f'Writing exhaustive match pairs to {pairs}.')
    #     assert not pairs.exists(), pairs
    #
    #     # get the list of images from the feature file
    #     images = []
    #     feature_file.visititems(
    #         lambda name, obj: images.append(obj.parent.name.strip('/')) if isinstance(obj, h5py.Dataset) else None)
    #     images = list(set(images))
    #
    #     pair_list = [' '.join((images[i], images[j])) for i in range(len(images)) for j in range(i)]
    #     with open(str(pairs), 'w') as f:
    #         f.write('\n'.join(pair_list))

    match_name = f'{features}_{conf["output"]}_{pairs_name}'
    match_path = Path(export_dir, match_name + '.h5')
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or (pair in match_file and 'matching_scores0' in match_file[pair]):
            continue

        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            data[k + '0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: jt.array(v)[None].float() for k, v in data.items()}
        data['shape0'] = data['image_size0']
        data['shape1'] = data['image_size1']
        data.pop('image_size0')
        data.pop('image_size1')



        # some matchers might expect an image but only use its size
        # data['image0'] = torch.empty((
        #                                  1,
        #                                  1,
        #                              ) + tuple(feats0['image_size'])[::-1])
        # data['image1'] = torch.empty((
        #                                  1,
        #                                  1,
        #                              ) + tuple(feats1['image_size'])[::-1])


        if use_fp16:
            for k, v in data.items():
                if isinstance(v, jt.Var) and v.dtype == "float32":
                    v.assign(v.float16())
            for v in model.parameters():
                if v.dtype == "float32":
                    v.assign(v.float16())
            jt.sync_all(True)

        pred = model(data)

        img0 = read_image(dataset_path / name0)
        img1 = read_image(dataset_path / name1)

        matches = pred['matches0'][0].numpy()
        confidence = pred['matching_scores0'][0].numpy()
        kpts0 = data['keypoints0']
        kpts1 = data['keypoints1']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        img_path = Path(export_dir, 'matching_imgs', pair)

        make_matching_plot_fast(img0, img1, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=img_path, show_keypoints=True)

        if pair in match_file:
            match_file.pop(pair)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].float16().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')
