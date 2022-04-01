#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import yaml
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path
from copy import deepcopy

from utils.opencvhelper import MatcherWrapper

from models import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def draw_keypoints(img, kp, path):

    cv_kpts = [cv2.KeyPoint(kp[i][0], kp[i][1], 1) for i in range(kp.shape[0])]

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    out_img = img.copy()

    cv2.drawKeypoints(out_img, cv_kpts, out_img)

    cv2.imwrite(str(path), out_img)


def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def read_image(path, resize, rotation, resize_float):
    image = cv2.imread(str(path))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    # return image, gray, (1, 1)

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    print('old ({:d}, {:d}) -> new ({:d}, {:d})'.format(w, h, w_new, h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
    image = image.astype(np.uint8)

    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()
        if rotation % 2:
            scales = scales[::-1]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    return image, gray, scales


with open(FLAGS.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = get_model('feat_model')(config['model_path'], **config['net'])

with open('yfcc_test_pairs_with_gt.txt', 'r') as f:
    pairs = [l.split() for l in f.readlines()]

print('num_pairs:{:d}'.format(len(pairs)))

input_dir = Path('/devdata/raw_data/yfcc100m/')
# output_dir = Path('/devdata/raw_data/cyberfeat_yfcc_r1600/')
output_dir = Path('/devdata/raw_data/cyberfeat_yfcc_tmp1600/')

output_dir.mkdir(parents=True, exist_ok=True)

solved = {}

for i, pair in enumerate(pairs):
    if i == 10:
        break

    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem

    print('stem0:{}, stem1:{}'.format(stem0, stem1))

    out_path0, out_path1 = output_dir / '{}.npz'.format(stem0), output_dir / '{}.npz'.format(stem1)

    outImg_path0, outImg_path1 = output_dir / '{}.png'.format(stem0), output_dir / '{}.png'.format(stem1)

    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0
    # Load the image pair.
    if not name0 in solved:
        image0, gray0, scales0 = read_image(input_dir / name0, (720, ), rot0, True)
        desc0, kpt0, score0 = model.run_test_data(gray0)
        draw_keypoints(image0, kpt0, outImg_path0)
        # np.savez_compressed(out_path0, keypoints=kpt0, scores=score0, descriptors=desc0, scales=scales0)
        solved[name0] = True

    if not name1 in solved:
        image1, gray1, scales1 = read_image(input_dir / name1, (720, ), rot1, True)
        desc1, kpt1, score1 = model.run_test_data(gray1)
        draw_keypoints(image1, kpt1, outImg_path1)
        # np.savez_compressed(out_path1, keypoints=kpt1, scores=score1, descriptors=desc1, scales=scales1)
        solved[name1] = True

    # print(desc0.shape, desc1.shape)
    # print(kpt0.shape, kpt1.shape)
    # print(score0.shape, score1.shape)

    # print(image0.shape, image1.shape)
    # break

    # break
