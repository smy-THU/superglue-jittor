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

from utils.opencvhelper import MatcherWrapper

from models import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def draw_keypoints(img, kp):

    cv_kpts = [cv2.KeyPoint(kp[i][0], kp[i][1], 1) for i in range(kp.shape[0])]

    out_img = img

    cv2.drawKeypoints(out_img, cv_kpts, out_img)

    cv2.imwrite('tmp.png', out_img)


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    # parse input
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = get_model('feat_model')(config['model_path'], **config['net'])

    img_folder = config['img_folder']
    out_folder = config['out_folder']

    for frame in os.listdir(img_folder):
        frameid = frame.rstrip('.png')
        print(frameid)
        img_path = os.path.join(img_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        desc, kpt, score = model.run_test_data(gray)
        print(desc.shape)
        print(kpt.shape)
        print(score.shape)
        out_path = os.path.join(out_folder, frameid + '.npz')
        np.savez_compressed(out_path, keypoints=kpt, scores=score, local_descriptors=desc)

        draw_keypoints(img, kpt)
        # break


if __name__ == '__main__':
    tf.compat.v1.app.run()
