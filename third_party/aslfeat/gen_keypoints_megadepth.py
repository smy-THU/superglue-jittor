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
import torch
from torch.utils.data import Dataset, DataLoader

from utils.opencvhelper import MatcherWrapper

from models import get_model
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def draw_keypoints(img, kp, path):

    cv_kpts = [cv2.KeyPoint(kp[i][0], kp[i][1], 1) for i in range(kp.shape[0])]

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    out_img = img.copy()

    cv2.drawKeypoints(out_img, cv_kpts, out_img)

    cv2.imwrite(str(path), out_img)


def process_resize(w, h, resize, f=min):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / f(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    # if max(w_new, h_new) < 160:
    #     print('Warning: input resolution is very small, results may vary')
    # elif max(w_new, h_new) > 2000:
    #     print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def read_image(path, resize, rotation, resize_float):
    image = cv2.imread(str(path))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    # return image, gray, (1, 1)

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    # print('old ({:d}, {:d}) -> new ({:d}, {:d})'.format(w, h, w_new, h_new))

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

# with open('yfcc_test_pairs_with_gt.txt', 'r') as f:
#     pairs = [l.split() for l in f.readlines()]


class ImageFolder(Dataset):
    def __init__(self, root, resize=(600, ), resize_float=True, rotation=0):
        self.root = Path(root)
        self.imgs = [self.root / x for x in sorted(os.listdir(root))]
        self.resize = resize
        self.resize_float = resize_float
        self.rotation = rotation

    def __getitem__(self, index):
        image, gray, scales = read_image(str(self.imgs[index]), self.resize, self.rotation, self.resize_float)
        return image, gray, scales, str(self.imgs[index])

    def __len__(self):
        return len(self.imgs)


# print('num_pairs:{:d}'.format(len(pairs)))

# usfm_path = Path('/devdata/megadepth/Undistorted_SfM')
# output_path = Path('/devdata/megadepth/aslfeatv2_n2048_r600_c8_ss')

shoes_path = Path('/mnt/disk/round_4')
out_path = Path('/mnt/disk/shoes_keypoint')

scene_names = sorted(os.listdir(shoes_path))

for scene_name in scene_names:
    # if scene_name != '0402':
    #     continue

    img_dir = shoes_path / scene_name
    if not img_dir.exists():
        continue

    feat_dir = out_path / scene_name

    if not feat_dir.exists():
        feat_dir.mkdir(exist_ok=True, parents=True)

    print("processing:", scene_name, " dir:", feat_dir)

    image_set = ImageFolder(img_dir)
    dataloader = DataLoader(image_set, batch_size=1, shuffle=False, num_workers=4)

    for idx, (image, gray, scales, img_path) in tqdm(enumerate(dataloader), total=len(dataloader)):
        image, gray, scales, img_path = image[0].numpy(), gray[0].numpy(), scales[0].numpy(), Path(img_path[0])
        # print(img_path)
        # print(gray.shape, image.shape, scales)
        desc, kpt, score = model.run_test_data(gray)
        # print(score.max(), score.min())
        image_path = Path('/mnt/disk/shoes_keypoint/visualization') / (img_path.stem + '.png')
        draw_keypoints(image, kpt, str(image_path))
        feat_path = feat_dir / (img_path.stem + '.npz')
        np.savez_compressed(feat_path, keypoints=kpt, scores=score, descriptors=desc, scales=scales)

