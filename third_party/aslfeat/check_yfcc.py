#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

with open('yfcc_test_pairs_with_gt.txt', 'r') as f:
    pairs = [l.split() for l in f.readlines()]

print('num_pairs:{:d}'.format(len(pairs)))

input_dir = Path('/home/archer/newDisk/OANet/raw_data/yfcc100m/')
output_dir = Path('/media/archer/Samsung_T5/cyberfeat_yfcc')

namemap = {}

for pair in tqdm(pairs):

    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem

    if (stem0 in namemap) and namemap[stem0] != name0:
        print("map error!", namemap[stem0], name0)

    namemap[stem0] = name0

    if (stem1 in namemap) and namemap[stem1] != name1:
        print("map error!", namemap[stem1], name1)

    namemap[stem1] = name1

print(len(namemap))
