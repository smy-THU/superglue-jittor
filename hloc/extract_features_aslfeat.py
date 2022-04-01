import argparse
import sys
from pathlib import Path
import h5py
import logging
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import torch

aslfeat_root = Path(__file__).parent / '../third_party/aslfeat'

confs = {
    'aslfeat_aachen': {
        'model_path': str(aslfeat_root / 'pretrained/cyberfeat/model.ckpt-150000'),
        'net': {
            'max_dim': 2048,
            'config': {
                'kpt_n': 4096,
                'kpt_refinement': True,
                'deform_desc': 1,
                'score_thld': 0.2,
                'edge_thld': 10,
                'multi_scale': False,  ###True or False
                'multi_level': True,
                'nms_size': 3,
                'eof_mask': 5,
                'need_norm': True,
                'use_peakiness': True,
                'cell_size': 8,
            },
        },
        'output': 'feats-aslfeat-n4096-r1024',
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'aslfeat_inloc': {
        'model_path': str(aslfeat_root / 'pretrained/cyberfeat/model.ckpt-150000'),
        'net': {
            'max_dim': 2048,
            'config': {
                'kpt_n': 4096,
                'kpt_refinement': True,
                'deform_desc': 1,
                'score_thld': 0.2,
                'edge_thld': 10,
                'multi_scale': False,  ###True or False
                'multi_level': True,
                'nms_size': 3,
                'eof_mask': 5,
                'need_norm': True,
                'use_peakiness': True,
                'cell_size': 8,
            },
        },
        'output': 'feats-aslfeat-n4096-r1600',
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
}


def draw_keypoints(img, kp, path):

    cv_kpts = [cv2.KeyPoint(kp[i][0], kp[i][1], 1) for i in range(kp.shape[0])]

    if len(img.shape) == 2:
        img = img[..., np.newaxis]

    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    out_img = img.copy()

    cv2.drawKeypoints(out_img, cv_kpts, out_img)

    cv2.imwrite(str(path), out_img)


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/' + g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = [i.relative_to(root) for i in self.paths]
        logging.info(f'Found {len(self.paths)} images in root {root}.')

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and max(w, h) > self.conf.resize_max:
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # if self.conf.grayscale:
        #     image = image[None]
        # else:
        #     image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        # image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)


def main(conf, image_dir, export_dir, as_half=False):


    logging.info('Extracting local features with configuration:' f'\n{pprint.pformat(conf)}')

    sys.path.append(str(aslfeat_root))
    from models import get_model
    model = get_model('feat_model')(conf['model_path'], **(conf['net']))

    loader = ImageDataset(image_dir, conf['preprocessing'])
    loader = torch.utils.data.DataLoader(loader, num_workers=8)

    feature_path = Path(export_dir, conf['output'] + '.h5')
    if feature_path.exists():
        logging.warning('feature extraction file already exist, check if it is right')
        return

    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    # id = 0
    for data in tqdm(loader):
        # if data['name'][0] in feature_file:
        #     continue
        img = data['image'][0].numpy().astype(np.uint8)
        desc, kpt, score = model.run_test_data(img[..., np.newaxis])
        pred = {'keypoints': kpt, 'scores': score, 'descriptors': desc.T}

        # print(kpt.shape)
        # draw_keypoints(img, kpt, '%03d.png' % (id))
        # id += 1

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(img.shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            # pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
            pred['keypoints'] = pred['keypoints'] * scales[None]
            # pred['keypoints'] = np.floor(pred['keypoints'] * scales[None])

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        grp = feature_file.create_group(data['name'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finished exporting features.')
    model.close()
    sys.path.pop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen', choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir)
