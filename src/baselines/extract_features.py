# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py
import numpy as np
from PIL import Image

import torch
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True, progress=True)
    # print(cnn._modules)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    if args.baseline != 'sa':
        layers.append(getattr(cnn, 'avgpool'))
    model = torch.nn.Sequential(*layers)
    # model.cuda()
    model.to(device)
    model.eval()
    return model


def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    # image_batch = torch.FloatTensor(image_batch).cuda()
    image_batch = torch.FloatTensor(image_batch).to(device)
    image_batch = torch.autograd.Variable(image_batch)

    with torch.no_grad():
        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()

    return feats


def main(args):
    input_image_dir = os.path.join(args.image_dir, args.template, args.action, args.target)
    output_h5_file = os.path.join(args.scene_dir, args.template,  args.action + '_' + args.target + '_' + args.baseline + '_feat.h5')
    input_paths = []
    idx_set = set()
    for fn in os.listdir(input_image_dir):
        if not fn.endswith('.png'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]
    # print(input_paths[0])
    # print(input_paths[-1])

    model = build_model(args)

    img_size = (args.image_height, args.image_width)
    with h5py.File(output_h5_file, 'w') as f:
        feat_dset = None
        i0 = 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = Image.open(path)
            img = img.resize(img_size)
            img = np.transpose(np.array(img), [2, 0, 1])[None]
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    # print(feats.shape)
                    feat_dset = f.create_dataset('features', (N, C, H, W),
                                                 dtype=np.float32)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)))
                cur_batch = []
        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            print('Processed %d / %d images' % (i1, len(input_paths)))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--template', default='SO_SIZE', help='Template type.'),
    parser.add_argument('--action', default='pick', help='Action type. pick | pick_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Directory to store images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Directory of stored scene data.')
    parser.add_argument('--baseline', default='lstm', help='Baseline model. lstm/cnn/sa')

    parser.add_argument('--max_images', default=None, type=int)

    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    parser.add_argument('--model', default='resnet101')
    parser.add_argument('--model_stage', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
