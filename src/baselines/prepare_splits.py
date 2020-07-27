import numpy as np
import argparse
import os
import random
import json
import pickle
import h5py
import sys

from torch.utils.data import DataLoader, Dataset
sys.path.append('../../')
from config.global_config import GlobalConfig


config = GlobalConfig()


def get_feats(scenes, all_feats, template):
    feats = {}
    for scene in scenes:
        uniq_img_id = scene['unique_img_id']
        feats[template + '_' + str(uniq_img_id)] = all_feats[uniq_img_id]
    return feats


def save_scenes(scenes, args, type):
    scene_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + type + '_scenes.json')
    with open(scene_path, 'w') as f:
        json.dump({'scenes': scenes}, f)


def save_feats(feats, args, type, mod):
    feat_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + mod + '_' + type + '_feat.pkl')
    with open(feat_path, 'wb') as f:
        pickle.dump(feats, f)


def main(args):
    scene_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + 'scenes.json')
    with open(scene_path, 'r') as f:
        all_scenes = json.load(f)['scenes']
    train_scenes = random.sample(all_scenes, int(len(all_scenes) * args.train_fraction))
    rem_scenes = [scene for scene in all_scenes if scene not in train_scenes]
    test_scenes = random.sample(rem_scenes, int(len(all_scenes) * args.test_fraction))
    val_scenes = [scene for scene in rem_scenes if scene not in test_scenes]

    save_scenes(train_scenes, args, type='train')
    save_scenes(val_scenes, args, type='dev')
    save_scenes(test_scenes, args, type='test')

    for model in ['lstm', 'cnn', 'sa']:
        feat_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + model + '_' + 'feat.h5')
        print('Reading features from ', feat_path)
        feats = h5py.File(feat_path, 'r')['features']

        train_feats, val_feats, test_feats = get_feats(train_scenes, feats, args.template), \
                                             get_feats(val_scenes, feats, args.template), \
                                             get_feats(test_scenes, feats, args.template)

        save_feats(train_feats, args, type='train', mod=model)
        save_feats(val_feats, args, type='dev', mod=model)
        save_feats(test_feats, args, type='test', mod=model)




def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for create dataset.')
    parser.add_argument('--template', default='SP_SO_ABS', type=str, help='Template type.')
    parser.add_argument('--action', default='pick', type=str, help='Action type. pick or pick_and_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Parent directory path for stored images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Parent directory path for stored scenes.')
    # parser.add_argument('--model', default='lstm', type=str, help='Model: lstm/cnn/sa')
    parser.add_argument('--train_fraction', default=0.7, type=float, help='Fraction of total data to be used for training.')
    parser.add_argument('--test_fraction', default=0.2, type=float, help='Fraction of total data to be used for test.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
