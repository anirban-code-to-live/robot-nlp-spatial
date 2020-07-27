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

ALL_TEMPLATES = ['SP_SO_REL_SOR',
                 'SP_SO_ABS',
                 'SO_SIZE',
                 'SIMPLE_ABSTRACT',
                 'COMPLEX_ABSTRACT',
                 'SIMPLE_ORDINAL',
                 'SIMPLE_CARDINAL']


def get_feats(scenes, all_feats, template):
    feats = {}
    for scene in scenes:
        uniq_img_id = scene['unique_img_id']
        feats[template + '_' + str(uniq_img_id)] = all_feats[uniq_img_id]
    return feats


def get_scene_data(args, type, template):
    scene_path = os.path.join(args.scene_dir, template, args.action + '_' + args.target + '_' + type + '_scenes.json')
    with open(scene_path, 'r') as f:
        scenes = json.load(f)['scenes']
    feat_path = os.path.join(args.scene_dir, template, args.action + '_' + args.target + '_' + args.model + '_' + type + '_feat.pkl')
    with open(feat_path, 'rb') as f:
        feats = pickle.load(f)
    return scenes, feats


def save_scenes(scenes, feats, args, type):
    scene_dir = os.path.join(args.scene_dir, args.template)
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
    scene_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + type + '_scenes.json')
    feat_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + args.model + '_' + type + '_feat.pkl')
    with open(scene_path, 'w') as f:
        json.dump({'scenes': scenes}, f)
    with open(feat_path, 'wb') as f:
        pickle.dump(feats, f)


def main(args):
    all_train_scenes, all_val_scenes, all_test_scenes = [], [], []
    all_train_feats, all_val_feats, all_test_feats = {}, {}, {}

    for template in ALL_TEMPLATES:
        train_scenes, train_feats = get_scene_data(args, 'train', template)
        all_train_scenes = all_train_scenes + train_scenes
        all_train_feats = {**all_train_feats, **train_feats}

        val_scenes, val_feats = get_scene_data(args, 'dev', template)
        all_val_scenes = all_val_scenes + val_scenes
        all_val_feats = {**all_val_feats, **val_feats}

        test_scenes, test_feats = get_scene_data(args, 'test', template)
        all_test_scenes = all_test_scenes + test_scenes
        all_test_feats = {**all_test_feats, **test_feats}

    save_scenes(all_train_scenes, all_train_feats, args, type='train')
    save_scenes(all_val_scenes, all_val_feats, args, type='dev')
    save_scenes(all_test_scenes, all_test_feats, args, type='test')


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for create dataset.')
    parser.add_argument('--template', default='COMBINED', type=str, help='Template type.')
    parser.add_argument('--action', default='pick', type=str, help='Action type. pick or pick_and_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Parent directory path for stored images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Parent directory path for stored scenes.')
    parser.add_argument('--model', default='lstm', type=str, help='Model: lstm/cnn/sa')
    parser.add_argument('--train_fraction', default=0.7, type=float, help='Fraction of total data to be used for training.')
    parser.add_argument('--test_fraction', default=0.2, type=float, help='Fraction of total data to be used for test.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
