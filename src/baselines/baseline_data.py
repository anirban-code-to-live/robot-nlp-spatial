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


class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_TOKEN: "<PAD>", config.UNK_TOKEN: '<UNK>'}
        self.num_words = 2  # Count <PAD>, <UNK>

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def get_token_id(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return config.UNK_TOKEN


def build_vocab(instructions):
    vocab = Vocab(name='fruits-world')
    for inst in instructions:
        words = tokenize(inst)
        for word in words:
            vocab.add_word(word)
    return vocab


def tokenize(inst):
    inst = inst.replace('.', '')
    return inst.split(' ')


def inst_to_tokens(instructions, vocab):
    Ts = np.array([[vocab.get_token_id(tok) for tok in tokenize(inst)] + [vocab.get_token_id(config.PAD_TOKEN)] * (
                    config.MAX_SEQ_LEN - len(tokenize(inst))) for inst in instructions])
    return Ts


def extract_scene_data(scene_data):
    instruction = scene_data['instruction']
    uniq_img_id = scene_data['unique_img_id']
    object_coords, object_pxords, object_idxs = [], [], []
    target_coord = scene_data['target_coord']
    for idx, obj in enumerate(scene_data['objects']):
        obj_coord, obj_pxord, obj_id = obj['coord'], obj['pixel_coord'], obj['id']
        object_coords.append(obj_coord)
        object_pxords.append(obj_pxord)
        object_idxs.append(obj_id)
        if obj_coord == target_coord:
            target_idx = idx
    return instruction, object_coords, object_pxords, object_idxs, target_coord, target_idx, uniq_img_id


def make_dataset(scenes, feats, args):
    Is, Xs, Ys, Yis, UIDs = [], [], [], [], []
    for scene in scenes:
        instruction, _, _, _, target_coord, target_idx, uniq_img_id = extract_scene_data(scene)
        Is.append(instruction)
        Xs.append(feats[scene['template_type'] + '_' + str(uniq_img_id)])
        Ys.append(target_coord)
        Yis.append(target_idx)
    return {
        'Is': Is,
        'Xs': np.array(Xs, dtype=np.float32),
        'Ys': np.array(Ys, dtype=np.float32),
        'Yis': np.array(Yis)
    }


def get_scene_data(args, type):
    scene_path = os.path.join(args.scene_dir, args.template, args.action + '_' + args.target + '_' + type + '_scenes.json')
    with open(scene_path, 'r') as f:
        scenes = json.load(f)['scenes']
    return scenes


def get_image_data(args, type):
    feat_path = os.path.join(args.scene_dir, args.template,
                             args.action + '_' + args.target + '_' + args.model + '_' + type + '_feat.pkl')
    with open(feat_path, 'rb') as f:
        feats = pickle.load(f)
    return feats


def main(args):
    train_scenes, train_feats = get_scene_data(args, 'train'), get_image_data(args, 'train')
    val_scenes, val_feats = get_scene_data(args, 'dev'), get_image_data(args, 'dev')
    test_scenes, test_feats = get_scene_data(args, 'test'), get_image_data(args, 'test')

    train_data, val_data, test_data = make_dataset(train_scenes, train_feats, args), \
                                      make_dataset(val_scenes, val_feats, args), \
                                      make_dataset(test_scenes, test_feats, args)
    vocab = build_vocab(train_data['Is'])
    vocab_size = vocab.num_words
    train_data['Ts'], val_data['Ts'], test_data['Ts'] = np.array(inst_to_tokens(train_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(val_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(test_data['Is'], vocab))
    return train_data, val_data, test_data, vocab_size


class BaselineDataset(Dataset):
    def __init__(self, data):
        super(BaselineDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        t = self.data['Ts'][idx]
        x = self.data['Xs'][idx]
        y = self.data['Ys'][idx]
        return t, x, y

    def __len__(self):
        return len(self.data['Is'])


def get_data(args):
    train_data, val_data, test_data, vocab_size = main(args)

    print('*' * 40)
    print('Count of train samples: {}'.format(len(train_data['Is'])))
    print('Count of validation samples: {}'.format(len(val_data['Is'])))
    print('Count of test samples: {}'.format(len(test_data['Is'])))
    print('Vocabulary size: {}'.format(vocab_size))

    #######################################
    # Load Dataset
    #######################################
    train_dataset = BaselineDataset(train_data)
    dev_dataset = BaselineDataset(val_data)
    test_dataset = BaselineDataset(test_data)
    #######################################

    #######################################
    # Data Loader
    #######################################
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    #######################################
    return train_loader, dev_loader, test_loader, vocab_size


def get_ungrounded_data_center_baseline(args):
    test_scenes = get_scene_data(args, 'test')
    Ys = []
    for scene in test_scenes:
        _, _, _, _, target_coord, _, _ = extract_scene_data(scene)
        Ys.append(target_coord)
    return {
        'Ys': np.array(Ys, dtype=np.float32)
    }


def make_bisk_rnn_dataset(scenes):
    Is, Xs, Ys, Yis = [], [], [], []
    for scene in scenes:
        instruction, object_coords, _, object_idxs, target_coord, target_idx, _ = extract_scene_data(scene)
        Is.append(instruction)
        X = np.zeros((1 + config.MAX_SCENE_OBJECT_COUNT, 2), dtype=np.float32)
        for i in range(len(object_coords)):
            obj_coord, obj_id = object_coords[i], int(object_idxs[i])
            # Handle array index out of bounds scenario
            if obj_coord[0] == 1.0: obj_coord[0] -= 0.001
            if obj_coord[1] == 1.0: obj_coord[1] -= 0.001
            # end
            X[i, :] = np.array([obj_coord[0], obj_coord[1]])
        Xs.append(X)
        Ys.append(target_coord)
        Yis.append(target_idx)
    return {
        'Is': Is,
        'Xs': np.array(Xs, dtype=np.float32),
        'Ys': np.array(Ys, dtype=np.float32),
        'Yis': np.array(Yis)
    }


def extract_scene_data_lang_fc(scene_data):
    instruction = scene_data['instruction']
    object_coords, object_pxords, object_idxs, object_sizes = [], [], [], []
    target_coord = scene_data['target_coord']
    for idx, obj in enumerate(scene_data['objects']):
        obj_coord, obj_pxord, obj_id, obj_size = obj['coord'], obj['pixel_coord'], obj['id'], obj['width']
        object_coords.append(obj_coord)
        object_pxords.append(obj_pxord)
        object_idxs.append(obj_id)
        object_sizes.append(obj_size)
        if obj_coord == target_coord:
            target_idx = idx
    return instruction, object_coords, object_pxords, object_idxs, object_sizes, target_coord, target_idx


def make_lang_fc_dataset(scenes):
    Is, Xs, Ys, Yis = [], [], [], []
    for scene in scenes:
        instruction, object_coords, _, object_idxs, object_sizes, target_coord, target_idx = extract_scene_data_lang_fc(scene)
        Is.append(instruction)

        X = np.zeros((config.MAX_SCENE_OBJECT_COUNT, 3 + config.UNIQUE_OBJECT_COUNT), dtype=np.float32)
        for i in range(len(object_coords)):
            obj_coord, obj_id, obj_size = object_coords[i], int(object_idxs[i]), [object_sizes[i]]
            # Handle array index out of bounds scenario
            if obj_coord[0] == 1.0: obj_coord[0] -= 0.001
            if obj_coord[1] == 1.0: obj_coord[1] -= 0.001
            # end
            X[i, :] = np.array(obj_coord + obj_size + list(np.eye(config.UNIQUE_OBJECT_COUNT)[obj_id]))
        Xs.append(X)
        Ys.append(target_coord)
        Yis.append(target_idx)
    return {
        'Is': Is,
        'Xs': np.array(Xs, dtype=np.float32),
        'Ys': np.array(Ys, dtype=np.float32),
        'Yis': np.array(Yis)
    }


def get_lang_fc_data(args):
    train_scenes = get_scene_data(args, 'train')
    val_scenes = get_scene_data(args, 'dev')
    test_scenes = get_scene_data(args, 'test')

    train_data, val_data, test_data = make_lang_fc_dataset(train_scenes), \
                                      make_lang_fc_dataset(val_scenes), \
                                      make_lang_fc_dataset(test_scenes)
    vocab = build_vocab(train_data['Is'])
    vocab_size = vocab.num_words
    train_data['Ts'], val_data['Ts'], test_data['Ts'] = np.array(inst_to_tokens(train_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(val_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(test_data['Is'], vocab))
    return train_data, val_data, test_data, vocab_size


def get_bisk_rnn_data(args):
    train_scenes = get_scene_data(args, 'train')
    val_scenes = get_scene_data(args, 'dev')
    test_scenes = get_scene_data(args, 'test')

    train_data, val_data, test_data = make_bisk_rnn_dataset(train_scenes), \
                                      make_bisk_rnn_dataset(val_scenes), \
                                      make_bisk_rnn_dataset(test_scenes)
    vocab = build_vocab(train_data['Is'])
    vocab_size = vocab.num_words
    train_data['Ts'], val_data['Ts'], test_data['Ts'] = np.array(inst_to_tokens(train_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(val_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(test_data['Is'], vocab))

    print('*' * 40)
    print('Count of train samples: {}'.format(len(train_data['Is'])))
    print('Count of validation samples: {}'.format(len(val_data['Is'])))
    print('Count of test samples: {}'.format(len(test_data['Is'])))
    print('Vocabulary size: {}'.format(vocab_size))

    #######################################
    # Load Dataset
    #######################################
    train_dataset = BaselineDataset(train_data)
    dev_dataset = BaselineDataset(val_data)
    test_dataset = BaselineDataset(test_data)
    #######################################

    #######################################
    # Data Loader
    #######################################
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    #######################################
    return train_loader, dev_loader, test_loader, vocab_size


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for create dataset.')
    parser.add_argument('--template', default='SP_SO_REL_SOR', type=str, help='Template type.')
    parser.add_argument('--action', default='pick', type=str, help='Action type. pick or pick_and_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Parent directory path for stored images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Parent directory path for stored scenes.')
    parser.add_argument('--model', default='lstm', type=str, help='Model: lstm/cnn/sa')
    parser.add_argument('--train_fraction', default=0.8, type=float, help='Fraction of total data to be used for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_data, val_data, feats, vocab_size = get_lang_fc_data(args)
    print('Vocab size: ', vocab_size)
    print(len(train_data['Is']))
    print(train_data['Xs'].shape)
    print(train_data['Ts'].shape)
    print(train_data['Ys'].shape)
    print(train_data['Yis'].shape)

    # train_loader, dev_loader, test_loader, vocab_size = get_bisk_rnn_data(args)
    # t, x, y = next(iter(train_loader))
    # print(t.shape, x.shape, y.shape)


