import numpy as np
import argparse
import os
import random
import json
import pickle
import sys
sys.path.append('../')
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


def make_dataset(scenes):
    Is, Xs, Ys, Yis = [], [], [], []
    for scene in scenes:
        instruction, object_coords, _, object_idxs, object_sizes, target_coord, target_idx = extract_scene_data(scene)
        Is.append(instruction)
        X = np.zeros((config.SCALE, config.SCALE, 3 + config.UNIQUE_OBJECT_COUNT), dtype=np.float32)
        for i in range(len(object_coords)):
            obj_coord, obj_id, obj_size = object_coords[i], int(object_idxs[i]), [object_sizes[i]]
            # Handle array index out of bounds scenario
            if obj_coord[0] == 1.0: obj_coord[0] -= 0.001
            if obj_coord[1] == 1.0: obj_coord[1] -= 0.001
            # end
            X[int(config.SCALE * obj_coord[0]), int(config.SCALE * obj_coord[1]), :] = np.array(obj_coord + obj_size + list(np.eye(config.UNIQUE_OBJECT_COUNT)[obj_id]))
        Xs.append(X)
        Ys.append(target_coord)
        Yis.append(target_idx)
    return {
        'Is': Is,
        'Xs': np.array(Xs, dtype=np.float32),
        'Ys': np.array(Ys, dtype=np.float32),
        'Yis': np.array(Yis)
    }


def get_scene_data(args, type):
    scene_path = os.path.join(args.data_dir, args.dataset, type + '_scenes.json')
    with open(scene_path, 'r') as f:
        scenes = json.load(f)['scenes']
    return scenes


def main(args):
    train_scenes = get_scene_data(args, 'train')
    val_scenes = get_scene_data(args, 'dev')
    test_scenes = get_scene_data(args, 'test')

    train_data, val_data, test_data = make_dataset(train_scenes), \
                                      make_dataset(val_scenes), \
                                      make_dataset(test_scenes)
    vocab = build_vocab(train_data['Is'])
    vocab_size = vocab.num_words
    train_data['Ts'], val_data['Ts'], test_data['Ts'] = np.array(inst_to_tokens(train_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(val_data['Is'], vocab)), \
                                                        np.array(inst_to_tokens(test_data['Is'], vocab))
    return train_data, val_data, test_data, vocab_size


def get_data(args):
    train_data, val_data, test_data, vocab_size = main(args)
    return train_data, val_data, test_data, vocab_size


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for create dataset.')
    parser.add_argument('--dataset', default='synthetic', type=str, help='Template type.')
    parser.add_argument('--data_dir', default='../data/', type=str, help='Parent directory path for stored scenes.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_data, val_data, test_data, vocab_size = main(args)
    print('Vocab size: ', vocab_size)
    print(len(train_data['Is']))
    print(train_data['Xs'].shape)
    print(train_data['Ts'].shape)
    print(train_data['Ys'].shape)
    print(train_data['Yis'].shape)


