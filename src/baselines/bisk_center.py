import numpy as np
import sys
import argparse
sys.path.append('../../')
from src.baselines.baseline_data import get_ungrounded_data_center_baseline


def get_mse_abd_ta(p, t, blk_len=0.166, tol=0.05):
    num_sample = t.shape[0]
    mse = np.sum(np.linalg.norm(p - t, axis=1) ** 2) / num_sample
    abd = np.sum(np.linalg.norm(p - t, axis=1)) / (num_sample * blk_len)
    ta = np.sum(np.min(np.abs(p - t) < tol, axis=1).astype(np.float)) / num_sample
    return mse, abd, ta


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
    print('Baseline for blocks world - Center Target locations!')
    args = parse_args()
    #######################################
    # Test data
    #######################################
    tYs = get_ungrounded_data_center_baseline(args)['Ys']
    print('Number of test samples: {}'.format(tYs.shape[0]))
    assert tYs.shape[1] == 2
    #######################################

    #######################################
    # Center Target
    #######################################
    pYs = np.zeros(tYs.shape)
    #######################################

    #######################################
    # Evaluation metrics for target
    #######################################
    tgt_mse, tgt_abd, tgt_ta = get_mse_abd_ta(pYs, tYs)
    print('*' * 40)
    print('Target Prediction')
    print('-' * 40)
    print('MSE: {:.4f} | Avg. Block Distance: {:.3f} | Tolerable Accuracy: {:.2f}%'.format(
        tgt_mse, tgt_abd, tgt_ta * 100)
    )
    print('*' * 40)
    #######################################

