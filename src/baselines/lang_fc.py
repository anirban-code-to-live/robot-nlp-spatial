import os
import argparse
import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.models as tfkm
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as tfkb
import tensorflow.keras.regularizers as tfkr
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Sequential
from PIL import Image, ImageDraw
import glob
import sys
sys.path.append('../../')
from config.global_config import GlobalConfig
from src.baselines.baseline_data import get_lang_fc_data


config = GlobalConfig()
W, H = config.SCALE, config.SCALE
N_TOKS = config.MAX_SEQ_LEN
global N_VOCAB
N_EMBED = config.EMBED_SIZE
N_OBJS = config.UNIQUE_OBJECT_COUNT
MAX_OBJS = config.MAX_SCENE_OBJECT_COUNT

# W, H = 93, 93
# OBJS = ['apple', 'banana', 'cucumber', 'orange', 'pineapple']
OBJ_IMGS = [Image.open(fn).resize((8, 8)) for fn in sorted(glob.glob('../emojis/*.png'))]
# N_TOKS = 16
# N_VOCAB = 32
# N_EMBED = 12


def visualize(X):
    img = Image.new('RGB', (W, H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img, 'RGBA')
    ew, eh = OBJ_IMGS[0].size
    ys, xs = np.where(X[:, :, 0] != 0)
    for x, y in zip(xs, ys):
        x, y = int(x), int(y)
        idx, = np.where(X[y, x, 2:] != 0)
        idx = int(idx)
        img.paste(OBJ_IMGS[idx], box=(x - ew // 2, y - eh // 2, x + ew // 2, y + eh // 2))
    return img


def fcnet(xt):
    s1 = tfkm.Sequential([
        tfkl.Dense(32, activation=None),
        tfkl.Dense(16, activation=None),
        tfkl.Dense(8, activation=None),
        tfkl.Dense(1, activation=None)
        ])(xt)
    attn = Sequential([tfkl.Reshape((-1,)), tfkl.Softmax(), tfkl.Reshape((MAX_OBJS, 1))])(s1)

    return attn


def build_1d_model(args):
    l2r = 1e-9

    T, X = tfkl.Input((N_TOKS,)), tfkl.Input((MAX_OBJS, 3 + N_OBJS))

    # print('T: ', T.shape)
    # print('X: ', X.shape)

    ti = tfkl.Embedding(N_VOCAB, N_EMBED, input_length=N_TOKS)(T)

    # print('ti :', ti.shape)

    th = tfkm.Sequential([
        tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True)),
        tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True)),
        tfkl.Conv1D(256, (1,), activation='elu', kernel_regularizer=tfkr.l2(l2r)),
        tfkl.Conv1D(6, (1,), activation=None, kernel_regularizer=tfkr.l2(l2r)),
        tfkl.Softmax(axis=-2, name='lstm_attn'),
    ], name='lstm_layers')(ti)

    # print('th: ', th.shape)

    tia = tfkb.sum(tfkl.Reshape((N_TOKS, 1, -1))(th) * tfkl.Reshape((N_TOKS, N_EMBED, 1))(ti), axis=-3)

    # print('tia: ', tia.shape)

    Xi = tfkb.sum(X[:, :, 3:], axis=-1, keepdims=True)

    # print('Xi: ', Xi.shape)

    s1 = tfkl.Dense(N_OBJS, activation='softmax')(tia[:, :, 0])
    s1b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, N_OBJS))])(s1)
    Xs1 = tfkb.sum(X[:, :, 3:] * s1b, axis=-1, keepdims=True)

    # print('s1: ', s1.shape)
    # print('s1b: ', s1b.shape)
    # print('Xs1: ', Xs1.shape)

    s2 = tfkl.Dense(3)(tia[:, :, 1])
    s2b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, 3))])(s2)
    s2c = tfkb.sum(s2b * X[:, :, 2:3] - (1 - Xi) * 20, axis=-1, keepdims=True)
    Xs2 = tfkm.Sequential([tfkl.Reshape((-1, 1)), tfkl.Softmax(axis=-2), tfkl.Reshape((MAX_OBJS, 1))])(s2c)
    Xs2 = Xs2 - tfkb.max(Xs2, axis=[1, 2], keepdims=True)

    # print('Xs2: ', Xs2.shape)

    s3 = tfkl.Dense(N_OBJS, activation='softmax')(tia[:, :, 2])
    s3b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, N_OBJS))])(s3)
    Xs3 = tfkb.sum(X[:, :, 3:] * s3b, axis=-1, keepdims=True)

    s4 = tfkl.Dense(16, activation='softmax')(tia[:, :, 3])
    s4b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, 16))])(s4)
    Xs4 = s4b * Xi

    # print('Xs4: ', Xs2.shape)

    s5 = tfkl.Dense(16, activation='softmax')(tia[:, :, 4])
    s5b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, 16))])(s5)
    Xs5 = s5b * Xi

    s6 = tfkl.Dense(16, activation='softmax')(tia[:, :, 5])
    s6b = tfkm.Sequential([tfkl.RepeatVector(MAX_OBJS), tfkl.Reshape((MAX_OBJS, 16))])(s6)
    Xs6 = s6b * Xi

    xt = tfkl.concatenate([Xi, Xs1, Xs2, Xs3, Xs4, Xs5, Xs6], axis=-1)
    # print('xt: ', xt.shape)

    attn = fcnet(xt)
    # print('attn: ', attn.shape)
    Y = tfkb.sum(attn * X[:, :, :2], axis=[1])
    # print('Y: ', Y.shape)

    model = tfkm.Model(inputs=[T, X], outputs=[Y])

    def acc(y_pred, y_true):
        return tfkb.mean(tfkb.min(tfkb.cast((tfkb.abs(y_true-y_pred) < args.tol), 'float32'), axis=1))

    model.compile(tfk.optimizers.Adam(args.lr), 'mse', metrics=[acc])

    return model


def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.log_level
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Ss, Ts, Xs, Ys = make_dataset(N=100)
    # dSs, dTs, dXs, dYs = make_dataset(N=10)
    global N_VOCAB
    train_data, val_data, test_data, N_VOCAB = get_lang_fc_data(args)
    Ss, Ts, Xs, Ys = train_data['Is'], train_data['Ts'], train_data['Xs'], train_data['Ys']
    dSs, dTs, dXs, dYs = val_data['Is'], val_data['Ts'], val_data['Xs'], val_data['Ys']
    tSs, tTs, tXs, tYs = test_data['Is'], test_data['Ts'], test_data['Xs'], test_data['Ys']
    print('*' * 80)
    print('Train samples: {} | Validation samples: {} | Test samples: {}'.format(len(Ss), len(dSs), len(tSs)))
    print('Vocab size: {}'. format(N_VOCAB))
    # print(Ts.shape, Xs.shape, Ys.shape)
    print('*' * 80)

    # print(Ss[0], Ys[0])
    # img = visualize(Xs[0])
    # img.save('../tmp/vis.png')

    model = build_1d_model(args)
    param_fpath = '../../params/params_fc' + str(args.run) + '.h5'
    os.system('rm ' + param_fpath)
    checkpointer = tfk.callbacks.ModelCheckpoint(param_fpath, monitor='val_acc', verbose=0,
                                                 save_weights_only=True, save_best_only=True, mode='max', period=1)
    model.fit(x=[Ts, Xs], y=[Ys], validation_data=([dTs, dXs], [dYs]), batch_size=args.batch_size, epochs=args.epochs, callbacks=[checkpointer])

    # Loads the weights
    model.load_weights(param_fpath)

    # Re-evaluate the model
    print('*' * 80)
    print('Validation set performance')
    model.evaluate([dTs, dXs], [dYs], batch_size=args.batch_size)
    print('*' * 80)
    print('Test set performance')
    print('-' * 80)
    model.evaluate(x=[tTs, tXs], y=[tYs], batch_size=args.batch_size)
    print('*' * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for create dataset.')
    parser.add_argument('--template', default='SP_SO_REL_SOR', type=str, help='Template type.')
    parser.add_argument('--action', default='pick', type=str, help='Action type. pick or pick_and_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Parent directory path for stored images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Parent directory path for stored scenes.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--tol', default=0.05, type=float, help='Tolerance value used for accuracy metric.')
    parser.add_argument('--gpu', default='0', type=str, help='Set GPU to use.')
    parser.add_argument('--log_level', default='3', type=str, help='Set log level for tensorflow.')
    parser.add_argument('--run', default=1, type=int,
                        help='If multiple runs on the same model, saves model params with unique name.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)