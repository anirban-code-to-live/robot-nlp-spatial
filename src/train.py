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
import glob
import sys
sys.path.append('../')
from config.global_config import GlobalConfig
from src.data import get_data


config = GlobalConfig()
W, H = config.SCALE, config.SCALE
N_TOKS = config.MAX_SEQ_LEN
global N_VOCAB
N_EMBED = config.EMBED_SIZE
N_OBJS = config.UNIQUE_OBJECT_COUNT


def unet(xt):
    l2r = 1e-8

    # 93x93
    x0 = Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))(xt)  # 93x93
    x0 = Sequential([Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x0)
    x1 = Conv2D(32, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(x0)  # 91x91
    x1 = Sequential([Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x1)
    x2 = Conv2D(64, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x1)  # 45x45
    x2 = Sequential([Conv2D(64, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x2)
    x3 = Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(x2)  # 43x43
    x3 = Sequential([Conv2D(64, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x3)
    x4 = Conv2D(128, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x3)  # 21x21
    x4 = Sequential([Conv2D(128, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x4)
    x5 = Conv2D(128, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(x4)  # 19x19
    x5 = Sequential([Conv2D(128, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x5)
    x6 = Conv2D(128, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x5)  # 9x9
    x6 = Sequential([Conv2D(128, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x6)

    x6_ = Conv2DTranspose(64, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x6)  # 19x19
    x6_ = Sequential([Conv2D(64, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x6_)
    x5_ = Conv2DTranspose(128, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(
        tfk.layers.concatenate([x6_, x5], axis=-1))  # 21x21
    x5_ = Sequential([Conv2D(128, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x5_)
    x4_ = Conv2DTranspose(64, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x5_)  # 43x43
    x4_ = Sequential([Conv2D(64, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x4_)
    x3_ = Conv2DTranspose(64, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(
        tfk.layers.concatenate([x4_, x3], axis=-1))  # 45x45
    x3_ = Sequential([Conv2D(64, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x3_)
    x2_ = Conv2DTranspose(32, (3, 3), strides=2, activation='elu', kernel_regularizer=tfkr.l2(l2r))(x3_)  # 91x91
    x2_ = Sequential([Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x2_)
    x1_ = Conv2DTranspose(32, (3, 3), activation='elu', kernel_regularizer=tfkr.l2(l2r))(
        tfk.layers.concatenate([x2_, x1], axis=-1))  # 93x93
    x1_ = Sequential([Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x1_)
    x0_ = Conv2DTranspose(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))(
        tfk.layers.concatenate([x1_, x0], axis=-1))  # 93x93
    x0_ = Sequential([Conv2D(32, (1, 1), activation='elu', kernel_regularizer=tfkr.l2(l2r))])(x0_)

    attn = Conv2D(1, (1, 1), activation=None, kernel_regularizer=tfkr.l2(l2r))(x0_)
    attn = Sequential([tfkl.Reshape((-1,)), tfkl.Softmax(), tfkl.Reshape((H, W, 1))])(attn)

    return attn


def build_2d_model(args):
    l2r = 1e-9

    T, X = tfkl.Input((N_TOKS,)), tfkl.Input((H, W, 3 + N_OBJS))

    ti = tfkl.Embedding(N_VOCAB, N_EMBED, input_length=N_TOKS)(T)
    print(ti.shape)
    th = tfkm.Sequential([
        tfkl.Bidirectional(tfkl.CuDNNLSTM(128, return_sequences=True)),
        tfkl.Bidirectional(tfkl.CuDNNLSTM(128, return_sequences=True)),
        tfkl.Conv1D(256, (1,), activation='elu', kernel_regularizer=tfkr.l2(l2r)),
        tfkl.Conv1D(6, (1,), activation=None, kernel_regularizer=tfkr.l2(l2r)),
        tfkl.Softmax(axis=-2, name='lstm_attn'),
    ], name='lstm_layers')(ti)

    tia = tfkb.sum(tfkl.Reshape((N_TOKS, 1, -1))(th) * tfkl.Reshape((N_TOKS, N_EMBED, 1))(ti), axis=-3)

    Xi = tfkb.sum(X[:, :, :, 3:], axis=-1, keepdims=True)

    s1 = tfkl.Dense(N_OBJS, activation='softmax')(tia[:, :, 0])
    s1b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, N_OBJS))])(s1)
    Xs1 = tfkb.sum(X[:, :, :, 3:] * s1b, axis=-1, keepdims=True)

    s2 = tfkl.Dense(3)(tia[:, :, 1])
    s2b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, 3))])(s2)
    s2c = tfkb.sum(s2b * X[:, :, :, 2:3] - (1 - Xi) * 20, axis=-1, keepdims=True)
    Xs2 = tfkm.Sequential([tfkl.Reshape((-1, 1)), tfkl.Softmax(axis=-2), tfkl.Reshape((H, W, 1))])(s2c)
    Xs2 = Xs2 - tfkb.max(Xs2, axis=[1, 2], keepdims=True)

    s3 = tfkl.Dense(N_OBJS, activation='softmax')(tia[:, :, 2])
    s3b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, N_OBJS))])(s3)
    Xs3 = tfkb.sum(X[:, :, :, 3:] * s3b, axis=-1, keepdims=True)

    s4 = tfkl.Dense(16, activation='softmax')(tia[:, :, 3])
    s4b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, 16))])(s4)
    Xs4 = s4b * Xi

    s5 = tfkl.Dense(16, activation='softmax')(tia[:, :, 4])
    s5b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, 16))])(s5)
    Xs5 = s5b * Xi

    s6 = tfkl.Dense(16, activation='softmax')(tia[:, :, 5])
    s6b = tfkm.Sequential([tfkl.RepeatVector(W * H), tfkl.Reshape((H, W, 16))])(s6)
    Xs6 = s6b * Xi

    xt = tfkl.concatenate([Xi, Xs1, Xs2, Xs3, Xs4, Xs5, Xs6], axis=-1)

    attn = unet(xt)
    Y = tfkb.sum(attn * X[:, :, :, :2], axis=[1, 2])

    model = tfkm.Model(inputs=[T, X], outputs=[Y])

    def acc(y_pred, y_true):
        return tfkb.mean(tfkb.min(tfkb.cast((tfkb.abs(y_true-y_pred) < args.tol), 'float32'), axis=1))

    model.compile(tfk.optimizers.Adam(args.lr), 'mse', metrics=[acc])

    return model


def main(args):
    if not os.path.exists(args.params_dir):
        os.makedirs(args.params_dir)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.log_level
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    global N_VOCAB
    train_data, val_data, test_data, N_VOCAB = get_data(args)
    Ss, Ts, Xs, Ys = train_data['Is'], train_data['Ts'], train_data['Xs'], train_data['Ys']
    dSs, dTs, dXs, dYs = val_data['Is'], val_data['Ts'], val_data['Xs'], val_data['Ys']
    tSs, tTs, tXs, tYs = test_data['Is'], test_data['Ts'], test_data['Xs'], test_data['Ys']

    print('*' * 80)
    print('Train samples: {} | Validation samples: {} | Test samples: {}'.format(len(Ss), len(dSs), len(tSs)))
    print('Vocab size: {}'. format(N_VOCAB))
    print('*' * 80)

    model = build_2d_model(args)
    param_fpath = '../params/params_unet' + str(args.run) + '.h5'
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
    parser.add_argument('--dataset', default='synthetic', type=str, help='Template type.')
    parser.add_argument('--data_dir', default='../data/', type=str, help='Parent directory path for stored scenes.')
    parser.add_argument('--params_dir', default='../params/', type=str, help='Parent directory to save model weights.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--tol', default=0.05, type=float, help='Tolerance value used for accuracy metric.')
    parser.add_argument('--run', default=1, type=int, help='If multiple runs on the same model, saves model params with unique name.')
    parser.add_argument('--gpu', default='0', type=str, help='Set GPU to use.')
    parser.add_argument('--log_level', default='3', type=str, help='Set log level for tensorflow.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)