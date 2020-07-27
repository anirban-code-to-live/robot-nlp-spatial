import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################
# Device configuration
#######################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#######################################


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out


class EmbeddingLayer(nn.Module):
    """

    Parameters
    ------------
    embed_dim: Dimension of embedding layer
    vocab_size: Count of words in vocabulary

    Input
    -------
    x: # (batch_size, seq_len)

    Returns
    --------
    x: # (seq_len, batch_size, embed_dim)

    """

    def __init__(self, embed_dim, vocab_size, pad_idx=0, trainable=True):
        super(EmbeddingLayer, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if not trainable:
            self.embed_layer.weight.requires_grad = False

    def forward(self, x):
        x = self.embed_layer(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        return x


class BiLSTMEncoder(nn.Module):
    """

    Parameters
    ------------
    embed_dim: Embedding dimension for input data
    out_dim: Output dimension of BiLSTM network
    num_layers: Number of BiLSTM layers

    Inputs
    -------
    x: # (seq_len, batch_size, embed_dim)
    hc: Tuple of hidden_state, cell_state, both of shape # (2*num_layers, batch_size, out_dim)

    Returns
    --------
    f_out: # (seq_len, batch_size, out_dim)
    b_out: # (seq_len, batch_size, out_dim)
    fh_n: # (batch_size, out_dim)
    bh_n: # (batch_size, out_dim)

    """

    def __init__(self, rnn_embed_dim, rnn_dim, rnn_num_layers, rnn_dropout):
        super(BiLSTMEncoder, self).__init__()
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dim = rnn_dim
        self.lstm = nn.LSTM(rnn_embed_dim, rnn_dim, rnn_num_layers, dropout=rnn_dropout, bidirectional=True)

    def forward(self, x):
        # Set initial values for hidden and cell states
        h_0 = torch.zeros(self.rnn_num_layers * 2, x.size(1), self.rnn_dim).to(
            device)  # (2*num_layers, batch_size, out_dim)
        c_0 = torch.zeros(self.rnn_num_layers * 2, x.size(1), self.rnn_dim).to(
            device)  # (2*num_layers, batch_size, out_dim)
        #         h_0, c_0 = hc # (2*num_layers, batch_size, out_dim)

        # Forward propagate to LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # h_n, c_n: # (2*num_layers, batch_size, out_dim)
        # out: # (seq_len, batch_size, 2*out_dim)

        out = out.view(x.size(0), x.size(1), 2, self.rnn_dim)  # (seq_len, batch_size, 2, out_dim)
        f_out = out[:, :, 0, :]  # (seq_len, batch_size, out_dim)
        b_out = out[:, :, 1, :]  # (seq_len, batch_size, out_dim)
        h_n = h_n.view(self.rnn_num_layers, 2, x.size(1), self.rnn_dim)  # (num_layers, 2, batch_size, out_dim)
        fh_n = h_n[-1, 0, :, :]  # (batch_size, out_dim)
        bh_n = h_n[-1, 1, :, :]  # (batch_size, out_dim)

        return f_out, b_out, fh_n, bh_n


def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0,
                 rnn_emb_dim=256, rnn_dim=128, rnn_num_layers=2, rnn_dropout=0,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(32,)):
        super(BiLSTMModel, self).__init__()
        emb_kwargs = {
            'embed_dim': rnn_emb_dim,
            'vocab_size': vocab_size,
            'pad_idx': pad_idx
        }
        self.embed_layer = EmbeddingLayer(**emb_kwargs)
        rnn_kwargs = {
            'rnn_embed_dim': rnn_emb_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = BiLSTMEncoder(**rnn_kwargs)

        classifier_kwargs = {
            'input_dim': 2 * rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': 2,
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions, feats):
        embs = self.embed_layer(questions)  # (seq_len, batch_size, emb_dim)
        f_out, b_out, fh_n, bh_n = self.rnn(embs)
        q_feats = torch.cat((fh_n, bh_n), dim=-1)  # (batch_size, rnn_dim)
        outs = self.classifier(q_feats)  # (batch_size, 2)
        return outs


def build_cnn(feat_dim=(512, 1, 1),
              res_block_dim=128,
              num_res_blocks=0,
              proj_dim=128,
              pooling='maxpool'):
    C, H, W = feat_dim
    layers = []
    if num_res_blocks > 0:
        layers.append(nn.Conv2d(C, res_block_dim, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        C = res_block_dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(C))
    if proj_dim > 0:
        layers.append(nn.Conv2d(C, proj_dim, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        C = proj_dim
    if pooling == 'maxpool':
        layers.append(nn.MaxPool2d(kernel_size=1, stride=1))
        # H, W = H // 2, W // 2
    return nn.Sequential(*layers), (C, H, W)


class CNNBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0,
                 rnn_emb_dim=256, rnn_dim=128, rnn_num_layers=2, rnn_dropout=0,
                 cnn_feat_dim=(512, 1, 1), cnn_res_block_dim=128, cnn_num_res_blocks=0,
                 cnn_proj_dim=64, cnn_pooling='maxpool',
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(32,), ):
        super(CNNBiLSTMModel, self).__init__()
        emb_kwargs = {
            'embed_dim': rnn_emb_dim,
            'vocab_size': vocab_size,
            'pad_idx': pad_idx
        }
        self.embed_layer = EmbeddingLayer(**emb_kwargs)
        rnn_kwargs = {
            'rnn_embed_dim': rnn_emb_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = BiLSTMEncoder(**rnn_kwargs)

        cnn_kwargs = {
            'feat_dim': cnn_feat_dim,
            'res_block_dim': cnn_res_block_dim,
            'num_res_blocks': cnn_num_res_blocks,
            'proj_dim': cnn_proj_dim,
            'pooling': cnn_pooling,
        }
        self.cnn, (C, H, W) = build_cnn(**cnn_kwargs)

        classifier_kwargs = {
            'input_dim': 2 * rnn_dim + cnn_proj_dim,
            'hidden_dims': fc_dims,
            'output_dim': 2,
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions, feats):
        embs = self.embed_layer(questions)  # (seq_len, batch_size, emb_dim)
        f_out, b_out, fh_n, bh_n = self.rnn(embs)
        q_feats = torch.cat((fh_n, bh_n), dim=-1)  # (batch_size, rnn_dim)
        img_feats = self.cnn(feats).squeeze(3).squeeze(2)  # (batch_size, cnn_proj_dim)
        cat_feats = torch.cat([q_feats, img_feats], dim=1)
        outs = self.classifier(cat_feats)
        return outs


class StackedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def forward(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D
        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = torch.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W), dim=1).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
        next_u = u + v_tilde
        return next_u


class SACNNBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0,
                 rnn_emb_dim=256, rnn_dim=128, rnn_num_layers=2, rnn_dropout=0,
                 cnn_feat_dim=(1024, 14, 14),
                 stacked_attn_dim=64, num_stacked_attn=2,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(32,)):
        super(SACNNBiLSTMModel, self).__init__()
        emb_kwargs = {
            'embed_dim': rnn_emb_dim,
            'vocab_size': vocab_size,
            'pad_idx': pad_idx
        }
        self.embed_layer = EmbeddingLayer(**emb_kwargs)
        rnn_kwargs = {
            'rnn_embed_dim': rnn_emb_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = BiLSTMEncoder(**rnn_kwargs)

        C, H, W = cnn_feat_dim
        self.image_proj = nn.Conv2d(C, 2 * rnn_dim, kernel_size=1, padding=0)
        self.stacked_attns = []
        for i in range(num_stacked_attn):
            sa = StackedAttention(2 * rnn_dim, stacked_attn_dim)
            self.stacked_attns.append(sa)
            self.add_module('stacked-attn-%d' % i, sa)

        classifier_kwargs = {
            'input_dim': 2 * rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': 2,
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions, feats):
        embs = self.embed_layer(questions)  # (seq_len, batch_size, emb_dim)
        f_out, b_out, fh_n, bh_n = self.rnn(embs)
        u = torch.cat((fh_n, bh_n), dim=-1)  # (batch_size, rnn_dim)
        v = self.image_proj(feats)

        for sa in self.stacked_attns:
            u = sa(v, u)

        out = self.classifier(u)
        return out


if __name__ == '__main__':
    batch_size = 4
    max_seq_len = 12

    # RNN Params
    rnn_emb_dim = 128
    vocab_size = 27
    pad_idx = 0
    rnn_dim = 64
    rnn_num_layers = 2
    rnn_dropout = 0.2

    # CNN Params
    cnn_feat_dim = (512, 1, 1)
    cnn_res_block_dim = 128
    cnn_num_res_blocks = 1
    cnn_proj_dim = 64
    cnn_pooling = 'maxpool'

    # FC Params
    fc_dims = (32,)
    fc_use_batchnorm = False
    fc_dropout = 0

    # Input data
    seqs = torch.randint(0, 5, (batch_size, max_seq_len))
    feats = torch.rand((batch_size, 512, 1, 1))

    sacnnlstm_kwargs = {
        'vocab_size': vocab_size,
        'pad_idx': pad_idx,

        'rnn_emb_dim': rnn_emb_dim,
        'rnn_dim': rnn_dim,
        'rnn_num_layers': rnn_num_layers,
        'rnn_dropout': rnn_dropout,

        'cnn_feat_dim': cnn_feat_dim,

        'fc_dims': fc_dims,
        'fc_use_batchnorm': fc_use_batchnorm,
        'fc_dropout': fc_dropout
    }

    sa_cnn_lstm = SACNNBiLSTMModel(**sacnnlstm_kwargs)
    sa_cnn_lstm_out = sa_cnn_lstm(seqs, feats)
    print('SA-CNN-LSTM out shape: ', sa_cnn_lstm_out.shape)

    # cnnlstm_kwargs = {
    #     'vocab_size': vocab_size,
    #     'pad_idx': pad_idx,
    #
    #     'rnn_emb_dim': rnn_emb_dim,
    #     'rnn_dim': rnn_dim,
    #     'rnn_num_layers': rnn_num_layers,
    #     'rnn_dropout': rnn_dropout,
    #
    #     'cnn_feat_dim': cnn_feat_dim,
    #     'cnn_res_block_dim': cnn_res_block_dim,
    #     'cnn_num_res_blocks': cnn_num_res_blocks,
    #     'cnn_proj_dim': cnn_proj_dim,
    #     'cnn_pooling': cnn_pooling,
    #
    #     'fc_dims': fc_dims,
    #     'fc_use_batchnorm': fc_use_batchnorm,
    #     'fc_dropout': fc_dropout
    # }
    #
    # cnn_lstm = CNNBiLSTMModel(**cnnlstm_kwargs)
    # cnn_lstm_out = cnn_lstm(seqs, feats)
    # print('CNN-BiLSTM out shape: ', cnn_lstm_out.shape)

    # lstm_kwargs = {
    #     'vocab_size': vocab_size,
    #     'pad_idx': pad_idx,
    #
    #     'rnn_emb_dim': rnn_emb_dim,
    #     'rnn_dim': rnn_dim,
    #     'rnn_num_layers': rnn_num_layers,
    #     'rnn_dropout': rnn_dropout,
    #
    #     'fc_dims': fc_dims,
    #     'fc_use_batchnorm': fc_use_batchnorm,
    #     'fc_dropout': fc_dropout,
    # }
    #
    # lstm = BiLSTMModel(**lstm_kwargs)
    # lstm_out = lstm(seqs, None)
    # print('Full LSTM Model output shape: ', lstm_out.shape)
