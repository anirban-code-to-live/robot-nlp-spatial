import numpy as np
import torch
import torch.nn as nn

#######################################
# Device configuration
#######################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#######################################


class EmbeddingLayer(nn.Module):
    """

    Parameters
    ------------
    embed_dim: Dimension of embedding layer
    vocab_size: Count of words in vocabulary
    weight_mat: Lookup table for GloVe embedding. Shape # (vocab_size, embed_dim)

    Input
    -------
    x: # (batch_size, seq_len)

    Returns
    --------
    x: # (seq_len, batch_size, embed_dim)

    """

    def __init__(self, embed_dim, vocab_size, weight_mat, pad_idx=0, trainable=True):
        super(EmbeddingLayer, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # self.embed_layer.weight = nn.Parameter(torch.tensor(weight_mat))
        # if not trainable:
        #     self.embed_layer.weight.requires_grad = False

    def forward(self, x):
        x = self.embed_layer(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        return x


class BiLSTMNet(nn.Module):
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

    def __init__(self, embed_dim, out_dim, num_layers=2):
        super(BiLSTMNet, self).__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.lstm = nn.LSTM(embed_dim, out_dim, num_layers, bidirectional=True)

    def forward(self, x):
        # Set initial values for hidden and cell states
        h_0 = torch.zeros(self.num_layers * 2, x.size(1), self.out_dim).to(
            device)  # (2*num_layers, batch_size, out_dim)
        c_0 = torch.zeros(self.num_layers * 2, x.size(1), self.out_dim).to(
            device)  # (2*num_layers, batch_size, out_dim)
        #         h_0, c_0 = hc # (2*num_layers, batch_size, out_dim)

        # Forward propagate to LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # h_n, c_n: # (2*num_layers, batch_size, out_dim)
        # out: # (seq_len, batch_size, 2*out_dim)

        out = out.view(x.size(0), x.size(1), 2, self.out_dim)  # (seq_len, batch_size, 2, out_dim)
        f_out = out[:, :, 0, :]  # (seq_len, batch_size, out_dim)
        b_out = out[:, :, 1, :]  # (seq_len, batch_size, out_dim)
        h_n = h_n.view(self.num_layers, 2, x.size(1), self.out_dim)  # (num_layers, 2, batch_size, out_dim)
        fh_n = h_n[-1, 0, :, :]  # (batch_size, out_dim)
        bh_n = h_n[-1, 1, :, :]  # (batch_size, out_dim)

        return f_out, b_out, fh_n, bh_n


class AttentionNet(nn.Module):
    """

    Parameters
    -------------
    emb_dim: Dimension of input(hidden and encoder vectors)

    Inputs
    -------
    hidden: tuple of (forward, backward) hidden vecs # (batch_size, emb_dim)
    enc_outs: tuple of (forward, backward) encoder outs # (batch_size, seq_len, emb_dim)

    Returns
    --------
    atn_out: Attention encoder output # (batch_size, 2*emb_dim)

    """

    def __init__(self, emb_dim, method='dot'):
        super(AttentionNet, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate method')
        if self.method == 'general':
            self.attn = torch.nn.Linear(emb_dim, emb_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(2 * emb_dim, emb_dim)
            self.v = nn.Parameter(torch.FloatTensor(emb_dim))

    def dot_score(self, hidden, enc_outs):
        attn_energies = torch.bmm(enc_outs, hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        return attn_energies

    def general_score(self, hidden, enc_outs):
        enc_outs = self.attn(enc_outs)  # (batch_size, seq_len, emb_dim)
        attn_energies = torch.bmm(enc_outs, hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        return attn_energies

    def concat_score(self, hidden, enc_outs):
        hidden = hidden.expand(enc_outs.size(1), -1, -1).permute(1, 0, 2)  # (batch_size, seq_len, emb_dim)
        enc_outs = torch.cat((enc_outs, hidden), 2)  # (batch_size, seq_len, 2*emb_dim)
        enc_outs = self.attn(enc_outs).tanh()  # (batch_size, seq_len, emb_dim)
        v = self.v.expand(enc_outs.size(0), -1)  # (batch_size, emb_dim)
        attn_energies = torch.bmm(enc_outs, v.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        return attn_energies

    def attention_net(self, hidden, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, enc_outs)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # (batch_size, seq_len)
        attn_out = torch.bmm(enc_outs.transpose(1, 2), attn_energies.unsqueeze(2)).squeeze(2)  # (batch_size, emb_dim)
        return attn_out

    def forward(self, hidden, enc_outs):
        f_h, b_h = hidden
        f_enc, b_enc = enc_outs
        atnf_out = self.attention_net(f_h, f_enc)  # (batch_size, emb_dim)
        atnb_out = self.attention_net(b_h, b_enc)  # (batch_size, emb_dim)
        atn_out = torch.cat((atnf_out, atnb_out), dim=1)  # (batch_size, 2*emb_dim)
        return atn_out


class AnchorNet(nn.Module):
    """

    Parameters
    ------------
    in_dim: Attention vector input dimension
    out_dim: Output dimension of attention vector though FC layer

    Inputs
    -------
    x: # (batch_size, out_dim, 2)
    attn_out: # (batch_size, in_dim)

    Returns
    --------
    out: # (batch_size, 2)

    """

    def __init__(self, in_dim, out_dim):
        super(AnchorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x, attn_out):
        attn_out = self.fc(attn_out)  # (batch_size, out_dim)
        attn_out = attn_out.unsqueeze(2).permute(0, 2, 1)  # (batch_size, 1, out_dim)
        out = torch.bmm(attn_out, x).squeeze(1)  # (batch_size, 2)
        return out


class OffsetNet(nn.Module):
    """

    Parameters
    ------------
    in_dim: Attention vector input dimension

    Inputs
    -------
    off_x: # (batch_size, 1, 1)
    attn_out: # (batch_size, in_dim)

    Returns
    --------
    out: # (batch_size, 2)

    """

    def init_weight(self):
        T = 0.166
        weight = np.array([
            [-T, -T],
            [-T, 0.],
            [-T, T],
            [0., -T],
            [0., T],
            [T, -T],
            [T, 0.],
            [T, T],

        ], dtype=np.float32)
        weight = np.reshape(weight, (16, -1))
        self.linear.weight.data = torch.tensor(weight, dtype=torch.float32)

    def __init__(self, in_dim):
        super(OffsetNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.Softmax(dim=1)
        )

        self.linear = nn.Linear(1, 16, bias=False)
        self.init_weight()
        self.linear.weight.requires_grad = False

    def forward(self, off_x, attn_out):
        attn_out = self.fc(attn_out)  # (batch_size, 8)
        attn_out = attn_out.unsqueeze(2).permute(0, 2, 1)  # (batch_size, 1, 8)
        off_x = self.linear(off_x)  # (batch_size, 1, 16)
        off_x = torch.reshape(off_x, (off_x.size(0), 8, 2))  # (batch_size, 8, 2)
        out = torch.bmm(attn_out, off_x).squeeze(1)  # (batch_size, 2)
        return out


class Model(nn.Module):
    """

    Parameters
    -----------
    emb_dim: Dimension of sentence embeddings
    out_dim: Bi-LSTM and Attention Network output dimension
    anc_dim: Output dimension of FC layer in Anchor Network
    vocab_size: Count of words in vocabulary
    weight_mat: GloVe embeddings # (vocab_size, 300)
    pad_idx: Padding token index

    Inputs
    --------
    s: Input sentences # (batch_size, seq_len)
    x: Input states # (batch_size, 22, 2)

    Returns
    --------
    out: Vector with source and target co-ordinates # (batch_size, 4)

    """

    def __init__(self, emb_dim, out_dim, anc_dim, vocab_size, weight_mat, pad_idx=0):
        super(Model, self).__init__()
        self.embed_layer = EmbeddingLayer(emb_dim, vocab_size, weight_mat, pad_idx)
        self.s_bilstm = BiLSTMNet(emb_dim, out_dim)
        self.t_bilstm = BiLSTMNet(emb_dim, out_dim)
        self.o_bilstm = BiLSTMNet(emb_dim, out_dim)
        self.s_attn_net = AttentionNet(out_dim)
        self.t_attn_net = AttentionNet(out_dim)
        self.o_attn_net = AttentionNet(out_dim)
        self.s_anchor_net = AnchorNet(2 * out_dim, anc_dim)
        self.t_anchor_net = AnchorNet(2 * out_dim, anc_dim)
        self.t_offset_net = OffsetNet(2 * out_dim)

    def forward(self, s, x):
        e = self.embed_layer(s)  # (seq_len, batch_size, emb_dim)
        fs_out, bs_out, fs_h, bs_h = self.s_bilstm(e)
        ft_out, bt_out, ft_h, bt_h = self.t_bilstm(e)
        fo_out, bo_out, fo_h, bo_h = self.o_bilstm(e)
        # Shapes of fs_out, bs_out, ft_out, bt_out: # (seq_len, batch_size, out_dim)
        # Shapes of fs_h, bs_h, ft_h, bt_h: # (batch_size, out_dim)
        fs_out, bs_out = fs_out.permute(1, 0, 2), bs_out.permute(1, 0, 2)
        ft_out, bt_out = ft_out.permute(1, 0, 2), bt_out.permute(1, 0, 2)
        fo_out, bo_out = fo_out.permute(1, 0, 2), bo_out.permute(1, 0, 2)
        # Shapes of fs_out, bs_out, ft_out, bt_out: # (batch_size, seq_len, out_dim)
        s_hidden, t_hidden, o_hidden = (fs_h, bs_h), (ft_h, bt_h), (fo_h, bo_h)
        s_enc_outs, t_enc_outs, o_enc_outs = (fs_out, bs_out), (ft_out, bt_out), (fo_out, bo_out)
        s_attn_out = self.s_attn_net(s_hidden, s_enc_outs)  # (batch_size, 2*out_dim)
        t_attn_out = self.t_attn_net(t_hidden, t_enc_outs)  # (batch_size, 2*out_dim)
        o_attn_out = self.o_attn_net(o_hidden, o_enc_outs)  # (batch_size, 2*out_dim)
        s_out = self.s_anchor_net(x[:, :25, :], s_attn_out)  # (batch_size, 2)
        t_anc_out = self.t_anchor_net(x[:, :25, :], t_attn_out)  # (batch_size, 2)
        t_off_out = self.t_offset_net(x[:, 25, 0:1], o_attn_out)  # (batch_size, 2)
        t_out = t_anc_out + t_off_out  # (batch_size, 2)
        out = torch.cat((s_out, t_out), dim=1)  # (batch_size, 4)
        return out


import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from src.baselines.baseline_data import get_bisk_rnn_data
from src.helper import *
from config.global_config import GlobalConfig

config = GlobalConfig()

#######################################
# Hyper-parameters
#######################################
pad_idx = config.PAD_TOKEN
anc_dim = config.MAX_SCENE_OBJECT_COUNT
#######################################

#######################################
# Device configuration
#######################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#######################################

global logger


def accuracy_metric(pred, target, tolerance=0.05):
    acc = torch.min(torch.abs(pred - target) < tolerance, dim=1).values
    acc = acc.type(torch.FloatTensor)
    return torch.sum(acc)


def block_distance_metric(pred, target):
    distance = torch.sum(torch.norm(pred - target, dim=1))
    return distance


#######################################
# Train Model
#######################################
def train(model, optimiser, criterion, train_loader, step, num_epochs, total_epochs, print_every=1):
    model.train()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        iter_loss = []
        accs, ds = [], []
        total = 0
        for i, (seqs, states, targets) in enumerate(train_loader):
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            if pred_type == 'source': preds = preds[:, :2]
            elif pred_type == 'target': preds = preds[:, 2:]
            loss = criterion(preds, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            iter_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets)
            accs.append(acc)
            ds.append(block_distance_metric(preds, targets))

        epoch_loss = np.mean(iter_loss)
        accuracy = torch.sum(torch.tensor(accs)).item() / total
        avg_blk_dis = torch.sum(torch.tensor(ds)).item() / (total * 0.166)

        if epoch % print_every == 0:
            epoch_time = time.time() - start_time
            logger.info(
                ' Epoch: [{}/{}] | Time: {:.3f} s/epoch | Loss: {:.6f} | Accuracy: {:.2f} % | Avg. Dis: {:.3f}'.format(
                    (step - 1) * num_epochs + epoch, total_epochs, epoch_time / print_every, epoch_loss, accuracy * 100, avg_blk_dis
                ))
            start_time = time.time()


#######################################
# Evaluate model
#######################################
def evaluate(model, criterion, dev_loader, end_of_epoch, start_time):
    model.eval()
    with torch.no_grad():
        total_loss = []
        accs, ds = [], []
        total = 0
        for seqs, states, targets in dev_loader:
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            if pred_type == 'source': preds = preds[:, :2]
            elif pred_type == 'target': preds = preds[:, 2:]
            loss = criterion(preds, targets)
            total_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets)
            accs.append(acc)
            ds.append(block_distance_metric(preds, targets))

        loss = np.mean(total_loss)
        acc = torch.sum(torch.tensor(accs)).item() / total
        avg_blk_dis = torch.sum(torch.tensor(ds)).item() / (total * 0.166)

        logger.info('-' * 120)
        logger.info(
            ' End of epoch {} | Time: {:.2f}s | Loss: {:.6f} | Accuracy: {:.2f} % | Avg. Dis: {:.3f}'.format(
                end_of_epoch, (time.time() - start_time), loss, acc * 100, avg_blk_dis
            ))
        logger.info('-' * 120)

        return loss, acc, avg_blk_dis, total


#######################################
# Test model
#######################################
def test(model, criterion, loader):
    model.eval()
    with torch.no_grad():
        total_loss = []
        accs, ds = [], []
        total = 0
        for (seqs, states, targets) in loader:
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            if pred_type == 'source': preds = preds[:, :2]
            elif pred_type == 'target': preds = preds[:, 2:]
            loss = criterion(preds, targets)
            total_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets)
            accs.append(acc)
            ds.append(block_distance_metric(preds, targets))

        loss = np.mean(total_loss)
        acc = torch.sum(torch.tensor(accs)).item() / total
        avg_blk_dis = torch.sum(torch.tensor(ds)).item() / (total * 0.166)
        return loss, acc, avg_blk_dis, total


#######################################
# Save model
#######################################
def save_model(model, optimiser, val_acc, val_loss, model_name, model_dir, save_at):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.makedirs(os.path.join(model_dir, model_name))
    torch.save({
        'model': model.state_dict(),
        'opt': optimiser.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss
    }, os.path.join(os.path.join(model_dir, model_name), '{}.tar'.format(save_at)))


#######################################
# Parse args
#######################################
def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for blocks-nlp')
    parser.add_argument('--name', default='blocks-nlp', help='Set run name for saving/restoring the files')

    parser.add_argument('--template', default='SP_SO_ABS', type=str, help='Template type.')
    parser.add_argument('--action', default='pick', type=str, help='Action type. pick or pick_and_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str,
                        help='Parent directory path for stored images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str,
                        help='Parent directory path for stored scenes.')
    parser.add_argument('--model', default='lstm', type=str, help='Model: lstm/cnn/sa')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')

    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--steps', default=1, type=int, help='Number of validation steps')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    parser.add_argument('--emb_dim', default=256, type=int, help='Embedding layer output dimension')
    parser.add_argument('--hid_dim', default=256, type=int, help='Hidden layer output dimension')
    parser.add_argument('--prediction', default='source', type=str, help='Prediction type: source/target')
    parser.add_argument('--checkpoint', default=None, type=int, help='Epoch checkpoint for model loading')
    parser.add_argument('--logdir', default='./log/', help='Log directory')
    parser.add_argument('--modeldir', default='./saved_model/', help='Model save directory')
    parser.add_argument('--config', default='../../config/', help='Config directory')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode: ON')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Debug mode: OFF')
    parser.add_argument('--print_every', default=1, type=int, help='Print training progress at interval of epcohs.')
    parser.add_argument('--gpu', default='0', type=str, help='Set GPU to use.')
    parser.add_argument('--log_level', default='3', type=str, help='Set log level for tensorflow.')
    parser.add_argument('--run', default=1, type=int,
                        help='If multiple runs on the same model, saves model params with unique name.')
    return parser.parse_args()


if __name__ == '__main__':
    #######################################
    # Parse command line arguments
    #######################################
    args = parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.log_level
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_epochs = args.epochs
    total_val_step = args.steps
    checkpoint = args.checkpoint
    log_dir = args.logdir
    model_dir = args.modeldir
    config_dir = args.config
    name = args.name
    learning_rate = args.lr
    pred_type = args.prediction
    debug = args.debug
    emb_dim = args.emb_dim
    out_dim = args.hid_dim

    #######################################
    # Define logger
    #######################################
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(name, log_dir, config_dir)
    logger.info('**' * 80)
    logger.info(args)
    logger.info('**' * 80)

    #######################################
    # Data Loader
    #######################################
    train_loader, dev_loader, test_loader, vocab_size = get_bisk_rnn_data(args)

    #######################################
    # Weight matrix for embedding layer
    #######################################
    weight_mat = np.random.uniform(-1, 1, (vocab_size, emb_dim)).astype(np.float32)
    logger.info('Shape of embedding matrix: {}\n'.format(weight_mat.shape))
    #######################################

    #######################################
    # Model
    #######################################
    model = Model(emb_dim, out_dim, anc_dim, vocab_size, weight_mat, pad_idx)
    if checkpoint is not None:
        # If loading on same machine the model was trained on
        saved_model = torch.load(os.path.join(os.path.join(model_dir, name), '{}.tar'.format(checkpoint)))
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
        model_sd = saved_model['model']
        model.load_state_dict(model_sd)
    model = model.to(device)

    #######################################
    # Loss and Optimiser
    #######################################
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if checkpoint is not None:
        optimiser_sd = saved_model['opt']
        optimiser.load_state_dict(optimiser_sd)
        print('\nModel training starting from checkpoint {}\n'.format(checkpoint))

    #######################################
    # Training and Validate
    #######################################
    logger.info('**' * 10)
    logger.info('  TRAINING  ')
    logger.info('**' * 10)

    best_val_acc = float("-inf")
    best_model, best_optimiser, best_val_loss, best_val_blk_dis = None, None, None, None

    for step in range(1, total_val_step + 1):
        step_start_time = time.time()
        train(model, optimiser, criterion, train_loader,
              step=step,
              num_epochs=num_epochs,
              total_epochs=total_val_step * num_epochs,
              print_every=args.print_every)
        val_loss, val_acc, val_blk_dis, _ = evaluate(model, criterion, dev_loader,
                                                          end_of_epoch=step * num_epochs,
                                                          start_time=step_start_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_optimiser = optimiser
            best_val_loss = val_loss
            best_val_blk_dis = val_blk_dis

    #######################################
    # Save best model and optimiser
    #######################################
    save_at = checkpoint + total_val_step * num_epochs if checkpoint else total_val_step * num_epochs
    save_model(best_model, best_optimiser, best_val_acc, best_val_loss, name, model_dir, save_at)

    #######################################
    # VALIDATE MODEL
    #######################################
    logger.info('**' * 20)
    logger.info('  VALIDATION  ')
    logger.info('--' * 20)
    logger.info(' [[ Best Model ]] Val Acc:  {:.2f} | Val Loss:  {:.6f} | Avg. Block Dist.: {:.3f}'.format(
        best_val_acc * 100, best_val_loss, best_val_blk_dis)
    )
    logger.info('**' * 20)

    #######################################
    # TEST MODEL
    #######################################
    logger.info('**' * 20)
    logger.info('  TEST  ')
    test_loss, test_acc, test_blk_dis, _ = test(best_model, criterion, test_loader)
    logger.info('--' * 20)
    logger.info(' [[ Best Model ]] Test Acc:  {:.2f} | Test Loss:  {:.6f} | Avg. Block Dist.: {:.3f}'.format(
        test_acc * 100, test_loss, test_blk_dis)
    )
    logger.info('**' * 20)
