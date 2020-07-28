import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
from baseline_data import get_data
from models import BiLSTMModel, CNNBiLSTMModel, SACNNBiLSTMModel
import sys
sys.path.append('../')
from helper import *
from config.global_config import GlobalConfig

config = GlobalConfig()

#######################################
# Hyper-parameters
#######################################
pad_idx = config.PAD_TOKEN
#######################################

#######################################
# Device configuration
#######################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#######################################

global logger


def accuracy_metric(pred, target, tolerance=0.025):
    acc = torch.min(torch.abs(pred - target) < tolerance, dim=1).values
    acc = acc.type(torch.FloatTensor)
    return torch.sum(acc)


def mse_metric(pred, target):
    mse = torch.sum(torch.norm(pred - target, dim=1) ** 2)
    return mse


#######################################
# Train Model
#######################################
def train(model, optimiser, criterion, train_loader, step, num_epochs, total_epochs, print_every=10, tolerance=0.025):
    model.train()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        iter_loss = []
        accs = []
        total = 0
        for i, (seqs, states, targets) in enumerate(train_loader):
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            loss = criterion(preds, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            iter_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets, tolerance=tolerance)
            accs.append(acc)

        epoch_loss = np.mean(iter_loss)
        accuracy = torch.sum(torch.tensor(accs)).item() / total

        if epoch % print_every == 0:
            epoch_time = time.time() - start_time
            logger.info(
                ' Epoch: [{}/{}] | Time: {:.3f} s/epoch | Loss: {:.6f} | Acc: {:.2f} %'.format(
                    (step - 1) * num_epochs + epoch, total_epochs, epoch_time / print_every, epoch_loss, accuracy * 100
                ))
            start_time = time.time()


#######################################
# Evaluate model
#######################################
def evaluate(model, criterion, dev_loader, end_of_epoch, start_time, tolerance=0.025):
    model.eval()
    with torch.no_grad():
        total_loss = []
        accs = []
        total = 0
        for seqs, states, targets in dev_loader:
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            loss = criterion(preds, targets)
            total_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets, tolerance=tolerance)
            accs.append(acc)

        val_loss = np.mean(total_loss)
        val_acc = torch.sum(torch.tensor(accs)).item() / total

        logger.info('-' * 120)
        logger.info(
            ' End of epoch {} | Time: {:.2f}s | Valid Loss: {:.6f} | Overall Acc: {:.2f} %'.format(
                end_of_epoch, (time.time() - start_time), val_loss, val_acc * 100
            ))
        logger.info('-' * 120)

        return val_loss, val_acc, total


#######################################
# Test model
#######################################
def test(model, criterion, test_loader, tolerance=0.025):
    model.eval()
    with torch.no_grad():
        total_loss = []
        accs, mse = [], []
        total = 0
        for seqs, states, targets in test_loader:
            seqs = seqs.to(device)
            states = states.to(device)
            targets = targets.to(device)
            preds = model(seqs, states)
            loss = criterion(preds, targets)
            total_loss.append(loss.item())
            total += targets.size(0)
            acc = accuracy_metric(preds, targets, tolerance)
            accs.append(acc)
            err = mse_metric(preds, targets)
            mse.append(err)

        test_loss = np.mean(total_loss)
        test_acc = torch.sum(torch.tensor(accs)).item() / total
        test_mse = torch.sum(torch.tensor(mse)).item() / total

        return test_loss, test_acc, test_mse, total


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
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--steps', default=1, type=int, help='Number of validation steps')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--tol', default=0.05, type=float, help='Tolerance value used for accuracy metric.')

    parser.add_argument('--rnn_emb_dim', default=256, type=int, help='Output dimension of embedding layer.')
    parser.add_argument('--rnn_dim', default=128, type=int, help='LSTM output dimension')
    parser.add_argument('--rnn_num_layers', default=2, type=int, help='Number of LSTM layers')
    parser.add_argument('--rnn_dropout', default=0.4, type=float, help='Dropout value for LSTM')

    parser.add_argument('--cnn_feat_dim', default=(2048, 1, 1), help='Input feature dimension')
    parser.add_argument('--cnn_res_block_dim', default=256, type=int, help='Residual block input dimension')
    parser.add_argument('--cnn_num_res_blocks', default=2, type=int, help='Number of residual blocks')
    parser.add_argument('--cnn_proj_dim', default=128, type=int, help='Output dimension of CNN layer')
    parser.add_argument('--cnn_pooling', default='maxpool', type=str, help='Pooling type for CNN')

    parser.add_argument('--fc_dims', default=(64, 32,))
    parser.add_argument('--fc_use_batchnorm', default=False, type=bool)
    parser.add_argument('--fc_dropout', default=0., type=float)

    parser.add_argument('--run_mode', default='train', type=str, help='Mode: train/dev/test')
    parser.add_argument('--model', default='lstm', type=str, help='Model: lstm/cnn/sa')
    parser.add_argument('--prediction', default='target', type=str, help='Prediction type: source/target')
    parser.add_argument('--checkpoint', default=None, type=int, help='Epoch checkpoint for model loading')
    parser.add_argument('--logdir', default='../../log/', help='Log directory')
    parser.add_argument('--modeldir', default='../../saved_baseline_model/', help='Model save directory')
    parser.add_argument('--config', default='../../config/', help='Config directory')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode: ON')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Debug mode: OFF')
    parser.add_argument('--fast', default=False, action='store_true', help='Subset of data for faster training.')
    parser.add_argument('--print_every', default=1, type=int, help='Print training progress at interval of epcohs.')

    parser.add_argument('--dataset', default='synthetic', type=str, help='Template type.')
    parser.add_argument('--data_dir', default='../../data/', type=str, help='Parent directory path for stored scenes.')
    parser.add_argument('--params_dir', default='../params/', type=str, help='Parent directory to save model weights.')
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
    train_loader, dev_loader, test_loader, vocab_size = get_data(args)

    #######################################
    # Model
    #######################################
    # LSTM only baseline model
    if args.model == 'lstm':
        lstm_kwargs = {
            'vocab_size': vocab_size,
            'pad_idx': pad_idx,

            'rnn_emb_dim': args.rnn_emb_dim,
            'rnn_dim': args.rnn_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,

            'fc_dims': args.fc_dims,
            'fc_use_batchnorm': args.fc_use_batchnorm,
            'fc_dropout': args.fc_dropout,
        }
        model = BiLSTMModel(**lstm_kwargs)

    # LSTM + CNN baseline model
    elif args.model == 'cnn':
        cnnlstm_kwargs = {
            'vocab_size': vocab_size,
            'pad_idx': pad_idx,

            'rnn_emb_dim': args.rnn_emb_dim,
            'rnn_dim': args.rnn_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,

            'cnn_feat_dim': args.cnn_feat_dim,
            'cnn_res_block_dim': args.cnn_res_block_dim,
            'cnn_num_res_blocks': args.cnn_num_res_blocks,
            'cnn_proj_dim': args.cnn_proj_dim,
            'cnn_pooling': args.cnn_pooling,

            'fc_dims': args.fc_dims,
            'fc_use_batchnorm': args.fc_use_batchnorm,
            'fc_dropout': args.fc_dropout,
        }
        model = CNNBiLSTMModel(**cnnlstm_kwargs)

    # LSTM + CNN + Stacked Attention (SA) baseline model
    elif args.model == 'sa':
        sacnnlstm_kwargs = {
            'vocab_size': vocab_size,
            'pad_idx': pad_idx,

            'rnn_emb_dim': args.rnn_emb_dim,
            'rnn_dim': args.rnn_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,

            # 'cnn_feat_dim': args.cnn_feat_dim,

            'fc_dims': args.fc_dims,
            'fc_use_batchnorm': args.fc_use_batchnorm,
            'fc_dropout': args.fc_dropout
        }

        model = SACNNBiLSTMModel(**sacnnlstm_kwargs)

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
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
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
    best_model, best_optimiser, best_val_loss = None, None, None

    for step in range(1, total_val_step + 1):
        step_start_time = time.time()
        train(model, optimiser, criterion, train_loader,
              step=step,
              num_epochs=num_epochs,
              total_epochs=total_val_step * num_epochs,
              print_every=args.print_every,
              tolerance=args.tol)
        val_loss, val_acc, _ = evaluate(model, criterion, dev_loader,
                                        end_of_epoch=step * num_epochs,
                                        start_time=step_start_time,
                                        tolerance=args.tol)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model = model
            best_optimiser = optimiser

    #######################################
    # Save best model and optimiser
    #######################################
    save_at = checkpoint + total_val_step * num_epochs if checkpoint else total_val_step * num_epochs
    save_model(best_model, best_optimiser, best_val_acc, best_val_loss, name, model_dir, save_at)
    logger.info('-+' * 50)
    logger.info(' [[ Best Model ]] Val Acc:  {:.2f} | Val Loss:  {:.6f}'.format(
        best_val_acc * 100, best_val_loss)
    )
    logger.info('-+' * 50)

    #######################################
    # Test on best model
    #######################################
    logger.info('**' * 10)
    logger.info('  TEST  ')
    logger.info('**' * 10)
    test_loss, test_acc, test_mse, _ = test(best_model, criterion, test_loader, tolerance=args.tol)
    logger.info('-+' * 50)
    logger.info(
        ' [[ Best Model ]] Test Acc:  {:.2f} | Test MSE: {:.6f}'.format(test_acc * 100, test_mse)
    )
    logger.info('-+' * 50)
