import pandas as pd
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from models import BiLSTM, GRU_CNN, Seq2seq, DARNet


def random_seed_set(seed, gpu):
    """
    set random seed
    :param seed:
    :param gpu: whether to use gpu
    :return:
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if gpu and torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)
        dev = gpu
    else:
        dev = 'cpu'
    device = torch.device(dev)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def load_data(dataset):
    """
    :param url: data url
    :return:
    """
    url = './data/{}.csv'.format(dataset)
    data = pd.read_csv(url, sep=',', index_col='time')
    return data


def normalization(data):
    """
    :param data: original data with load
    :return: normalized data, scaler of data, scaler of load
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    scaler_y = MinMaxScaler()
    scaler_y.fit_transform(data[[data.columns[-1]]])
    return normalized_data, scaler, scaler_y


def series_to_supervise(data, seq_len, target_len):
    """
    convert series data to supervised data
    :param data: original data
    :param seq_len: length of input sequence
    :param target_len: length of ouput sequence
    :return: return two ndarrays-- input and output in format suitable to feed to RNN
    """
    dim_0 = data.shape[0] - seq_len - target_len + 1
    dim_1 = data.shape[1]
    x = np.zeros((dim_0, seq_len, dim_1))
    y = np.zeros((dim_0, target_len, dim_1))
    for i in range(dim_0):
        x[i] = data[i:i + seq_len]
        y[i] = data[i + seq_len:i + seq_len + target_len]
    print("supervised data: shape of x: {}, shape of y: {}".format(
        x.shape, y.shape))
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch load forecasting')
    parser.add_argument('--dataset', type=str, default='D1', help='location of the data file')
    parser.add_argument('--model', type=str, default='DARNet', help='')
    parser.add_argument('--seq_len', type=int, default=72, help='length of conditioning range')
    parser.add_argument('--target_len', type=int, default=24, help='length of prediction range')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=80, help='')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--drop_prob', type=float, default=0, help='dropout probability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--mse_thresh', type=float, default=0.01, help='mse thresh decide whether to stop training')
    parser.add_argument('--gpu', type=str, default=None, help='whether to use gpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    data = load_data(args.dataset)
    train_data = data[:int(0.8 * len(data))]
    train_data, scaler, scaler_y = normalization(train_data)
    train_x, train_y = series_to_supervise(train_data, args.seq_len, args.target_len)
    valid_x = train_x[int(0.8 * len(train_x)):]
    valid_y = train_y[int(0.8 * len(train_y)):]
    train_x = train_x[:int(0.8 * len(train_x))]
    train_y = train_y[:int(0.8 * len(train_y))]
    input_size = train_x.shape[2]
    output_size = args.target_len
    test_data = data[int(0.8 * len(data)):]
    test_data = scaler.transform(test_data)
    test_x, test_y = series_to_supervise(test_data, args.seq_len, args.target_len)
    device = random_seed_set(args.seed, args.gpu)

    if args.model == 'BiLSTM':
        model, train_loss_list, valid_loss_list = BiLSTM.train_model(
            train_x, train_y, valid_x, valid_y, input_size, output_size,
            args.mse_thresh, args.batch_size, args.lr, args.num_epoch,
            args.hidden_dim, args.n_layers, args.drop_prob, args.weight_decay,
            device)
        # plot training loss and validation loss
        plt.plot(train_loss_list[10:], 'm', label='train_loss')
        plt.plot(valid_loss_list[10:], 'g', label='valid_loss')
        plt.grid('both')
        plt.title('{}'.format(args.model))
        plt.legend()
        plt.savefig('./figs/{}.png'.format(args.model), dpi=300, format='png')
        plt.show()
        # test
        MAPE, SMAPE, MAE, RRSE, load_pred, load_true = BiLSTM.test_model(
            model, test_x, test_y, scaler_y, args.batch_size, device)
        # print evaluation metrics
        print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RRSE:{:.6f}'.format(MAPE, SMAPE, MAE, RRSE))
        np.save('./results/{}-{}-{}-{}-TRUE.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_true)
        np.save('./results/{}-{}-{}-{}-PRED.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_pred)
    elif args.model == 'GRU_CNN':
        model, train_loss_list, valid_loss_list = GRU_CNN.train_model(
            train_x, train_y, valid_x, valid_y, input_size, output_size,
            args.mse_thresh, args.batch_size, args.lr, args.num_epoch,
            args.hidden_dim, args.n_layers, args.drop_prob, args.weight_decay,
            device)
        # plot training loss and validation loss
        plt.plot(train_loss_list[10:], 'm', label='train_loss')
        plt.plot(valid_loss_list[10:], 'g', label='valid_loss')
        plt.grid('both')
        plt.title('{}'.format(args.model))
        plt.legend()
        plt.savefig('./figs/{}.png'.format(args.model), dpi=300, format='png')
        plt.show()
        # test
        MAPE, SMAPE, MAE, RRSE, load_pred, load_true = GRU_CNN.test_model(
            model, test_x, test_y, scaler_y, args.batch_size, device)
        # print evaluation metrics
        print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RRSE:{:.6f}'.format(MAPE, SMAPE, MAE, RRSE))
        np.save('./results/{}-{}-{}-{}-TRUE.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_true)
        np.save('./results/{}-{}-{}-{}-PRED.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_pred)
    elif args.model == 'Seq2Seq':
        model, train_loss_list, valid_loss_list = Seq2seq.train_model(
            train_x, train_y, valid_x, valid_y, input_size, args.mse_thresh, args.hidden_dim,
            args.n_layers, args.num_epoch, args.batch_size, args.lr, args.drop_prob,
            args.weight_decay, device)
        # plot training loss and validation loss
        plt.plot(train_loss_list[10:], 'm', label='train_loss')
        plt.plot(valid_loss_list[10:], 'g', label='valid_loss')
        plt.grid('both')
        plt.title('{}'.format(args.model))
        plt.legend()
        plt.savefig('./figs/{}.png'.format(args.model), dpi=300, format='png')
        plt.show()
        # test
        MAPE, SMAPE, MAE, RRSE, load_pred, load_true = Seq2seq.test_model(
            model, test_x, test_y, scaler_y, args.batch_size, device)
        # print evaluation metrics
        print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RRSE:{:.6f}'.format(MAPE, SMAPE, MAE, RRSE))
        np.save('./results/{}-{}-{}-{}-TRUE.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_true)
        np.save('./results/{}-{}-{}-{}-PRED.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_pred)
    elif args.model == 'DARNet':
        model, train_loss_list, valid_loss_list = DARNet.train_model(
            train_x, train_y, valid_x, valid_y, input_size, args.seq_len,
            args.mse_thresh, args.hidden_dim, args.n_layers, args.num_epoch,
            args.batch_size, args.lr, args.drop_prob, args.weight_decay, device)
        # plot training loss and validation loss
        plt.plot(train_loss_list[10:], 'm', label='train_loss')
        plt.plot(valid_loss_list[10:], 'g', label='valid_loss')
        plt.grid('both')
        plt.title('{}'.format(args.model))
        plt.legend()
        plt.savefig('./figs/{}.png'.format(args.model), dpi=300, format='png')
        plt.show()
        # test
        MAPE, SMAPE, MAE, RRSE, load_pred, load_true = DARNet.test_model(
            model, test_x, test_y, scaler_y, args.batch_size, device)
        # print evaluation metrics
        print('MAPE:{:.6f},SMAPE:{:.6f},MAE:{:.6f},RRSE:{:.6f}'.format(MAPE, SMAPE, MAE, RRSE))
        np.save('./results/{}-{}-{}-{}-TRUE.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_true)
        np.save('./results/{}-{}-{}-{}-PRED.npy'.format(args.model, args.dataset,
                                                        args.seq_len, args.target_len), load_pred)
    else:
        print('please choose a model from BiLSTM, GRU_CNN, Seq2Seq, DARNet')


