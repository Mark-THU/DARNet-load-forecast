import numpy as np
import torch
import torch.nn as nn
import os
import math

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

abs_path = os.getcwd()


class CNN1D_RNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=3):
        super(CNN1D_RNN_Block, self).__init__()
        # params
        padding = int(kernel_size / 2)

        # layers
        self.CNN = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size,
                             padding=padding)
        self.RNN1 = nn.GRU(input_size=out_channels,
                           hidden_size=out_channels,
                           num_layers=2,
                           dropout=dropout,
                           batch_first=True)
        self.RNN2 = nn.GRU(input_size=in_channels,
                           hidden_size=out_channels,
                           num_layers=2,
                           dropout=dropout,
                           batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: x shape (batch_size, seq_len, d_feature)
        :return:
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        cnn_input = x.permute(0, 2, 1)
        # cnn_input shape (batch_size, in_channel, seq_len)
        cnn_out = self.CNN(cnn_input)
        # cnn_out = self.bn1(cnn_out)
        cnn_out = self.relu(cnn_out)
        # cnn_out shape (batch_size, out_channel, seq_len)

        rnn_input = cnn_out.permute(0, 2, 1)
        # rnn_input shape (batch_size, seq_len, out_channels)
        rnn_out_1, _ = self.RNN1(rnn_input)
        # rnn_out_1 shape (batch_size, seq_len, out_channels)

        rnn_out_2, _ = self.RNN2(x)
        # rnn_out_2 shape (batch_size, seq_len, out_channels)

        out = self.relu(rnn_out_1 + rnn_out_2)
        return out


class RNN_CNN2D_Block(nn.Module):
    def __init__(self, input_size, out_channels, seq_len, dropout, n_layers=2):
        super(RNN_CNN2D_Block, self).__init__()

        # layers
        self.RNN = nn.GRU(input_size=input_size,
                          hidden_size=out_channels,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.CNN = nn.Conv2d(in_channels=1,
                             out_channels=out_channels,
                             kernel_size=(seq_len, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: x shape (batch_size, seq_len, input_size)
        :return:
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        rnn_out, hidden_state = self.RNN(x)
        # rnn_out shape (batch_size, seq_len, d_features)

        cnn_input = rnn_out.unsqueeze(1)
        # cnn_input shape (batch_size, 1, seq_len, d_featues)
        cnn_out = self.CNN(cnn_input)
        # cnn_out shape (batch_size, out_channels, 1, d_features)
        cnn_out = cnn_out.squeeze(2).permute(0, 2, 1)
        # cnn_out = self.bn2(cnn_out)
        cnn_out = self.relu(cnn_out)

        # cnn_out shape (batch_size, d_features, out_channels)

        return cnn_out, hidden_state


class Encoder(nn.Module):
    def __init__(self, input_size, num_channels, seq_len, dropout=0.5):
        super(Encoder, self).__init__()
        '''
        input_size(int): dimension of features
        num_channels(list): channels of each cnn-rnn layer
        seq_len(int): window length of input
        '''
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [CNN1D_RNN_Block(in_channels, out_channels, dropout)]

        self.cnn_rnn = nn.Sequential(*layers)
        self.rnn_cnn = RNN_CNN2D_Block(input_size, num_channels[-1], seq_len, dropout=0)

    def forward(self, x, rho):
        """
        :param x: x shape (batch_size, seq_len, input_size)
        :param rho:
        :return:
        """
        features = x[:, :, :-1]
        values = x[:, :, -1]
        # features shape (batch_size, num_steps, input_size -1)
        # values shape (batch_size, num_steps)

        values_mean = torch.mean(values, dim=1)
        # values_mean shape (batch_size)
        values = torch.cat((values_mean.unsqueeze(1), values), dim=1)
        # values shape (batch_size, num_steps + 1)
        values = values[:, 1:] - rho * values[:, :-1]
        # values shape (batch_size, num_steps)
        inp = torch.cat((features, values.unsqueeze(2)), dim=2)
        # inp shape (batch_size, num_steps, input_size)

        cnn_rnn_out = self.cnn_rnn(inp)
        # cnn_rnn_out shape (batch_size, seq_len, num_channels[-1])

        rnn_cnn_out, hidden_state = self.rnn_cnn(inp)
        # rnn_cnn_out shape (batch_size, num_channels[-1], num_channels[-1])

        return cnn_rnn_out, rnn_cnn_out, hidden_state


class AdditiveAttention(nn.Module):
    """additive attention"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """dot-product attention"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class RNN_Attn_Block(nn.Module):
    def __init__(self, input_size, hidden_dim, i, dropout=0.5):
        super(RNN_Attn_Block, self).__init__()
        # params
        self.i = i
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # layers
        self.RNN = nn.GRU(input_size=input_size + hidden_dim,
                          hidden_size=hidden_dim,
                          num_layers=2,
                          batch_first=True,
                          dropout=dropout)
        self.attention1 = DotProductAttention(dropout)
        self.attention2 = DotProductAttention(dropout)
        # self.attention1 = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, 2, dropout)
        # self.attention2 = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, 2, dropout)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, state):
        """
        x shape (batch_size, 1, input_size)
        state[0] cnn_rnn_out shape (batch_size, 72, hidden_dim)
        state[1] rnn_cnn_out shape (batch_size, hidden_dim, hidden_dim)
        state[2][i] shape (num_layers, batch_size, hidden_dim)
        """
        cnn_rnn_out = state[0]
        rnn_cnn_out = state[1]

        query = state[2][self.i][-1].unsqueeze(1)
        # query shape (batch_size, 1, hidden_dim)

        context_1 = self.attention1(query, cnn_rnn_out, cnn_rnn_out)
        # context_1 shape (batch_size, 1, hidden_dim)

        rnn_input = torch.cat((x, context_1), dim=-1)
        rnn_out, state[2][self.i] = self.RNN(rnn_input, state[2][self.i])
        # rnn_out shape (batch_size, 1, hidden_dim)
        # state shape (num_layers, batch_size, hidden_dim)

        context_2 = self.attention2(rnn_out, rnn_cnn_out, rnn_cnn_out)
        # context_2 shape (batch_size, 1, hidden_dim)

        out = self.dense(torch.cat((rnn_out, context_2), dim=-1))
        if self.input_size != self.hidden_dim:
            out = self.relu(out + rnn_out)
        # out shape (batch_size, 1, hidden_dim)
        else:
            out = self.relu(out + x)

        return out, state


class Decoder(nn.Module):
    def __init__(self, input_size, num_hidden_dim, dropout):
        super(Decoder, self).__init__()
        layers = []
        dense_layers = []
        num_levels = len(num_hidden_dim)
        for i in range(num_levels):
            in_size = input_size if i == 0 else num_hidden_dim[i - 1]
            out_size = num_hidden_dim[i]
            layers += [RNN_Attn_Block(in_size, out_size, i, dropout=0)]

        input_dim = num_hidden_dim[-1]
        while input_dim > 4:
            dense_layers += [
                nn.Linear(input_dim, round(input_dim / 2)),
                nn.ReLU()
            ]
            input_dim = round(input_dim / 2)

        dense_layers += [nn.Linear(input_dim, 1)]

        self.blks = nn.Sequential(*layers)
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x, state):
        """
        x shape (batch_size, 1, input_size)
        state[0] cnn_rnn_out shape (batch_size, 72, hidden_dim)
        state[1] rnn_cnn_out shape (batch_size, hidden_dim, hidden_dim)
        state[2] shape n * (num_layers, batch_size, hidden_dim)
        """
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)

        out = self.dense(x)
        # out shape (batch_size, 1, 1)
        return out, state


class DARNet(nn.Module):
    def __init__(self,
                 input_size,
                 num_channels,
                 seq_len,
                 num_hidden_dim,
                 dropout,
                 device):
        super(DARNet, self).__init__()
        # params
        self.num_layers = len(num_hidden_dim)
        self.device = device
        # layers
        self.encoder = Encoder(input_size, num_channels, seq_len, dropout)
        self.decoder = Decoder(input_size, num_hidden_dim, dropout)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs shape (batch_size, seq_len, input_size)
        dec_inputs shape (batch_size, tar_len, input_size)
        """
        rho = torch.ones(1, device=self.device)
        y_ = dec_inputs[:, :1, -1].clone()
        # y_ shape (batch_size, 1)
        dec_inputs[:, :1, -1] = dec_inputs[:, :1, -1] - rho * torch.mean(
            dec_inputs[:, :, -1], dim=-1, keepdim=True)

        cnn_rnn_out, rnn_cnn_out, hidden_state = self.encoder(enc_inputs, rho)
        state = [cnn_rnn_out, rnn_cnn_out, [hidden_state] * self.num_layers]

        outputs = []

        for i in range(dec_inputs.shape[1]):
            if i:
                x = torch.cat((dec_inputs[:, i:i + 1, :-1], out.detach()),
                              dim=-1)
            else:
                x = dec_inputs[:, i:i + 1, :]
                # x shape (batch_size, 1, input_size)
            out, state = self.decoder(x, state)
            # out shape (batch_size, 1, 1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        outputs = outputs + rho * y_
        # outputs shape (batch_size, 24)
        return outputs


def train_model(train_x, train_y, valid_x, valid_y, input_size, seq_len,
                mse_thresh, hidden_dim, n_layers, number_epoch,
                batch_size, lr, drop_prob, weight_decay, device):
    while 1:
        model = DARNet(input_size, [hidden_dim]*n_layers, seq_len, [hidden_dim]*n_layers, drop_prob, device)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        # scheduler = SchedulerCosineDecayWarmup(optimizer, lr, 10, number_epoch)
        valid_loss_min = np.Inf
        train_dataset = TensorDataset(torch.FloatTensor(train_x),
                                      torch.FloatTensor(train_y))
        valid_dataset = TensorDataset(torch.FloatTensor(valid_x),
                                      torch.FloatTensor(valid_y))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
        train_losses = list()

        num_without_imp = 0

        train_loss_list = []
        valid_loss_list = []
        #train
        for epoch in range(1, number_epoch + 1):
            loop = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        leave=True,
                        ncols=100)
            for i, (inputs, labels) in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                encoder_inputs = inputs
                decoder_inputs = torch.cat(
                    (inputs[:, -1:, :], labels[:, :-1, :]), dim=1)
                outputs = model(encoder_inputs, decoder_inputs)
                loss = criterion(outputs, labels[:, :, -1])
                train_losses.append(loss.item)
                loss.backward()
                optimizer.step()

                # eval
                if i % 5 == 0:
                    num_without_imp = num_without_imp + 1
                    valid_losses = list()
                    model.eval()
                    for inp, lab in valid_loader:
                        inp = inp.to(device)
                        lab = lab.to(device)
                        encoder_inp = inp
                        decoder_inp = torch.cat(
                            (inp[:, -1:, :], lab[:, :-1, :]), dim=1)
                        out = model(encoder_inp, decoder_inp)
                        valid_loss = criterion(out, lab[:, :, -1])
                        valid_losses.append(valid_loss.item())
                    model.train()
                    loop.set_description("Epoch: {}/{}...".format(
                        epoch, number_epoch))
                    loop.set_postfix(train_loss=loss.item(),
                                     valid_loss=np.mean(valid_losses))
                    train_loss_list.append(loss.item())
                    valid_loss_list.append(np.mean(valid_losses))
                    if np.mean(valid_losses) < valid_loss_min:
                        num_without_imp = 0
                        torch.save(model.state_dict(), abs_path + "/models/model/DARNet_state_dict.pt")
                        valid_loss_min = np.mean(valid_losses)
            scheduler.step()
        if valid_loss_min < mse_thresh:
            break
    return model, train_loss_list, valid_loss_list


def test_model(model, test_x, test_y, scaler_y, batch_size, device):
    test_dataset = TensorDataset(torch.FloatTensor(test_x),
                                 torch.FloatTensor(test_y))
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True)
    model.load_state_dict(torch.load(abs_path + "/models/model/DARNet_state_dict.pt"))
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            encoder_inputs = inputs
            decoder_inputs = torch.cat((inputs[:, -1:, :], labels[:, :-1, :]),
                                       dim=1)
            outputs = model(encoder_inputs, decoder_inputs)
            y_pred += outputs.cpu().numpy().flatten().tolist()
            y_true += labels[:, :, -1].cpu().numpy().flatten().tolist()
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    #     pdb.set_trace()
    load_pred = scaler_y.inverse_transform(y_pred)
    load_true = scaler_y.inverse_transform(y_true)
    mean_pred = np.mean(load_pred)
    mean_true = np.mean(load_true)
    MAPE = np.mean(np.abs(load_true - load_pred) / load_true)
    SMAPE = 2 * np.mean(
        np.abs(load_true - load_pred) / (load_true + load_pred))
    MAE = np.mean(np.abs(load_true - load_pred))
    RRSE = np.sqrt(np.sum(np.square(load_true - load_pred))) / np.sqrt(
        np.sum(np.square(load_true - mean_true)))
    return MAPE, SMAPE, MAE, RRSE, load_pred, load_true
