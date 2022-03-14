import numpy as np
import torch
import torch.nn as nn
import os
import math
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

abs_path = os.getcwd()


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, drop_prob):
        super(Encoder, self).__init__()
        # init
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # layer
        self.rnn = nn.GRU(input_size,
                          hidden_dim,
                          n_layers,
                          bidirectional=False,
                          dropout=drop_prob)

    def forward(self, x):
        # x shape：(`batch_size`, `num_steps`, `input_size`)
        x = torch.transpose(x, 0, 1)
        rnn_out, state = self.rnn(x)
        return rnn_out, state


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, drop_prob):
        super(AttnDecoder, self).__init__()

        self.attention = DotProductAttention(drop_prob)
        self.rnn = nn.GRU(input_size + hidden_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.fc = nn.Sequential()

        input_size = hidden_dim
        i = 0
        while (input_size > 8):
            self.fc.add_module('linear{}'.format(i),
                               nn.Linear(input_size, round(input_size / 2)))
            self.fc.add_module('relu{}'.format(i), nn.ReLU())
            input_size = round(input_size / 2)
            i += 1
        self.fc.add_module('linear{}'.format(i), nn.Linear(input_size, 1))

    def forward(self, inputs, encoder_outputs, encoder_state):
        """
        :inputs shape (batch_size, target_len, input_size)
        :encoder_outputs shape (seq_len, batch_size, hidden_dim)
        :encoder_state shape (n_layers, batch_size, hidden_dim)
        """
        # inputs shape（target_len, batch_size, input_size）
        inputs = torch.transpose(inputs, 0, 1)
        # encoder_outputs shape (batch_size, seq_len, hidden_dim)
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)
        # decoder_state init
        decoder_state = encoder_state
        outputs = []

        for i, x in enumerate(inputs):
            # query shape is (batch_size, 1, hidden_dim)
            query = torch.unsqueeze(decoder_state[-1], dim=1)
            context = self.attention(query, encoder_outputs, encoder_outputs)
            # training process is different from eval process
            # if i and not self.training:
            if i:
                x[:, -1] = out.detach().flatten()
            # x shape is (batch_size, 1, hidden_dim + input_size)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # x reshape to (1, batch_size, hidden_dim + input_size)
            x = torch.transpose(x, 0, 1)
            # out shape (1, batch_size, hidden_dim)
            out, decoder_state = self.rnn(x, decoder_state)
            # out shape (batch_size, 1)
            out = self.fc(out.squeeze(dim=0))
            outputs.append(out)
        # outputs shape (batch_size, target_len)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Seq2Seq_Attn(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, drop_prob):
        super(Seq2Seq_Attn, self).__init__()
        self.encoder = Encoder(input_size, hidden_dim, n_layers, drop_prob)
        self.decoder = AttnDecoder(input_size, hidden_dim, n_layers, drop_prob)

    def forward(self, encoder_inputs, decoder_inputs):
        # encoder_inputs shape (batch_size, seq_len, input_size)
        # decoder_inputs shape (batch_size, target_len, input_size)
        encoder_outputs, encoder_state = self.encoder(encoder_inputs)
        outputs = self.decoder(decoder_inputs, encoder_outputs, encoder_state)
        return outputs


def train_model(train_x, train_y, valid_x, valid_y, input_size, mse_thresh, hidden_dim,
                n_layers, number_epoch, batch_size, lr, drop_prob,
                weight_decay, device):
    while 1:
        model = Seq2Seq_Attn(input_size, hidden_dim, n_layers, drop_prob)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
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
        # train
        for epoch in range(1, number_epoch + 1):
            loop = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        leave=True, ncols=100)
            for i, (inputs, labels) in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                encoder_inputs = inputs
                decoder_inputs = torch.cat((inputs[:, -1:, :], labels[:, :-1, :]), dim=1)
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
                        decoder_inp = torch.cat((inp[:, -1:, :], lab[:, :-1, :]), dim=1)
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
                        torch.save(model.state_dict(), abs_path + "/models/model/seq2seq_DPA_state_dict.pt")
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
    model.load_state_dict(torch.load(abs_path + "/models/model/seq2seq_DPA_state_dict.pt"))
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
