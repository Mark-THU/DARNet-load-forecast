import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

abs_path = os.getcwd()


class GRU_CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,
                 drop_prob):
        super(GRU_CNN, self).__init__()
        # layers
        cnn_1 = nn.Conv1d(input_size, 16, kernel_size=5)
        cnn_1 = nn.Conv1d(16, 16, kernel_size=5, padding=1)
        cnn_2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        cnn_3 = nn.Conv1d(32, 32, kernel_size=3)
        cnn_4 = nn.Conv1d(32, 64, kernel_size=3, stride=2)

        self.cnn = nn.Sequential(cnn_1, nn.ReLU(), cnn_2, nn.ReLU(), cnn_3, nn.ReLU(), cnn_4, nn.ReLU())
        self.dropout = nn.Dropout(drop_prob)
        self.rnn = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob)
        self.dense = nn.Sequential()

        input_size = hidden_dim
        i = 0
        while input_size > 32:
            self.dense.add_module('linear{}'.format(i),
                                  nn.Linear(input_size, round(input_size / 2)))
            self.dense.add_module('relu{}'.format(i), nn.ReLU())
            input_size = round(input_size / 2)
            i += 1
        self.dense.add_module('linear{}'.format(i), nn.Linear(input_size, output_size))

    def forward(self, x):
        # x shape (batch_size, num_steps, input_size)
        mean = x[:, :, -1].mean(dim=-1, keepdim=True)
        x[:, :, -1] /= mean
        x = x.permute(0, 2, 1)
        # x shape (batch_size, input_size, num_steps)
        cnn_out = self.cnn(x)
        # cnn_out shape (batch_size, 64, 15)
        cnn_out = cnn_out.permute(2, 0, 1)
        cnn_out = self.dropout(cnn_out)
        # cnn_out shape (15, batch_size, 64)
        rnn_out, _ = self.rnn(cnn_out)
        # rnn_out shape (15, batch_size, hidden_dim)
        rnn_out = rnn_out[-1, :, :]
        # rnn_out shape (batch_size, hidden_dim)
        out = self.dense(rnn_out) * mean
        # out shape (batch_size, output_size)
        return out


def train_model(train_x, train_y, valid_x, valid_y, input_size, output_size,
                mse_thresh, batch_size, lr, number_epoch, hidden_dim, n_layers,
                drop_prob, weight_decay, device):
    while 1:
        model = GRU_CNN(input_size, output_size, hidden_dim, n_layers,
                        drop_prob)
        model.to(device=device)
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
                                  drop_last=False)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)
        num_without_imp = 0
        train_loss_list = []
        valid_loss_list = []
        # training process
        for epoch in range(1, number_epoch + 1):
            loop = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        leave=True, ncols=100)
            for i, (inputs, labels) in loop:
                inputs = inputs.to(device=device)
                labels = labels[:, :, -1].to(device=device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 5 == 0:
                    num_without_imp = num_without_imp + 1
                    valid_losses = list()
                    model.eval()
                    for inp, lab in valid_loader:
                        inp = inp.to(device)
                        lab = lab[:, :, -1].to(device)
                        out = model(inp)
                        valid_loss = criterion(out, lab)
                        valid_losses.append(valid_loss.item())

                    model.train()
                    loop.set_description("Epoch: {}/{}".format(
                        epoch, number_epoch))
                    loop.set_postfix(train_loss=loss.item(),
                                     valid_loss=np.mean(valid_losses))

                    train_loss_list.append(loss.item())
                    valid_loss_list.append(np.mean(valid_losses))
                    if np.mean(valid_losses) < valid_loss_min:
                        num_without_imp = 0
                        torch.save(model.state_dict(), abs_path + '/models/model/GRU_CNN_state_dict.pt')
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
                             drop_last=False)
    model.load_state_dict(torch.load(abs_path + '/models/model/GRU_CNN_state_dict.pt'))
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = inputs.to(device)
            label = label[:, :, -1].to(device)
            outputs = model(inputs)
            y_pred += outputs.cpu().numpy().flatten().tolist()
            y_true += label.cpu().numpy().flatten().tolist()
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
    return MAPE, SMAPE, MAE,  RRSE, load_pred, load_true
