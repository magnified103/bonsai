import os

import scipy.sparse
import time

import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import Dataset

from network_generator.models.FFNN.loss import choose_loss_fn, NegativesPenalty
from network_generator.models.FFNN.networks import create_network
from network_generator.models.FFNN.optimizers import choose_optimizer, choose_scheduler

torch.set_default_dtype(torch.float64)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchDataframe(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame or pd.Series):
            X = X.to_numpy()
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            X = X.toarray()
        if isinstance(y, pd.DataFrame or pd.Series):
            y = y.to_numpy()
        self.X = X
        self.y = y.reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(get_device()), y.to(get_device())

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    total_loss /= len(dataloader)
    # print(f"Avg loss: {total_loss:>8f}")
    return total_loss


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(get_device()), y.to(get_device())
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Avg loss: {test_loss:>8f}")
    return test_loss


class LossFN(nn.Module):
    def __init__(self, lossfn, penalty=0):
        super(LossFN, self).__init__()
        self.lossfn = lossfn
        self.penalty = NegativesPenalty(penalty)

    def forward(self, pred, y):
        loss = self.lossfn(pred, y)
        loss += self.penalty(pred, y)
        return loss

    def set_zero(self, zero):
        self.penalty.set_zero(zero)

class FFNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, network, activation, hidden_size, output_size=1, learning_rate=1e-3, batch_size=64, epochs=100,
                 scaler=None, loss='MSE_MAPE', loss_param=0.5, loss_penalty=0, optimizer='adam', scheduler=None, scheduler_kwargs={},
                 init=None, dropout_layers=None, dropout_rate=0.5, batch_norm=None, method=None):

        super(FFNNRegressor, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = scaler

        self.model = create_network(network, activation, input_size, output_size, hidden_size, init, dropout_layers,
                                    dropout_rate, batch_norm, method)

        if get_device().type == 'cuda':
            torch.cuda.synchronize()
            #print('Using GPU')
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                #print('Using GPU in parallel mode')
        else:
            #print('Using CPU')
            pass

        self.model = self.model.to(get_device())
        self.loss_fn = LossFN(choose_loss_fn(loss, loss_param), loss_penalty)
        self.optimizer = choose_optimizer(optimizer, self.model, learning_rate)
        self.scheduler = choose_scheduler(self.optimizer, scheduler, scheduler_kwargs)

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.to_numpy().reshape(-1, 1)
        if self.scaler is not None:
            y = self.scaler.fit_transform(y)
            self.loss_fn.set_zero(self.zero())
            if isinstance(self.model, nn.DataParallel):
                self.model.module.set_zero(self.zero())
            else:
                self.model.set_zero(self.zero())
            if get_device().type == 'cuda':
                torch.cuda.synchronize()

        if self.batch_size == -1:
            self.batch_size = len(X)
        trainset = PyTorchDataframe(X, y)
        train_dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.batch_size)
        for t in range(self.epochs):
            start = time.time()
            # print(f"Epoch {t + 1}\n-------------------------------")
            loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.scheduler)
            self.last_loss = loss

        return self

    def predict(self, X):
        preds = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(PyTorchDataframe(X, np.zeros(X.shape[0])),
                                                     batch_size=self.batch_size)
            for X, _ in dataloader:
                X = X.to(get_device())
                pred = self.model(X).detach().cpu().numpy()
                if self.scaler is not None:
                    pred = self.scaler.inverse_transform(pred)
                preds.extend(pred)
        return np.concatenate(preds)

    def score(self, X, y):
        if self.scaler is not None:
            y = self.scaler.transform(y)
        valdata = PyTorchDataframe(X, y)
        valdataloader = torch.utils.data.DataLoader(dataset=valdata, batch_size=self.batch_size)
        return test(valdataloader, self.model, self.loss_fn), self.last_loss

    def zero(self):
        if self.scaler is None:
            return 0
        return self.scaler.transform(np.zeros((1, 1)))[0][0]