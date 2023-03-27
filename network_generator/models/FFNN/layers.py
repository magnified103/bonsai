import torch
import torch.nn as nn


def apply_init(layer, init, param=None):
    if type(layer) != nn.Linear:
        return

    if init == 'xavier_uniform':
        if param == 'relu':
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        elif param == 'leaky_relu':
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        else:
            torch.nn.init.xavier_uniform_(layer.weight)
    elif init == 'xavier_normal':
        if param == 'relu':
            torch.nn.init.xavier_normal_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        elif param == 'leaky_relu':
            torch.nn.init.xavier_normal_(layer.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        else:
            torch.nn.init.xavier_normal_(layer.weight)
    elif init == 'kaiming_uniform':
        if param == 'relu':
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif param == 'leaky_relu':
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in')
    elif init == 'kaiming_normal':
        if param == 'relu':
            torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif param == 'leaky_relu':
            torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
    else:
        raise ValueError('Unknown initialization method {}'.format(init))


class NNLayer(nn.Module):
    def __init__(self):
        super(NNLayer, self).__init__()
        self.layer = None

    def forward(self, x):
        x = self.layer(x)
        return x


class LinearLayer(NNLayer):
    def __init__(self, input_size, output_size, activation, init=None):
        super(LinearLayer, self).__init__()
        if activation == 'relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()
            )
        elif activation == 'leaky_relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LeakyReLU()
            )
        else:
            raise ValueError('Unknown activation function {}'.format(activation))
        if init is not None:
            self.layer.apply(lambda x: apply_init(x, init, activation))


class LastNNLayer(nn.Module):
    def __init__(self):
        super(LastNNLayer, self).__init__()
        self.zero = 0
        self.layer = None

    def forward(self, x):
        x = self.layer(x)
        x = torch.nn.functional.threshold(x, self.zero, self.zero)
        return x

    def set_zero(self, zero):
        self.zero = zero


class LastLayer(LastNNLayer):
    def __init__(self, input_size, output_size, init=None):
        super(LastLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        if init is not None:
            apply_init(self.layer, init)


class BatchNormLinearLayer(NNLayer):
    def __init__(self, input_size, output_size, activation, init=None):
        super(BatchNormLinearLayer, self).__init__()
        if activation == 'relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU()
            )
        elif activation == 'leaky_relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.LeakyReLU()
            )
        else:
            raise ValueError('Unknown activation function {}'.format(activation))
        if init is not None:
            self.layer.apply(lambda x: apply_init(x, init, activation))



class DropoutLinearLayer(NNLayer):
    def __init__(self, input_size, output_size, activation, dropout=0.5, init=None):
        super(DropoutLinearLayer, self).__init__()
        if activation == 'relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Dropout(dropout),
                nn.ReLU()
            )
        elif activation == 'leaky_relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Dropout(dropout),
                nn.LeakyReLU()
            )
        else:
            raise ValueError('Unknown activation function {}'.format(activation))
        if init is not None:
            self.layer.apply(lambda x: apply_init(x, init, activation))


class BatchNormDropoutLayer(NNLayer):
    def __init__(self, input_size, output_size, activation, dropout=0.5, init=None):
        super(BatchNormDropoutLayer, self).__init__()
        if activation == 'relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.Dropout(dropout),
                nn.ReLU()
            )
        elif activation == 'leaky_relu':
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.Dropout(dropout),
                nn.LeakyReLU()
            )
        else:
            raise ValueError('Unknown activation function {}'.format(activation))
        if init is not None:
            self.layer.apply(lambda x: apply_init(x, init, activation))
