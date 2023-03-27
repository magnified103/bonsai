import torch
import torch.nn as nn

from network_generator.models.FFNN.layers import BatchNormDropoutLayer, BatchNormLinearLayer, DropoutLinearLayer, \
    LinearLayer, LastLayer


def create_network(network, activation, input_size, output_size, hidden_size, init, dropout_layers, dropout_rate,
                   batch_norm, method):
    if network == 'linear':
        return LinearNetwork(input_size, len(hidden_size), hidden_size, output_size, activation, init, dropout_layers,
                             dropout_rate, batch_norm)
    elif network == 'linear_offset':
        return LinearOffsetNetwork(input_size, len(hidden_size), hidden_size, output_size, activation, init,
                                   dropout_layers, dropout_rate, batch_norm)
    elif network == 'residual':
        return ResidualNetwork(input_size, hidden_size, output_size, activation, method, init)
    elif network == 'residual_offset':
        return ResidualOffsetNetwork(input_size, hidden_size, output_size, activation, method, init)
    else:
        raise ValueError('Unknown network type {}'.format(network))


def get_next_dropout(dropout_layers, dropout_rate, current_layer):
    if dropout_layers is not None:
        if len(dropout_layers) > 0 and dropout_layers[0] == current_layer:
            dropout_layers.pop(0)
            # check if dropout rate is a list
            if isinstance(dropout_rate, list):
                next_dropout_rate = dropout_rate.pop(0)
            else:
                next_dropout_rate = dropout_rate
            return True, next_dropout_rate
    return False, dropout_rate


def get_next_batch_norm(batch_norm, current_layer):
    if batch_norm is not None:
        if len(batch_norm) > 0 and batch_norm[0] == current_layer:
            batch_norm.pop(0)
            return True
    return False


def create_layer(input_size, hidden_size, activation, batchnorm, dropout, dropout_rate, init):
    if batchnorm and dropout:
        return BatchNormDropoutLayer(input_size, hidden_size, activation, dropout_rate, init)
    elif batchnorm:
        return BatchNormLinearLayer(input_size, hidden_size, activation, init)
    elif dropout:
        return DropoutLinearLayer(input_size, hidden_size, activation, dropout_rate, init)
    else:
        return LinearLayer(input_size, hidden_size, activation, init)


class LinearNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation, init=None, dropout_layers=None,
                 dropout_rate=0.5, batch_norm=None, add_last_layer=True):
        super(LinearNetwork, self).__init__()
        layers = []
        next_dropout, next_dropout_rate = get_next_dropout(dropout_layers, dropout_rate, 0)
        next_batch_norm = get_next_batch_norm(batch_norm, 0)
        if hidden_layers > 1:
            next_dropout, next_dropout_rate = get_next_dropout(dropout_layers, dropout_rate, 0)
            next_batch_norm = get_next_batch_norm(batch_norm, 0)
            layer = create_layer(input_size, hidden_size[0], activation, next_batch_norm, next_dropout,
                                 next_dropout_rate, init)
            layers.append(layer)
            for i in range(1, hidden_layers):
                next_dropout, next_dropout_rate = get_next_dropout(dropout_layers, dropout_rate, i)
                next_batch_norm = get_next_batch_norm(batch_norm, i)
                layer = create_layer(hidden_size[i-1], hidden_size[i], activation, next_batch_norm, next_dropout,
                                     next_dropout_rate, init)
                layers.append(layer)
            if add_last_layer:
                layer = nn.Linear(hidden_size[-1], output_size)
            else:
                layer = create_layer(hidden_size[hidden_layers - 1], output_size, activation, next_batch_norm, next_dropout,
                                 next_dropout_rate, init)
            layers.append(layer)
        else:
            layer = create_layer(input_size, output_size, activation, next_batch_norm, next_dropout, next_dropout_rate,
                                 init)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def set_zero(self, zero):
        pass


class LinearOffsetNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation, init=None, dropout_layers=None,
                 dropout_rate=0.5, batch_norm=None):
        super(LinearOffsetNetwork, self).__init__()
        if hidden_layers > 1:
            self.layers = LinearNetwork(input_size, hidden_layers - 1, hidden_size[:-1], hidden_size[-1], activation,
                                        init, dropout_layers, dropout_rate, batch_norm, add_last_layer=False)
            self.last_layer = LastLayer(hidden_size[-1], output_size, init=init)
        else:
            self.layers = None
            self.last_layer = LastLayer(input_size, output_size, init=init)

    def forward(self, x):
        if self.layers is not None:
            x = self.layers(x)
        x = self.last_layer(x)
        return x

    def set_zero(self, zero):
        self.last_layer.set_zero(zero)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, activation, method, init=None, dropout_layers=None,
                 dropout_rate=0.5, batch_norm=None):
        super(ResidualBlock, self).__init__()
        input_size = hidden_size[0]
        output_size = hidden_size[-1]
        if method not in ['add', 'concat']:
            raise ValueError('Unknown residual method {}'.format(method))
        if method == 'add' and input_size != output_size:
            raise ValueError('Input and output size must be equal for add method')

        self.network = LinearNetwork(input_size, len(hidden_size), hidden_size, output_size, activation, init,
                                     dropout_layers,
                                     dropout_rate, batch_norm)
        self.method = method
        if method == 'add':
            self.residual = nn.Linear(input_size, output_size)
        elif method == 'concat':
            self.residual = nn.Linear(output_size + input_size, output_size)

    def forward(self, x):
        if self.method == 'add':
            return self.residual(x + self.network(x))
        elif self.method == 'concat':
            return self.residual(torch.cat((x, self.network(x)), dim=1))
        else:
            raise ValueError('Unknown residual method {}'.format(self.method))


class ResidualNetwork(nn.Module):
    def __init__(self, input_size, residual_blocks, output_size, activation, method, init=None):
        super(ResidualNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, residual_blocks[0]['hidden_size'][0]))
        for residual_block in residual_blocks:
            hidden_size = residual_block['hidden_size']
            if 'dropout_layers' in residual_block:
                dropout_layers = residual_block['dropout_layers']
                dropout_rate = residual_block['dropout_rate']
            else:
                dropout_layers = None
                dropout_rate = 0.5
            if 'batch_norm' in residual_block:
                batch_norm = residual_block['batch_norm']
            else:
                batch_norm = None

            layer = ResidualBlock(hidden_size, activation, method, init, dropout_layers,
                                  dropout_rate, batch_norm)
            layers.append(layer)
        layers.append(nn.Linear(residual_blocks[-1]['hidden_size'][-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def set_zero(self, zero):
        pass


class ResidualOffsetNetwork(nn.Module):
    def __init__(self, input_size, residual_blocks, output_size, activation, method, init=None):
        super(ResidualOffsetNetwork, self).__init__()
        self.residual_network = ResidualNetwork(input_size, residual_blocks, output_size, activation, method, init)
        self.last_layer = LastLayer(output_size, output_size, init)

    def forward(self, x):
        x = self.residual_network(x)
        x = self.last_layer(x)
        return x

    def set_zero(self, zero):
        self.last_layer.set_zero(zero)
