import torch.nn
from torchmetrics.functional import mean_absolute_percentage_error, mean_squared_error


def choose_loss_fn(loss, param=None):
    if loss == 'MSE':
        return torch.nn.MSELoss()
    elif loss == 'MAPE':
        return MAPE_Loss()
    elif loss == 'MSE_MAPE':
        return MSE_MAPE_Loss(param)
    elif loss == 'SmoothL1':
        return torch.nn.SmoothL1Loss(beta=param)
    elif loss == 'L1':
        return torch.nn.L1Loss()
    elif loss == 'Huber':
        return torch.nn.HuberLoss(delta=param)
    else:
        raise Exception(f'Loss {loss} not supported!')


class MAPE_Loss(torch.nn.Module):
    def __init__(self):
        super(MAPE_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        return mean_absolute_percentage_error(y_pred, y_true)


class MSE_MAPE_Loss(torch.nn.Module):
    def __init__(self, lamb):
        super(MSE_MAPE_Loss, self).__init__()
        self.lamb = lamb
        self.mse = torch.nn.MSELoss()
        self.mape = MAPE_Loss()

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        mape = self.mape(y_pred, y_true)
        return self.lamb * mse + (1 - self.lamb) * mape


class NegativesPenalty(torch.nn.Module):
    def __init__(self, penalty):
        super(NegativesPenalty, self).__init__()
        self.zero = 0
        self.penalty = penalty

    def forward(self, y_pred, y_true):
        negatives = torch.sum(y_pred < self.zero)
        return self.penalty * negatives

    def set_zero(self, zero):
        self.zero = zero

