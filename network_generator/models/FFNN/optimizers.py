import torch


def choose_optimizer(optimizer, model, learning_rate):
    if optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise Exception(f'Optimizer {optimizer} not supported!')


def choose_scheduler(optimizer, scheduler, scheduler_kwargs):
    if scheduler is None:
        return None

    if scheduler == 'MultiStep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)
    elif scheduler == 'CosineAnnealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_kwargs)
    elif scheduler == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
    elif scheduler == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_kwargs)
    elif scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    elif scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    else:
        raise Exception(f'Scheduler {scheduler} not supported!')


