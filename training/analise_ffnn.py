import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.analyse_results import plot_predictions, plot_cdf, plot_loss_vs_training_loss, set_out_dir, plot_error
from network_generator import models
from training.loader import load_all_data

train, test = load_all_data('',
                            data_file='/Users/akos/Documents/Projects/bonsai/training/data/small_sample_data.csv.gz',
                            sample_size=1000)
X = train[0]
y = train[1]
X_test = test[0]
y_test = test[1]

epochs = 20

out_dir = '/Users/akos/Documents/Projects/bonsai/training/results/ffnn'
os.makedirs(out_dir, exist_ok=True)

set_out_dir(out_dir)

mapes = []
losses = []
last_results = []
all_results = []

constant_config = {'input_size': X.shape[1], 'output_size': y.shape[1],
                   'batch_size': 512,
                   'transform': 'power',
                   'init': 'kaiming_uniform',
                   'epochs': epochs,
                   'scheduler': 'CosineAnnealing',
                   'scheduler_kwargs': {'T_max': epochs * 5},
                   'activation': 'relu',
                   #'loss': 'MSE',
                   'loss_penalty': 0.3,
                   'learning_rate': 0.001,
                   'optimizer': 'adam',
                   'dropout_rate': 0.2,
                   }

hidden_size = [256, 64, 256, 64, 254, 64, 254, 64, 256, 64, 256, 64, 32]
dropout_layers = [2, 4, 6, 8, 10]
batch_norm = [i for i in range(0, len(hidden_size))]
res_hidden_size = [
    {'hidden_size': [256 for _ in range(3)], 'dropout_layers': [0, 1, 2], 'dropout_rate': 0.5, 'batch_norm': [2]},
    {'hidden_size': [256 for _ in range(3)], 'dropout_layers': [0, 1, 2], 'dropout_rate': 0.5, 'batch_norm': [2]},
    {'hidden_size': [256 for _ in range(3)], 'dropout_layers': [0, 1, 2], 'dropout_rate': 0.5, 'batch_norm': [2]},
]


configs = [
    #{'network': 'linear_offset', 'hidden_size': hidden_size, 'dropout_layers': dropout_layers, 'batch_norm': batch_norm},
    {'network': 'linear', 'hidden_size': hidden_size, 'dropout_layers': dropout_layers, 'dropout_rate': 0.2, 'batch_norm': batch_norm},
    # {'network': 'residual_offset', 'hidden_size': res_hidden_size, 'method': 'add'},
    # {'network': 'residual', 'hidden_size': res_hidden_size, 'method': 'add'},
    # {'network': 'residual_offset', 'hidden_size': res_hidden_size, 'method': 'concat'},
    # {'network': 'residual', 'hidden_size': res_hidden_size, 'method': 'concat'},
]

hidden_size = [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 128, 64, 32]
dropout_layers = [2, 4, 6, 8, 10]
batch_norm = [i for i in range(0, len(hidden_size))]
configs.append({'network': 'linear', 'hidden_size': hidden_size, 'dropout_layers': dropout_layers, 'dropout_rate': 0.2, 'batch_norm': batch_norm},)

hidden_size = [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 128, 64, 32]
dropout_layers = [2, 4, 6, 8, 10, 12, 14]
batch_norm = [i for i in range(0, len(hidden_size))]
configs.append({'network': 'linear', 'hidden_size': hidden_size, 'dropout_layers': dropout_layers, 'dropout_rate': 0.2, 'batch_norm': batch_norm},)



all_configs = []
for loss in ['MSE_MAPE']:
    for config in configs:
        _config = config.copy()
        _config['loss'] = loss
        if loss == 'Huber' or loss == 'SmoothL1':
            _config['loss_param'] = 1.0
        elif loss == 'MSE_MAPE':
            _config['loss_param'] = 1
        all_configs.append(_config)

for config in all_configs:
    print(f'------- Testing {config["network"]} -------')
    config = {**constant_config, **config}
    print(config)

    model = models.SimpleFFNNModel(config)

    scores = []
    for i in range(epochs):
        model.fit(X, y)
        training_loss, loss, r2, rmse, mae, mape = model.score(X_test, y_test)
        scores.append(
            {'epoch': i, 'loss': loss, 'training loss': training_loss, 'r2': r2, 'rmse': rmse, 'mae': mae,
             'mape': mape})
        print(
            f'epoch: {i}, loss: {loss}, training_loss: {training_loss}, r2: {r2}, rmse: {rmse}, mae: {mae}, mape: {mape}')
        # if loss > 1:
        #     print(' Stopping early due to high loss')
        #     break

    y_pred = model.predict(X_test)

    # set y_pred to be 1 dimensional
    y_pred_ = y_pred.reshape(-1)
    # set y_test to be 1 dimensional
    y_test_ = y_test.reshape(-1)

    negative_values = np.sum(y_pred_ < 0)
    if negative_values > 0:
        print(' ---------- WARNING: negative values in prediction ----------')
        print(f'---------- negative values: {negative_values} ----------')
        for i in range(len(y_pred_)):
            if y_pred_[i] < 0:
                print(f'---------- negative value: {y_pred_[i]} at index {i} ----------')

    results = {'prediction_results': pd.DataFrame({'actual': y_test_, 'pred': y_pred_}),
               'scores': pd.DataFrame(scores),
               'name': f"{config['network']}_reg_{config['method'] if 'method' in config else ''}_{config['loss']}_{config['loss_penalty']}"}

    plot_loss_vs_training_loss(results)
    plot_cdf(results)
    plot_error(results)
    plot_predictions(results)
    last_results.append({'name': results['name'], 'mape': results['scores']['mape'].iloc[-1],
                         'loss': results['scores']['loss'].iloc[-1],
                         'training loss': results['scores']['training loss'].iloc[-1],
                         'r2': results['scores']['r2'].iloc[-1], 'rmse': results['scores']['rmse'].iloc[-1],
                         'mae': results['scores']['mae'].iloc[-1]})

    all_results.append(results)
    losses.append(results['scores']['loss'].iloc[-1])
    mapes.append(results['scores']['mape'].iloc[-1])

# plot mapes
plt.figure()
plt.plot(mapes)
plt.ylabel('MAPE')
plt.xlabel('Model')
plt.title('MAPE for different models')
plt.savefig(os.path.join(out_dir, 'mapes.png'))
plt.show()

plt.figure()
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Model')
plt.title('Loss for different models')
plt.ylim(0, 1)
plt.savefig(os.path.join(out_dir, 'loss.png'))
plt.show()

plt.figure()
for i, r in enumerate(all_results):
    plt.plot(r['scores']['loss'], label=f'{i}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.title('Loss for different models')
plt.legend()
plt.savefig(os.path.join(out_dir, 'losses.png'))
plt.show()

last_results = pd.DataFrame(last_results)
last_results.to_csv(os.path.join(out_dir, 'last_results.csv'))
