import itertools
import os

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from analysis.analyse_results import plot_predictions, plot_cdf, plot_loss_vs_training_loss, set_out_dir, plot_error
from bonsai import models
from bonsai.models.FFNN.loss import NegativeSlopePenalty, NegativesPenalty
from training.loader import load_all_data, load

data = load('data/aggPingsClean.csv', sample_size=2000, max_std=1)

featureChooser = data.choose_features()
features = featureChooser.choose_features([
    'source_country',
    'dest_country',
    'source_asn',
    'dest_asn',
    'source_lat',
    'source_lon',
    'dest_lat',
    'dest_lon',
    'distance',
])
target = featureChooser.choose_targets(['rtt_p50'])
train, test = data.train_test_split()
X = train.get_features(features)
y = train.get_target(target)
X_test = test.get_features(features)
y_test = test.get_target(target)

epochs = 20

out_dir = '/Users/akos/Documents/Projects/bonsai/training/results/ffnn'
os.makedirs(out_dir, exist_ok=True)

set_out_dir(out_dir)

mapes = []
r2s = []
rmses = []
maes = []

losses = []
last_results = []
all_results = []

constant_config = {'input_size': X.shape[1], 'output_size': y.shape[1],
                   'batch_size': 512,
                   'init': 'kaiming_uniform',
                   'epochs': epochs,
                   'scheduler': 'CosineAnnealing',
                   'scheduler_kwargs': {'T_max': epochs * 5},
                   'activation': 'relu',
                   # 'loss': 'MAPE',
                   'loss_penalty': NegativesPenalty(1),
                   'learning_rate': 0.001,
                   'optimizer': 'adam',
                   'dropout_rate': 0.2,
                   }

hidden_sizes = [
    [8, 6, 4, 2],
    [16, 16, 16, 16],
    [10, 12, 14, 16],
    [10, 12, 14, 16, 10, 8, 6, 4, 2],
]

#hidden_size = [512, 64, 64, 512, 256]
# dropout_layers = None
# batch_norm = None
#dropout_layers = [i for i in range(0, len(hidden_size))]
#batch_norm = [i for i in range(0, len(hidden_size))]
# res_hidden_size = [
#     {'hidden_size': [256, 128, 64], 'batch_norm': [i for i in range(3)]},
#     {'hidden_size': [64, 32, 16], 'batch_norm': [i for i in range(3)]},
#     #{'hidden_size': [256 for _ in range(3)], 'batch_norm': [i for i in range(3)]},
# ]
#
configs = [
    # {'network': 'linear_offset', 'hidden_size': hidden_size},
    #{'network': 'linear', 'hidden_size': hidden_size, },
    #{'network': 'linear', 'hidden_size': [512, 64, 64, 64, 512, 256], },
    #{'network': 'linear', 'hidden_size': [512, 64, 64, 512, 256, 128], },
    # {'network': 'linear', 'hidden_size': hidden_size, 'batch_norm':None },
    # {'network': 'linear', 'hidden_size': hidden_size, 'dropout_layers':None },
    # {'network': 'linear', 'hidden_size': hidden_size, 'batch_norm': None, 'dropout_layers': None },
    # {'network': 'residual_offset', 'hidden_size': res_hidden_size, 'method': 'add'},
    # {'network': 'residual', 'hidden_size': res_hidden_size, 'method': 'add'},
    # {'network': 'residual_offset', 'hidden_size': res_hidden_size, 'method': 'concat'},
    # {'network': 'residual', 'hidden_size': res_hidden_size, 'method': 'concat'},
]

for hidden_size in hidden_sizes:
    configs.append({'network': 'linear', 'hidden_size': hidden_size})

all_configs = []
for loss in ['MSE']:
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

    transformer = ColumnTransformer([
        # ('standard', StandardScaler(), features.get_feature_indices(['source_lat',
        #                                                              'source_lon',
        #                                                              'dest_lat',
        #                                                              'dest_lon'])
        #  ),
        ('power', PowerTransformer(), features.get_feature_indices(['distance',
                                                                    'source_lat',
                                                                    'source_lon',
                                                                    'dest_lat',
                                                                    'dest_lon'
                                                                    ])
         ),
        ('target', TargetEncoder(), features.get_feature_indices(['source_country',
                                                                  'dest_country',
                                                                  'source_asn',
                                                                  'dest_asn'])
         )
    ])
    out_transformer = PowerTransformer()
    model = models.SimpleFFNNModel(config, 'test_ffnn', transformer, out_transformer)

    scores = []
    for i in range(epochs):
        model.fit(X, y)
        training_loss, _, _, _, _ = model.score(X, y)
        loss, r2, rmse, mae, mape = model.score(X_test, y_test)
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
               'name': f"net_{config['hidden_size']}_act_{config['network']}_{config['loss']}_{config['loss_penalty']}"}

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
    r2s.append(results['scores']['r2'].iloc[-1])
    rmses.append(results['scores']['rmse'].iloc[-1])
    maes.append(results['scores']['mae'].iloc[-1])

#plot mapes
plt.figure()
plt.plot(mapes)
plt.ylabel('MAPE')
plt.xlabel('Model')
plt.title('MAPE for different models')
plt.savefig(os.path.join(out_dir, 'mapes.png'))
plt.show()
#plot rmse
plt.figure()
plt.plot(rmses)
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.title('RMSE for different models')
plt.savefig(os.path.join(out_dir, 'rmse.png'))
plt.show()
#plot re
plt.figure()
plt.plot(r2s)
plt.ylabel('R2')
plt.xlabel('Model')
plt.title('R2 for different models')
plt.savefig(os.path.join(out_dir, 'r2.png'))
plt.show()
#plot mae
plt.figure()
plt.plot(maes)
plt.ylabel('MAE')
plt.xlabel('Model')
plt.title('MAE for different models')
plt.savefig(os.path.join(out_dir, 'mae.png'))
plt.show()
# plt.figure()
# plt.plot(losses)
# plt.ylabel('Loss')
# plt.xlabel('Model')
# plt.title('Loss for different models')
# plt.ylim(0, 1)
# plt.savefig(os.path.join(out_dir, 'loss.png'))
# plt.show()
#
plt.figure()
for i, r in enumerate(all_results):
    plt.plot(r['scores']['loss'], label=f'{i}_{r["name"]}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.title('Loss for different models')
plt.legend()
plt.savefig(os.path.join(out_dir, 'losses.png'))
plt.show()
#
# last_results = pd.DataFrame(last_results)
# last_results.to_csv(os.path.join(out_dir, 'last_results.csv'))
