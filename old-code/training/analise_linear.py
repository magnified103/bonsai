import itertools
import os

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures

from analysis.analyse_results import plot_predictions, plot_cdf, plot_loss_vs_training_loss, set_out_dir, plot_error
from bonsai import models
from training.loader import load_all_data, load

data = load('data/aggPingsClean.csv', sample_size=15000, max_std=1)

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

out_dir = '/Users/akos/Documents/Projects/bonsai/training/results/linear'
os.makedirs(out_dir, exist_ok=True)

set_out_dir(out_dir)

mapes = []
training_losses = []
losses = []
rmses = []
r2s = []
maes = []
last_results = []
all_results = []


constant_config = {'lambd': 0.0}

configs = [{}]

for config in configs:
    config = {**constant_config, **config}
    print(f'------- Testing {config["lambd"]} -------')
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
    _config = config.copy()

    model = models.SimpleLinearModel(_config, 'simple_linear', transformer, out_transformer)
    model.fit(X, y)
    training_loss, _, _, _, _ = model.score(X, y)
    loss, r2, rmse, mae, mape = model.score(X_test, y_test)

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
                'scores': pd.DataFrame([{'loss': loss, 'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}]),
               'name': f"Linear lamdb: {config['lambd']}"}

    #plot_loss_vs_training_loss(results)
    set_out_dir(os.path.join(out_dir, f"{results['name']}"))
    plot_cdf(results)
    plot_error(results)
    plot_predictions(results)
    model.save(os.path.join(out_dir, f"{results['name']}, model.pth"))

    training_losses.append(training_loss)
    losses.append(results['scores']['loss'].iloc[-1])
    mapes.append(results['scores']['mape'].iloc[-1])
    rmses.append(results['scores']['rmse'].iloc[-1])
    r2s.append(results['scores']['r2'].iloc[-1])
    maes.append(results['scores']['mae'].iloc[-1])

# plot mapes
# plt.figure()
# plt.plot(mapes, '.-')
# plt.ylabel('MAPE')
# plt.xlabel('Model')
# plt.xscale("log")
# plt.title('MAPE for different models')
# plt.savefig(os.path.join(out_dir, 'mapes.png'))
# plt.show()
# #
# plt.figure()
# plt.plot(maes, '.-')
# plt.ylabel('MAE')
# plt.xlabel('Model')
# plt.xscale("log")
# plt.title('MAE for different models')
# plt.savefig(os.path.join(out_dir, 'mae.png'))
# plt.show()
# #
# plt.figure()
# plt.plot(r2s, '.-')
# plt.ylabel('R2')
# plt.xlabel('Model')
# plt.xscale("log")
# plt.title('R2 for different models')
# plt.savefig(os.path.join(out_dir, 'r2.png'))
# plt.show()
# #
# plt.figure()
# plt.plot(rmses, '.-')
# plt.ylabel('RMSE')
# plt.xlabel('Model')
# plt.xscale("log")
# plt.title('RMSE for different models')
# plt.savefig(os.path.join(out_dir, 'rmse.png'))
# plt.show()
# #
# print('Training losses')
# print(training_losses)
# print('Losses')
# print(losses)
# plt.figure()
# plt.plot(training_losses, label='training loss')
# plt.plot(losses, label='loss')
# plt.ylabel('Loss')
# plt.xlabel('Model')
# plt.title('Loss for different models')
# plt.xscale("log")
# plt.legend()
# #plt.ylim(0, 1)
# plt.savefig(os.path.join(out_dir, 'loss.png'))
# plt.show()
#
# plt.figure()
# for i, r in enumerate(all_results):
#     plt.plot(r['scores']['loss'], label=f'{i}_{r["name"]}')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.ylim(0, 1)
# plt.title('Loss for different models')
# plt.legend()
# plt.savefig(os.path.join(out_dir, 'losses.png'))
# plt.show()
#
# last_results = pd.DataFrame(last_results)
# last_results.to_csv(os.path.join(out_dir, 'last_results.csv'))
