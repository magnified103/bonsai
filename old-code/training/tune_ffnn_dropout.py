import argparse
import json

import os

import numpy as np
import ray
from hyperopt import hp
import optuna
from ray import tune
from ray.air import session, RunConfig, CheckpointConfig
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import CLIReporter
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split

import training.loader as loader
from bonsai.models.base import SimpleFFNNModel, load_model
from training import trainable
from training.train_ffnn import train_network
from training.tune_ffnn import train_ffnn, write_best_config


def create_dropout_layers(hidden_size):
    dropout_layers = []
    for i in range(len(hidden_size)):
        if np.random.random() < 0.5:
            dropout_layers.append(i)
    return dropout_layers


def explore(num_trials=10, max_num_epochs=10, cpus_per_trial=2, gpus_per_trial=2,
            current_best_config=None, data_file=None, output_dir='.', how_many=5, metric='mape', mode='min',
            layers=None):
    print("Starting exploration with {} trials {} epochs {} cpus per trial {} gpus per_trial"
          .format(num_trials, max_num_epochs, cpus_per_trial, gpus_per_trial))

    train, test = loader.load_all_data(None, data_file=data_file)

    sizes = [2 ** i for i in range(6, 9)]

    config = {
        # 'network': tune.choice(['relu', 'leaky_relu']),
        'network': 'relu',
        'input_size': train[0].shape[1],
        'output_size': train[1].shape[1],
        'epochs': 5,
        # 'learning_rate': tune.choice([1e-4, 1e-3]),
        'learning_rate': 1e-3,
        'batch_size': 512,
        'transform': 'power',
        'optimizer': 'adam',
        # "'optimizer': tune.choice(['adam', 'rmsprop']),
        # 'init': tune.choice(['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']),
        'init': 'kaiming_uniform',
        'layers': 15,
        'hidden_size': [128, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 64, 64, 128, 128, 128, 128, 64],
        'scheduler': 'CosineAnnealing',
        'scheduler_kwargs': {'T_max': 100},
        'loss_param': 1.0,
        'dropout_rate': tune.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]),
        'dropout_layers': tune.sample_from(lambda spec: create_dropout_layers(spec.__config.hidden_size)),
    }

    # Asha uses a variant of hyperband to choose which trials to stop early
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=5,
        metric=metric,
        mode=mode,
        brackets=1,
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        sort_by_metric=True,
        metric_columns=["training_iteration", "time_total_s", "loss", "r2", "mape", "mae", "rmse"],
        metric=metric,
        mode=mode,
    )

    tuner = tune.Tuner(
        trainable.Trainable,
        tune_config=tune.TuneConfig(
            # metric=metric,
            # mode=mode,
            scheduler=scheduler,
            num_samples=num_trials,
            # search_alg=search_alg,
        ),
        run_config=RunConfig(
            stop={"training_iteration": max_num_epochs},
            progress_reporter=reporter,
                             local_dir='~/ray_results',
                             checkpoint_config=CheckpointConfig(
                                 num_to_keep=1,
                             ),
                             log_to_file=False,
                             verbose=1,
                             # callbacks=[
                             #     MLflowLoggerCallback(experiment_name=os.path.basename(output_dir)),
                             # ],
                             ),
        param_space=config,
    )
    results = tuner.fit()

    # the best trial is the one thas has the lowest loss and was trained for the longest
    best_result = results.get_best_result(metric, mode, "last")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation r2: {}".format(
        best_result.metrics["r2"]))
    print("Best trial final validation rmse: {}".format(
        best_result.metrics["rmse"]))
    print("Best trial final validation mae: {}".format(
        best_result.metrics["mae"]))
    print("Best trial final validation mape: {}".format(
        best_result.metrics["mape"]))

    # best_checkpoint_dir = best_result.checkpoint.to_directory()
    # print("Loading best model from checkpoint {}...".format(best_checkpoint_dir))
    # best_trained_model = load_model(os.path.join(
    #     best_checkpoint_dir, "model"))
    #
    # _, loss, r2, rmse, mae, mape = validate_ffnn(best_trained_model, test)
    # print("Best trial test set loss: {}, r2: {}, rmse: {}, mae: {}, mape: {}".format(loss, r2, rmse, mae, mape))
    df_results = results.get_dataframe()
    df_sorted = df_results.sort_values(by=[metric])
    print(df_sorted)
    os.makedirs(output_dir, exist_ok=True)
    df_sorted.to_csv(os.path.join(output_dir, f'results_ffnn.csv'), index=False)
    write_best_config(df_sorted, output_dir, how_many)


parser = argparse.ArgumentParser()
parser.add_argument('--cpus_per_trial', type=float, default=1)
parser.add_argument('--gpus_per_trial', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--max_num_epochs', type=int, default=20)
parser.add_argument('--current_best_config_path', type=str, default=None)
parser.add_argument('--data_file', type=str, default='data/small_sample_data.csv.gz')
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--ray_address', type=str, default=None)
parser.add_argument('--how_many', type=int, default=5)
parser.add_argument('--layers', type=int, nargs='+', default=None)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    args = parser.parse_args()
    current_best_config = None
    if args.current_best_config_path is not None:
        with open(args.current_best_config_path, 'r') as f:
            current_best_config = json.load(f)

    # if args.ray_address is not None:
    #     ray.init(address=args.ray_address, log_to_driver=False)

    os.makedirs(args.output_dir, exist_ok=True)

    explore(num_trials=args.num_trials, max_num_epochs=args.max_num_epochs, cpus_per_trial=args.cpus_per_trial,
            gpus_per_trial=args.gpus_per_trial,
            current_best_config=current_best_config,
            data_file=args.data_file, output_dir=args.output_dir, how_many=args.how_many, layers=args.layers,
            metric='loss')
