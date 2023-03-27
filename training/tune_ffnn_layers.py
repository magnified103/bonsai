import argparse
import json
import os

import numpy as np
import ray
from hyperopt import hp
from ray import tune
from ray.air import session, RunConfig, CheckpointConfig
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split

import training.loader as loader
from network_generator.models.base import SimpleFFNNModel, load_model
from training.train_ffnn import train_network


def train_ffnn(config, max_epochs, data_file):
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        raise NotImplementedError("Loading from checkpoint not implemented")
        # with loaded_checkpoint.as_directory() as checkpoint_dir:
        #     net = load_model(os.path.join(checkpoint_dir, "model"))
    else:
        _config = config.copy()
        layers = _config['layers']
        hidden_size = [_config[f"layers_{layers}_l{i}"] for i in range(layers)]
        _config['hidden_size'] = hidden_size
        del _config['layers']
        for i in range(layers):
            del _config[f"layers_{layers}_l{i}"]

        net = SimpleFFNNModel(_config)

    train, test = loader.load_all_data(None, data_file)
    trainset = np.concatenate((train[0], train[1]), axis=1)
    train_subset, val_subset = train_test_split(
        trainset, test_size=0.2, random_state=42)

    train_X, train_y = np.split(train_subset, [train_subset.shape[1] - 1], axis=1)
    val_X, val_y = np.split(val_subset, [val_subset.shape[1] - 1], axis=1)

    for epoch in range(max_epochs):
        _, loss, test_r2, test_rmse, test_mae, test_mape = train_network(net, (train_X, train_y), (val_X, val_y))

        # checkpoint_dir = f"checkpoint-{epoch}"
        # net.save(os.path.join(checkpoint_dir, "model"))
        session.report(
            metrics={"loss": loss, "mape": test_mape, "r2": test_r2, "rmse": test_rmse, "mae": test_mae},
            # checkpoint=Checkpoint.from_directory(checkpoint_dir),
        )


def validate_ffnn(best_trained_model, test):
    return best_trained_model.score(test[0], test[1])


def write_best_config(df_sorted, output_dir='.', how_many=5):
    # best = df_sorted[:5][
    #     ['config/batch_size', 'config/epochs', 'config/hidden_size', 'config/input_size', 'config/learning_rate',
    #      'config/network', 'config/output_size']].rename(
    #     columns={'config/batch_size': 'batch_size', 'config/epochs': 'epochs', 'config/hidden_size': 'hidden_size',
    #              'config/input_size': 'input_size', 'config/learning_rate': 'learning_rate',
    #              'config/network': 'network', 'config/output_size': 'output_size'})
    config_cols = [col for col in df_sorted.columns if col.startswith('config/')]
    rename_cols = {col: col.replace('config/', '') for col in config_cols}
    best = df_sorted[:how_many][config_cols].rename(
        columns=rename_cols)

    best_config = []
    for row in best.iterrows():
        config = row[1].to_dict()
        best_config.append(config)

    import json
    best_config_file = f'best_config_ffnn.json'
    json_obj = json.dumps(best_config, indent=4)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, best_config_file), 'w') as f:
        f.write(json_obj)


def explore(num_trials=10, max_num_epochs=10, cpus_per_trial=2, gpus_per_trial=2,
            current_best_config=None, data_file=None, output_dir='.', how_many=5, metric='mape', mode='min', ray_cluster_address=None):
    print("Starting exploration with {} trials {} epochs {} cpus per trial {} gpus per_trial"
          .format(num_trials, max_num_epochs, cpus_per_trial, gpus_per_trial))

    train, test = loader.load_all_data(None, data_file=data_file)

    sizes = [2 ** i for i in range(6, 10)]
    top_output_dir = output_dir
    for layers in range(3, 16):
        ray.init(address=ray_cluster_address, log_to_driver=False)
        output_dir = os.path.join(top_output_dir, 'layers_{}'.format(layers))
        config = {
            'network': 'relu',
            'input_size': train[0].shape[1],
            'output_size': train[1].shape[1],
            'epochs': 5,
            'batch_size': 512,
            'transform': 'power',
            'optimizer': 'adam',
            'init': 'xavier',
            'loss_param': 0.5,
            'layers': layers,
        }


        for l in range(layers):
            config['layers_{}_l{}'.format(layers, l)] = tune.grid_search(sizes)

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
        resource_group = tune.PlacementGroupFactory([{"CPU": cpus_per_trial, "GPU": gpus_per_trial}])
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_ffnn, max_epochs=max_num_epochs, data_file=data_file),
                resources=resource_group),
            tune_config=tune.TuneConfig(
                # metric=metric,
                # mode=mode,
                scheduler=scheduler,
                num_samples=num_trials,
                # search_alg=search_alg,
            ),
            run_config=RunConfig(progress_reporter=reporter,
                                 local_dir='~/ray_results',
                                 checkpoint_config=CheckpointConfig(
                                     num_to_keep=1,
                                 ),
                                 log_to_file=False,
                                 verbose=1,
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

        df_results = results.get_dataframe()
        df_sorted = df_results.sort_values(by=[metric])
        print(df_sorted)
        os.makedirs(output_dir, exist_ok=True)
        df_sorted.to_csv(os.path.join(output_dir, f'results_ffnn.csv'), index=False)
        write_best_config(df_sorted, output_dir, how_many)
        ray.shutdown()


parser = argparse.ArgumentParser()
parser.add_argument('--cpus_per_trial', type=float, default=1)
parser.add_argument('--gpus_per_trial', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--max_num_epochs', type=int, default=10)
parser.add_argument('--current_best_config_path', type=str, default=None)
parser.add_argument('--data_file', type=str, default='data/small_sample_data.csv.gz')
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--ray_address', type=str, default=None)
parser.add_argument('--how_many', type=int, default=5)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    args = parser.parse_args()
    current_best_config = None
    if args.current_best_config_path is not None:
        with open(args.current_best_config_path, 'r') as f:
            current_best_config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    explore(num_trials=args.num_trials, max_num_epochs=args.max_num_epochs, cpus_per_trial=args.cpus_per_trial,
            gpus_per_trial=args.gpus_per_trial,
            current_best_config=current_best_config,
            data_file=args.data_file, output_dir=args.output_dir, how_many=args.how_many, ray_cluster_address=args.ray_address)
