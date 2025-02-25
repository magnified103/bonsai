import argparse
import json
import os

import numpy as np
import optuna
import ray
from category_encoders import TargetEncoder
from ray import tune
from ray.air import session, RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

import training.loader as loader
from bonsai.models.base import SimpleFFNNModel
from training.train_ffnn import train_network


def get_scheduler_kwargs(scheduler, _config):
    scheduler_kwargs = {}
    if scheduler == 'MultiStep':
        scheduler_kwargs['milestones'] = _config["scheduler"]["milestones"]
        scheduler_kwargs['gamma'] = _config["scheduler"]["multi_gamma"]
    elif scheduler == 'CosineAnnealing':
        scheduler_kwargs['T_max'] = _config["scheduler"]["T_max"]
    elif scheduler == 'CosineAnnealingWarmRestarts':
        scheduler_kwargs['T_0'] = _config["scheduler"]["T_0"]
        scheduler_kwargs['T_mult'] = _config["scheduler"]["T_mult"]
    elif scheduler == 'ExponentialLR':
        scheduler_kwargs['gamma'] = _config["scheduler"]["exp_gamma"]
    return scheduler_kwargs


def train_ffnn(config, max_epochs, data_file):
    _config = config.copy()
    del _config["layers"]

    data = loader.load(data_file, sample_size=10000, max_std=1)
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
    train, _ = data.train_test_split()
    train_subset, val = train.train_test_split()
    X = train_subset.get_features(features)
    y = train_subset.get_target(target)
    X_test = val.get_features(features)
    y_test = val.get_target(target)

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
    _config['batch_norm'] = [i for i in range(config['layers'])]
    net = SimpleFFNNModel(_config, 'tune_ffnn', transformer, out_transformer)

    for epoch in range(max_epochs):
        training_loss, loss, test_r2, test_rmse, test_mae, test_mape = train_network(net, (X, y),
                                                                                     (X_test, y_test))
        session.report(
            metrics={"training_loss": training_loss, "loss": loss, "mape": test_mape, "r2": test_r2, "rmse": test_rmse,
                     "mae": test_mae},
        )


def validate_ffnn(best_trained_model, test):
    return best_trained_model.score(test[0], test[1])


def write_best_config(df_sorted, output_dir='.', how_many=5):
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


def choose_scheduler_kwargs(scheduler):
    if scheduler == 'CosineAnnealingWarmRestarts':
        return tune.choice([{'T_0': i} for i in range(50, 100, 10)])
    elif scheduler == 'CosineAnnealing':
        return {'T_max': 100}
    elif scheduler == 'ExponentialLR':
        return tune.choice([{'gamma': g * 0.1} for g in range(1, 10)])
    elif scheduler == 'MultiStep':
        return tune.choice(
            [{'milestones': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'gamma': g * 0.1} for g in range(1, 10)])
    else:
        return {}  # default


def to_hyperopt(current_best_configs):
    configs = []
    for current_best_config in current_best_configs:
        hyperopt_config = {}
        for k, v in current_best_config.items():
            if str(v) != 'nan':
                tokens = k.split('/')
                if len(tokens) == 1:
                    hyperopt_config[k] = v
                elif len(tokens) == 2:
                    if tokens[0] not in hyperopt_config:
                        hyperopt_config[tokens[0]] = {}
                    hyperopt_config[tokens[0]][tokens[1]] = v
        configs.append(hyperopt_config)
    return configs


def define_optuna_search_space(trial: optuna.Trial):
    constant_params = {'epochs': 5, 'batch_size': 512}

    trial.suggest_categorical('network', ['relu', 'leaky_relu'])
    trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    scheduler = trial.suggest_categorical('scheduler',
                                          [None, 'CosineAnnealingWarmRestarts', 'CosineAnnealing', 'ExponentialLR',
                                           'MultiStep'])

    if scheduler == 'CosineAnnealingWarmRestarts':
        trial.suggest_int('T_0', 50, 100, 10)
        trial.suggest_int('T_mult', 1, 10, step=1)
    elif scheduler == 'CosineAnnealing':
        constant_params['T_max'] = 100
    elif scheduler == 'ExponentialLR':
        trial.suggest_float('gamma', 0.1, 1.0, step=0.1)
    elif scheduler == 'MultiStep':
        trial.suggest_float('gamma', 0.4, 0.9, step=0.1)
        constant_params['milestones'] = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    trial.suggest_categorical('init', ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'])
    layers = trial.suggest_int('layers', 10, 100, 10)
    for i in range(layers):
        trial.suggest_categorical('hidden_size_{}'.format(i), [2 ** i for i in range(6, 10)])
    trial.suggest_float('loss_param', 0.8, 1, step=0.1)

    return constant_params


def create_hidden_sizes(layers, sizes):
    first = []
    last = [sizes[np.random.randint(0, len(sizes))] for _ in range(len(first), layers)]
    return first + last


def create_hidden_sizes_blocks(layers, sizes):
    n_blocks = layers / len(sizes)
    encoder_block = sorted(sizes)
    decoder_block = sorted(sizes, reverse=True)
    hidden_sizes = []
    for i in range(int(n_blocks)):
        choice = np.random.choice([0, 1])
        if choice == 0:
            hidden_sizes += encoder_block
        else:
            hidden_sizes += decoder_block
    return hidden_sizes


def explore(num_trials=10, max_num_epochs=10, cpus_per_trial=2, gpus_per_trial=2,
            current_best_config=None, data_file=None, output_dir='.', how_many=5, metric='mape', mode='min',
            layers=None):
    print("Starting exploration with {} trials {} epochs {} cpus per trial {} gpus per_trial"
          .format(num_trials, max_num_epochs, cpus_per_trial, gpus_per_trial))

    data = loader.load(data_file)
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

    sizes = [2 ** i for i in range(6, 10)]

    config = {
        # 'network': tune.choice(['relu', 'leaky_relu']),
        'network': 'linear',
        'activation': 'relu',
        'input_size': len(features),
        'output_size': len(target),
        'epochs': 5,
        #'learning_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
         'learning_rate': 1e-3,
        'batch_size': 512,
        'optimizer': 'adam',
        # 'optimizer': tune.choice(['adam', 'rmsprop']),
        #'init': tune.choice(['xavier_uniform', 'xavier_normal']),
        'init': 'xavier_normal',
        'layers': tune.choice(np.arange(16, 81, 4)),
        #'layers': 20,
        # 'hidden_size': tune.sample_from(
        #     lambda spec: create_hidden_sizes(spec.config.layers, sizes)),
        'hidden_size': tune.sample_from(
            lambda spec: create_hidden_sizes_blocks(spec.__config.layers, sizes)),

         # 'scheduler': tune.choice(
         #     ['CosineAnnealing', 'CosineAnnealingWarmRestarts']),
        'scheduler': 'CosineAnnealing',
         # 'scheduler_kwargs': tune.sample_from(lambda spec: choose_scheduler_kwargs(spec.config.scheduler)),
        'scheduler_kwargs': {'T_max': 100},
        #'loss_param': tune.choice([0.7, 0.8, 0.9, 1.0]),
        # 'loss_param': 0,
        'loss': 'MSE',

    }

    # search algorithm to use
    # search algorithms try to find the best hyperparameters
    # Use HyperOpt to optimize hyperparameters
    # Allows to define conditional hyperparameters
    # Uses a Tree Parzen Estimator (TPE) to optimize hyperparameters
    # current_best_config = to_hyperopt(current_best_config)
    # print("Current best config: {}".format(current_best_config))
    # search_alg = HyperOptSearch(space=space, points_to_evaluate=current_best_config, metric=metric, mode=mode)

    # search_alg = OptunaSearch(define_optuna_search_space, metric=metric, mode=mode)

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
                             # callbacks=[
                             #     MLflowLoggerCallback(experiment_name=os.path.basename(output_dir)),
                             #     TBXLoggerCallback(),
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

    if args.ray_address is not None:
        ray.init(address=args.ray_address, log_to_driver=False)

    os.makedirs(args.output_dir, exist_ok=True)

    explore(num_trials=args.num_trials, max_num_epochs=args.max_num_epochs, cpus_per_trial=args.cpus_per_trial,
            gpus_per_trial=args.gpus_per_trial,
            current_best_config=current_best_config,
            data_file=args.data_file, output_dir=args.output_dir, how_many=args.how_many, layers=args.layers,
            metric='training_loss' )
